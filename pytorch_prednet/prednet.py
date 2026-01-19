from collections import defaultdict
from typing import Literal
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from convlstmcell import ConvLSTMCell


class SatLU(nn.Module):
    def __init__(self, lower=0, upper=1, inplace=False):
        super(SatLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input):
        return F.hardtanh(input, self.lower, self.upper, self.inplace)


    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' ('\
            + 'min_val=' + str(self.lower) \
	        + ', max_val=' + str(self.upper) \
	        + inplace_str + ')'


class AddSin(nn.Module):
    def __init__(self, amp=0.0, omg=1.0):
        super(AddSin, self).__init__()
        self._time = 0
        self._amp = amp
        self._omg = omg

    def forward(self, input):
        output = input + self._amp * math.sin(self._omg * self._time)
        self._time += 1
        return output


class PredNet(nn.Module):
    def __init__(self, A_channels, R_channels=None,
                 output_mode='error',
                 round_mode="down_up_down",
                 diff_mode: Literal["pos_neg", "pos", "neg"] = "pos_neg",
                 device=torch.device("cpu"),
                 amp=0.0,
                 omg=1.0,
    ):
        super(PredNet, self).__init__()
        self.device = device
        if R_channels is None:
            R_channels = A_channels
        self.r_channels = R_channels + [0,]  # for convenience
        self.a_channels = A_channels
        self.n_layers = len(R_channels)
        self.output_mode = output_mode
        self.round_mode = round_mode
        self.diff_mode = diff_mode
        self.loss = nn.MSELoss(reduction="none")
        self.outputs = defaultdict(list)

        default_output_modes = ['prediction', 'error']
        assert output_mode in default_output_modes, 'Invalid output_mode: ' + str(output_mode)

        for i in range(self.n_layers):
            cell = ConvLSTMCell(2 * self.a_channels[i] + self.r_channels[i + 1], self.r_channels[i],
                                (3, 3), amp=amp, omg=omg)
            setattr(self, 'cell{}'.format(i), cell)
            self.register_hooks(cell, "ConvLSTMCell_layer{}".format(i))

        for i in range(self.n_layers):
            conv = nn.Sequential(nn.Conv2d(self.r_channels[i], self.a_channels[i], 3, padding=1), AddSin(amp, omg), nn.ReLU())
            if i == 0:
                conv.add_module('satlu', SatLU())
            setattr(self, 'conv{}'.format(i), conv)
            self.register_hooks(conv, "Conv_sequential_layer{}".format(i))


        self.upsample = nn.Upsample(scale_factor=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        for l in range(self.n_layers - 1):
            update_A = nn.Sequential(nn.Conv2d(2 * self.a_channels[l], self.a_channels[l + 1], (3, 3), padding=1), AddSin(amp, omg), self.maxpool)
            setattr(self, 'update_A{}'.format(l), update_A)
            self.register_hooks(update_A, "UpdateA_layer{}".format(l))

        self.reset_parameters()

    def register_hooks(self, mod, name):
        def hook_fn(m, input, output):
            if isinstance(output[0], torch.Tensor):
                self.outputs[name].append(output[0].detach().cpu())
            elif isinstance(output[0], tuple):
                self.outputs[name].append([o.detach().cpu() for o in output[0]])
        mod.register_forward_hook(hook_fn)

    def reset_parameters(self):
        for l in range(self.n_layers):
            cell = getattr(self, 'cell{}'.format(l))
            cell.reset_parameters()

    def down_to_up(self, A, Ahat_seq, E_seq):
        # down -> up
        for l in range(self.n_layers):
            pos = F.relu(Ahat_seq[l] - A) if self.diff_mode in ["pos_neg", "pos"] else torch.zeros_like(Ahat_seq[l])
            neg = F.relu(A - Ahat_seq[l]) if self.diff_mode in ["pos_neg", "neg"] else torch.zeros_like(Ahat_seq[l])
            E = torch.cat([pos, neg], 1)
            E_seq[l] = E
            if l < self.n_layers - 1:
                update_A = getattr(self, 'update_A{}'.format(l))
                A = update_A(E)

    def up_to_down(self, t, Ahat_seq, E_seq, R_seq, H_seq, total_loss, gt):
        for l in reversed(range(self.n_layers)):
            cell = getattr(self, 'cell{}'.format(l))
            if t == 0:
                E = E_seq[l]
                R = R_seq[l]
                hx = (R, R)
            else:
                E = E_seq[l]
                R = R_seq[l]
                hx = H_seq[l]
            if l == self.n_layers - 1:
                R, hx = cell(E, hx)
            else:
                tmp = torch.cat((E, self.upsample(R_seq[l + 1])), 1)
                R, hx = cell(tmp, hx)
            R_seq[l] = R
            H_seq[l] = hx
            conv = getattr(self, 'conv{}'.format(l))
            Ahat_seq[l] = conv(R_seq[l])
            if l == 0:
                frame_prediction = Ahat_seq[l]
                total_loss += self.loss(frame_prediction, gt).mean(axis=(1, 2, 3))
        return frame_prediction

    def forward(self, input):
        self.outputs = defaultdict(list)
        R_seq = [None] * self.n_layers
        H_seq = [None] * self.n_layers
        E_seq = [None] * self.n_layers
        Ahat_seq = [None] * self.n_layers

        w, h = input.size(-2), input.size(-1)
        batch_size = input.size(0)

        for l in range(self.n_layers):
            E_seq[l] = torch.zeros(batch_size, 2 * self.a_channels[l], w, h).to(self.device)
            R_seq[l] = torch.zeros(batch_size, self.r_channels[l], w, h).to(self.device)
            Ahat_seq[l] = torch.zeros(batch_size, self.a_channels[l], w, h).to(self.device)
            w = w // 2
            h = h // 2

        time_steps = input.size(1)
        total_loss = torch.zeros(batch_size).to(self.device)
        eval_index = [torch.zeros(batch_size).to(self.device) for _ in range(self.n_layers)]
        for t in range(time_steps - 1):
            A = input[:, t]
            gt = input[:, t + 1]

            if self.round_mode == "down_up_down":
                self.down_to_up(A, Ahat_seq, E_seq)
                frame_prediction = self.up_to_down(t, Ahat_seq, E_seq, R_seq, H_seq, total_loss, gt)
            else:
                frame_prediction = self.up_to_down(t, Ahat_seq, E_seq, R_seq, H_seq, total_loss, gt)
                self.down_to_up(A, Ahat_seq, E_seq)

        # calculate eval index
        A = input[:, t + 1]
        for l in range(self.n_layers):
            eval_index[l] = self.loss(Ahat_seq[l], A).mean(axis=(1, 2, 3))
            pos = F.relu(Ahat_seq[l] - A) if self.diff_mode in ["pos_neg", "pos"] else torch.zeros_like(Ahat_seq[l])
            neg = F.relu(A - Ahat_seq[l]) if self.diff_mode in ["pos_neg", "neg"] else torch.zeros_like(Ahat_seq[l])
            E = torch.cat([pos, neg], 1)
            E_seq[l] = E
            if l < self.n_layers - 1:
                update_A = getattr(self, 'update_A{}'.format(l))
                A = update_A(E)

        if self.output_mode == 'error':
            return frame_prediction, total_loss, eval_index
        else:
            return frame_prediction
