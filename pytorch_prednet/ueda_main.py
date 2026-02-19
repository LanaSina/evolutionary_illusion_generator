import os
import numpy as np
from PIL import Image, ImageCms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
import prednet
import argparse
from tqdm import tqdm
from distutils.util import strtobool
from dataset import ImageListDataset, ImageHDF5Dataset
from corr_wise import CorrWise

parser = argparse.ArgumentParser(description='PredNet')
parser.add_argument('--images', '-i', default='data/train_list.txt', help='Path to image list file')
parser.add_argument('--sequences', '-seq', default='', help='Path to sequence list file')
parser.add_argument('--device', '-d', default="", type=str,
                    help='Computational device')
parser.add_argument('--root', '-r', default='.',
                    help='Root directory path of sequence and image files')
parser.add_argument('--initmodel', default='',
                    help='Initialize the model from given file')
parser.add_argument('--size', '-s', default='160,120',
                    help='Size of target images. width,height (pixels)')
parser.add_argument('--channels', '-c', default='3,48,96,192',
                    help='Number of channels on each layers')
parser.add_argument('--offset', '-o', default='0,0',
                    help='Center offset of clipping input image (pixels)')
parser.add_argument('--input_len', '-l', default=20, type=int,
                    help='Input frame length fo extended prediction on test (frames)')
parser.add_argument('--ext', '-e', default=10, type=int,
                    help='Extended prediction on test (frames)')
parser.add_argument('--bprop', default=20, type=int,
                    help='Back propagation length (frames)')
parser.add_argument('--save', default=100000, type=int,
                    help='Period of save model and state (frames)')
parser.add_argument('--period', default=5000000, type=int,
                    help='Period of training (frames)')
parser.add_argument('--test', dest='test', action='store_true')
parser.add_argument('--saveimg', dest='saveimg', action='store_true')
parser.add_argument('--useamp', dest='useamp', action='store_true', help='Flag for using AMP')
parser.add_argument('--lr', default=0.001, type=float,
                    help='Learning rate')
parser.add_argument('--lr_rate', default=1.0, type=float,
                    help='Reduction rate for Step lr scheduler')
parser.add_argument('--min_lr', default=0.0001, type=float,
                    help='Lower bound learning rate for Step lr scheduler')
parser.add_argument('--batchsize', default=1, type=int, help='Input batch size')
parser.add_argument('--shuffle', default=False, type=strtobool, help=' True is enable to sampl data randomly (default: False)')
parser.add_argument('--num_workers', default=0, type=int, help='Num. of dataloader process. (default: num of cpu cores')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='True is enable to log for Tensorboard')
parser.add_argument('--up_down_up', action='store_true', help='True is enable to cycle up-down-up in order')
parser.add_argument('--color_space', default='RGB', type=str, help='Image color space(RGB, HSV, LAB, CMYK, YcbCr) - the dimension of this color space and 1st channel must be same.')
parser.add_argument('--loss', type=str, default='mse', help='Loss name for training. Please select loss from "mse", "corr_wise", and "ensemble" (default: mse).')
parser.add_argument('--amp', default=0.0, type=float, help='Amplitude for sine function')
parser.add_argument('--omg', default=1.0, type=float, help='Angular velocity for sine function')
parser.add_argument('--num_max_window', default=None, type=int, help='Number of max windows for flow estimation')
parser.add_argument('--diff_mode', default="pos_neg", type=str, help='Diff mode for update_A. Please select diff_mode from "pos_neg", "pos", and "neg" (default: pos_neg).')
parser.set_defaults(test=False)
args = parser.parse_args()


def load_list(path, root):
    tuples = []
    for line in open(path):
        pair = line.strip().split(",")
        tuples.append([os.path.join(root, p) for p in pair])
    return tuples


srgb_p = ImageCms.createProfile("sRGB")
lab_p  = ImageCms.createProfile("LAB")
lab2rgb = ImageCms.buildTransformFromOpenProfiles(lab_p, srgb_p, "LAB", "RGB")

def write_image(image, path, mode="img", c_space="RGB"):
    if mode == "img":
        img = image * 255
        img = img.transpose(1, 2, 0)
        img = img.astype(np.uint8)
        h, w, ch = img.shape
        if c_space == "RGB":
            if ch == 1:
                img = img.reshape((h, w))
                result = Image.fromarray(img)
                result.save(path + ".jpg")
            elif ch == 4:
                img_gray = img[:, :, 3]
                img_color = img[:, :, :3]
                result_gray = Image.fromarray(img_gray)
                result_color = Image.fromarray(img_color)
                result_gray.save(path + "_gray.jpg")
                result_color.save(path + "_color.jpg")
            else:
                result = Image.fromarray(img)
                result.save(path + ".jpg")
        elif c_space == "LAB":
            tmp = Image.fromarray(img, mode="LAB")
            result = ImageCms.applyTransform(tmp, lab2rgb)
            result.save(path + ".jpg")
        else:
            result = Image.fromarray(img, mode=c_space).convert("RGB")
            result.save(path + ".jpg")
    else:
        np.save(path + ".npy", image)


def write_outputs(writer, outputs, count, prefix=""):
    for k, v in outputs.items():
        for i, vv in enumerate(v):
            if isinstance(vv, torch.Tensor):
                if len(vv.shape) == 3:
                    vv = vv.unsqueeze(axis=1)
                elif len(vv.shape) == 4:
                    vv = vv.reshape((-1, vv.shape[2], vv.shape[3]))
                    vv = vv.unsqueeze(axis=1)
                x = vutils.make_grid(vv, normalize=True, scale_each=True)
                if prefix == "":
                    writer.add_image(k + "_time{}".format(i), x, count)
                else:
                    writer.add_image(prefix + "/" + k + "_time{}".format(i), x, count)


def train(device=torch.device("cpu")):
    if args.sequences == '':
        sequencelist = [[args.images]]
    else:
        sequencelist = load_list(args.sequences, args.root)

    logf = open('log_t.txt', 'w')
    writer = SummaryWriter() if args.tensorboard else None
    time_loss_weights = 1./(args.bprop - 1) * torch.ones(args.bprop, 1)
    time_loss_weights[0] = 0
    time_loss_weights = time_loss_weights.to(device)
    net = prednet.PredNet(
        args.channels,
        round_mode="up_donw_up" if args.up_down_up else "down_up_down",
        diff_mode=args.diff_mode,
        device=device,
        amp=args.amp,
        omg=args.omg,
    ).to(device)

    base_loss = nn.L1Loss()
    loss = CorrWise(base_loss, flow_method="FBFlow", return_warped=False, reduction_clip=False, flow_cycle_loss=True, scale_clip=True, device=device)
    if args.initmodel:
        print('Load model from', args.initmodel)
        net.load_state_dict(torch.load(args.initmodel))
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    print("Automatic Mixed Precision: {}".format(args.useamp))
    scaler = torch.amp.GradScaler('cuda', enabled=args.useamp)#scaler = torch.cuda.amp.GradScaler(enabled=args.useamp)
    count = 0
    seq = 0
    lr_maker = lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=args.lr_rate)

    # create a dataset from initial imagelist
    if sequencelist[seq][0].endswith(('.h5', '.hdf5')):
        img_dataset = ImageHDF5Dataset(img_size=(args.size[0],args.size[1]),
                                       input_len=args.bprop, channels=args.channels[0])
        img_dataset.load_hdf5(sequencelist[seq][0], c_space=args.color_space)
        imagelist = list(img_dataset.hf_data.keys())
    else:
        imagelist = load_list(sequencelist[seq][0], args.root)
        img_dataset = ImageListDataset(img_size=(args.size[0],args.size[1]),
                                       input_len=args.bprop, channels=args.channels[0])
        img_dataset.load_images(img_paths=imagelist, c_space=args.color_space)
    # data loader
    data_loader = DataLoader(img_dataset, batch_size=args.batchsize, shuffle=args.shuffle, num_workers=args.num_workers)
    print("shuffle: ", args.shuffle)
    print("num_workers: ", args.num_workers)
    while count <= args.period:
        print("seqNo: {}".format(seq))
        if seq > 0:
            if sequencelist[seq][0].endswith(('.h5', '.hdf5')):
                img_dataset = ImageHDF5Dataset(img_size=(args.size[0],args.size[1]),
                                               input_len=args.bprop, channels=args.channels[0])
                img_dataset.load_hdf5(sequencelist[seq][0], c_space=args.color_space)
                imagelist = list(img_dataset.hf_data.keys())
            else:
                imagelist = load_list(sequencelist[seq][0], args.root) 
                # update dataset and loader 
                img_dataset = ImageListDataset(img_size=(args.size[0],args.size[1]),
                                        input_len=args.bprop, channels=args.channels[0])
                img_dataset.load_images(img_paths=imagelist, c_space=args.color_space)
            data_loader = DataLoader(img_dataset, batch_size=args.batchsize, shuffle=args.shuffle, num_workers=args.num_workers)

        if len(imagelist) == 0:
            print("Not found images.")
            return
        fn = 0
        for data in tqdm(data_loader, unit="batch"):
            print("frameNo: {}".format(fn))
            print("total frames: {}".format(count))
            with torch.amp.autocast('cuda', enabled=args.useamp):#with torch.cuda.amp.autocast(enabled=args.useamp):
                data = data.to(device)
                pred, errors, _ = net(data)
                if args.loss == 'corr_wise':
                    mean_error= loss(pred, data[:, -1])
                elif args.loss == 'ensemble':
                    corr_wise_error = loss(pred, data[:, -1])
                    mean_error = corr_wise_error + errors.mean()
                else:
                    mean_error = errors.mean()
                
            optimizer.zero_grad()
            scaler.scale(mean_error).backward()
            scaler.step(optimizer)
            scaler.update()
            if lr_maker.get_last_lr()[0] > args.min_lr:
                lr_maker.step()
            else:
                lr_maker.optimizer.param_groups[0]['lr'] = args.min_lr
            if args.saveimg:
                for j in range(len(data)):
                    write_image(data[j, -1].detach().cpu().numpy(), 'result/' + str(count) + '_' + str(fn / args.input_len + j) + 'x',
                                img_dataset.mode, args.color_space)
                    write_image(pred[j].detach().cpu().numpy(), 'result/' + str(count) + '_' + str(fn / args.input_len + j) + 'y',
                                img_dataset.mode, args.color_space)
            print("loss: ", mean_error.detach().cpu().numpy())
            logf.write(str(count) + ', ' + str(mean_error.detach().cpu().numpy()) + '\n')
            if writer is not None:
                writer.add_scalar("loss", mean_error.detach().cpu().numpy(), count)

            if count % args.save < len(data) * args.input_len:
                print("Save the model")
                torch.save(net.state_dict(), os.path.join("models", str(count) + ".pth"))
                if writer is not None:
                    for name, param in net.named_parameters():
                        writer.add_histogram(name, param.clone().cpu().data.numpy(), count)
                    write_outputs(writer, net.outputs, count)

            if count > args.period:
                break
            count += len(data) * args.input_len
            fn += len(data) * args.input_len
        seq = (seq + 1) % len(sequencelist)

    if writer is not None:
        print("Save tensorboard graph...")
        dummy_input = torch.zeros((1, 2, 3, args.size[0], args.size[1])).to(device)
        net.output_mode = 'prediction'
        writer.add_graph(net, dummy_input)
        writer.close()
    dot = make_dot(pred, params=dict(net.named_parameters()))
    f = open('model.dot', 'w')
    f.write(dot.source)

def test(device=torch.device("cpu")):
    if args.sequences == '':
        sequencelist = [[args.images]]
    else:
        sequencelist = load_list(args.sequences, args.root)
    logf = open('log_p.txt', 'w')
    writer = SummaryWriter() if args.tensorboard else None
    net = prednet.PredNet(args.channels, diff_mode=args.diff_mode, device=device).to(device)
    net.eval()
    if args.initmodel:
        print('Load model from', args.initmodel)
        net.load_state_dict(torch.load(args.initmodel))
    
    for seq in range(len(sequencelist)):
        imagelist = load_list(sequencelist[seq][0], args.root) 
        # update dataset and loader 
        img_dataset = ImageListDataset(img_size=(args.size[0],args.size[1]),
                                       input_len=args.input_len, channels=args.channels[0])
        img_dataset.load_images(img_paths=imagelist, c_space=args.color_space)
        data_loader = DataLoader(img_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)
               
        if len(imagelist) == 0:
            print("Not found images.")
            return

        for i, data in enumerate(tqdm(data_loader, unit="batch")):
            for j in range(len(data)):
                for k in range(args.input_len):
                    start_idx = max(0, k - args.num_max_window) if args.num_max_window is not None else 0
                    x_batch = data[j, start_idx:k+2].view(1, k + 2 - start_idx, args.channels[0], args.size[1], args.size[0])
                    with torch.no_grad():
                        with torch.amp.autocast('cuda',enabled=args.useamp):#with torch.cuda.amp.autocast(enabled=args.useamp):
                            pred, errors, eval_index = net(x_batch.to(device))
                    write_image(data[j, k].detach().cpu().numpy(), 'result/test_' + str(k + (i * args.batchsize + j) * args.input_len ) + 'x',
                                img_dataset.mode, args.color_space)
                    write_image(pred[0].detach().cpu().numpy(), 'result/test_' + str(k + (i * args.batchsize + j) * args.input_len ) + 'y_0',
                                img_dataset.mode, args.color_space)
                    if writer is not None:
                        prefix = f"test_{i}_{j}"
                        write_outputs(writer, net.outputs, k, prefix)
                    s = str(k + (i * args.batchsize + j) * args.input_len )
                    for l in range(net.n_layers):
                        s += ', ' + str(eval_index[l][0].detach().cpu().numpy())
                    logf.write(s + '\n')
                exts = [pred[0].view(1, 1, args.channels[0], args.size[1], args.size[0])]
                y_batch = data[j, args.input_len:].view(1, 1, args.channels[0], args.size[1], args.size[0])
                for k in range(args.ext):
                    with torch.no_grad():
                        pred_ext, _, _ = net(torch.cat([x_batch.to(device)] + exts + [y_batch.to(device)], axis=1))
                    exts.append(pred_ext.unsqueeze(0))
                    write_image(pred_ext[0].detach().cpu().numpy(), 'result/test_' + str((i * args.batchsize + j + 1) * args.input_len - 1) + 'y_' + str(k + 1),
                                img_dataset.mode, args.color_space)
                    if writer is not None:
                        prefix = f"text_ext_{i}_{j}"
                        write_outputs(writer, net.outputs, k, prefix)


if __name__ == '__main__':
    args.size = args.size.split(',')
    for i in range(len(args.size)):
        args.size[i] = int(args.size[i])
    args.channels = args.channels.split(',')
    for i in range(len(args.channels)):
        args.channels[i] = int(args.channels[i])

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == '' else args.device
    if args.test:
        test(device)
    else:
        train(device)
