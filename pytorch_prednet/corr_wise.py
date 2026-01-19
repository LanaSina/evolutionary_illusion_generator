import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_utils import warp, clip_flow, normalize_flow

import pdb

class CorrWise(nn.Module):
    def __init__(self, 
                 base_loss, 
                 flow_method='RAFT',
                 clip_size=.15,
                 return_warped=True,
                 backward_warp=False,
                 detach=True,
                 clip=True,
                 padding_mode='zeros',
                 flow_resize=None,
                 use_occlusions=False,
                 reg_flow_mag=False,
                 l2_flow_mag=False,
                 flow_cycle_loss=False,
                 no_forward_warp=False,
                 reduction_clip=False,
                 scale_clip=False,
                 device=None):
        '''
        base_loss [func] : 
            takes in two torch images, and calculates a distance

        flow_method [nn.Module] OR [string] : 
            if [nn.Module] : takes in two images normalized to 
            [0,1], returns flow with units of pixels

            if [string] : One of ['RAFT', 'PWC', 'FBFlow']

        clip_size [float] :
            floating point number between 0 and 1, percentage of 
            smallest image dim to clip.
            This option is inferior to `scale_clip`

        return_warped [bool] : 
            if true, return warped images in forward as a dict

        backward_warp [bool] :
            perform alignment backwards (target warped into 
            predicted) as well, and averages loss

        detach [bool] :
            if true, then detach the flow prediction so no 
            gradients are calculated

        clip [bool] :
            if true, then clip flow before warping (this should
            probably be set to true*)
            *Update: This option is inferior to `scale_clip`

        padding_mode [string] :
            one of ['zeros', 'border', 'reflection']. 
            Determines how warping is padded. 
            See:
            https://pytorch.org/docs/stable/nn.functional.html#grid-sample

        flow_resize [float] :
            resize images by a factor of flow_resize before
            warping. This is useful for small images
            
        use_occlusions [bool] :
            if true, then fine occlusions and omit them
            from the flow calculation
            
        reg_flow_mag [float] :
            if non-zero, then average magnitude of flow for regularization
            with `reg_flow_mag` as scale factor
            
        flow_cycle_loss [float] :
            if non-zero, then use flow cycle consistency as regularization
            with `flow_cycle_loss` as scale factor
            
        no_forward_warp [bool] :
            if true, then don't use forward warp (backward warp had 
            better be true then...)
            
        reduction_clip [bool] :
            if true, then use a hard clip, and set large flow values to 0. This
            reduces the loss to a simple pixelwise loss
            
        scale_clip [float] :
            if true, then use scale the flow (`clip` is a misnomer) by
            1 - \epsilon, \epsilon set as value of this arg
            
        device [int] :
            device to use, if None, then don't move tensors around
        '''

        super(CorrWise, self).__init__()

        if flow_method == 'RAFT':
            from flow_utils import RAFT
            self.flow_method = RAFT()
        elif flow_method == 'PWC':
            from flow_utils import PWCNet
            self.flow_method = PWCNet()
        elif flow_method == 'FBFlow':
            from flow_utils import FBFlow
            self.flow_method = FBFlow()
        else:
            self.flow_method = flow_method

        self.base_loss = base_loss
        self.return_warped = return_warped
        self.backward_warp = backward_warp
        self.detach = detach
        self.clip_size = clip_size
        self.clip = clip
        self.padding_mode = padding_mode
        self.flow_resize = flow_resize
        self.use_occlusions = use_occlusions
        self.reg_flow_mag = reg_flow_mag
        self.l2_flow_mag = l2_flow_mag
        self.flow_cycle_loss = flow_cycle_loss
        self.no_forward_warp = no_forward_warp
        self.reduction_clip = reduction_clip
        self.scale_clip = scale_clip
        self.device = device
        
        assert (not self.no_forward_warp) or (self.backward_warp), \
            "Must use at least one of `forward` or `backward` warp!"
        
        if self.reg_flow_mag:
            assert not self.detach, 'Must set `no_detach` to true to regularize flow magnitude!'
            
    def calculate_flow(self, im1, im2, return_mag=False, l2_flow_mag=False, clip_override=False):
        '''
        Calculates flow between im1 and im2 (relative pixel-scale)
        Clips flow based on self.clip_size
        Normalizes flow to (absolute [-1, 1]-scale)
        '''
        # Resize before flow (useful for small images)
        if self.flow_resize is not None:
            im1 = F.interpolate(im1, scale_factor=self.flow_resize, mode='bilinear')
            im2 = F.interpolate(im2, scale_factor=self.flow_resize, mode='bilinear')
            
        # Calculate flow
        if self.detach:
            with torch.no_grad():
                flow = self.flow_method(im2, im1)
                flow = flow.detach()    # Prob not necessary
        else:
            flow = self.flow_method(im2, im1)
            
        # Calculate flow magnitude
        if return_mag:
            # Scale flow
            _, _, h, w = flow.shape
            size = torch.tensor([w, h]).float().to(flow.device)
            scaled_flow = flow / (-1 + size)[:,None,None]
            
            if not l2_flow_mag:
                # Take L1 mean of scaled flow
                mag = scaled_flow.abs().sum(1).mean()
            else:
                # Take L2 mean of scaled flow
                mag = torch.norm(scaled_flow, dim=1).mean()
            
        # Clip
        if self.clip and not clip_override:
            min_dim_size = min(tuple(flow.shape[2:]))
            thresh = self.clip_size * min_dim_size
            flow = clip_flow(flow, thresh, 
                             reduction_clip=self.reduction_clip,
                             scale_clip=self.scale_clip)
        
        # Normalize flow to [-1, 1]
        flow = normalize_flow(flow, im1)
        
        # Resize flow to correct dimensions
        if self.flow_resize is not None:
            # Interpolate expects (b, c, h, w)
            flow = flow.permute(0,3,1,2)
            flow = F.interpolate(flow, scale_factor=1./self.flow_resize, 
                                 recompute_scale_factor=False)
            # grid_sample expects (b, h, w, 2)
            flow = flow.permute(0,2,3,1)
        
        if return_mag:
            return flow, mag
        else:
            return flow, None
            
    def get_cycle_consistency(self, im1, im2, flow_forward=None, flow_backward=None):
        '''
        Calculates F(im1, im2) \circ F(im2, im1)
        This tells us how cycle consistent the flow is
        '''
        
        # Calculate Flows
        if flow_forward is None:
            flow_forward, _ = self.calculate_flow(im1, im2, clip_override=True)
        if flow_backward is None:
            flow_backward, _ = self.calculate_flow(im2, im1, clip_override=True)
        
        # Get cycle consistency (sample backward flow according to where forward flow goes to)
        flow_backward = flow_backward.permute(0,3,1,2)
        cycle = F.grid_sample(flow_backward, flow_forward, padding_mode='border', align_corners=True)
        
        # Get base (zero) flow
        _, h, w, _ = flow_forward.shape
        x, y = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w))
        base = torch.stack((y,x), dim=0).unsqueeze(0).to(cycle.device)
        
        # Get deviation from 0 flow
        diff = cycle - base
        
        return diff

    def get_occlusion_mask(self, im1, im2, flow_forward=None, flow_backward=None, thresh=.01):
        '''
        occlusion mask by cycle consistency
        1 = occlusion, 0 = no occlusion
        
        optionally pass in precalculated flows
        '''
        
        diff = self.get_cycle_consistency(im1, im2, 
                                          flow_forward=flow_forward, 
                                          flow_backward=flow_backward)
        
        error = diff.abs().mean(1)
        mask = (error > thresh).float()
        
        return mask
        
    def align_and_compare(self, im1, im2):
        # Get flow im1 -> im2
        flow, mag = self.calculate_flow(im1, im2, 
                                        return_mag=self.reg_flow_mag,
                                        l2_flow_mag=self.l2_flow_mag)
        
        # Warp im1 to im2
        warped = warp(im1, flow, padding_mode=self.padding_mode)

        # Calculate loss
        if not self.use_occlusions:
            return self.base_loss(warped, im2), warped, mag
        else:
            '''
            # TODO: This can be much more efficient if
            # we reuse past flow calculations
            '''
            mask = self.get_occlusion_mask(im1, im2)
            mask = mask.unsqueeze(1)
            warped = warped * (1 - mask)
            im2 = im2 * (1 - mask)
            return self.base_loss(warped, im2), warped, mag
        

    def forward(self, pred, target):
        # Move objects around GPUs
        if self.device:
            old_device = pred.device
            self.flow_method = self.flow_method.to(self.device)
            
            pred = pred.to(self.device)
            target = target.to(self.device)
        
        # Information about flow mag and warped images
        info = {}
        
        # TODO: The logic of backward and/or forward flow could be improved
        # it's very awkward right now, because we assumed initially only
        # forward flow would be sufficient. But this might not be the
        # case any more...

        # Forward align
        if not self.no_forward_warp:
            loss, warped, mag = self.align_and_compare(pred, target)
            info['forward_warp'] = warped
            if self.reg_flow_mag:
                info['flow_mag'] = mag
        else:
            # Initialize losses for backward warp
            loss = torch.tensor(0.).to(pred.device)
            info['flow_mag'] = torch.tensor(0.).to(pred.device)

        # Backward align
        if self.backward_warp:
            back_loss, back_warped, back_mag = self.align_and_compare(target, pred)
            info['backward_warp'] = back_warped
            loss = loss + back_loss
            
            # Return average of forward and backward flow mags
            if self.reg_flow_mag:
                info['flow_mag'] = info['flow_mag'] + back_mag
                
            # Take mean if we also took the forward warp
            if not self.no_forward_warp:
                loss = loss / 2.
                if self.reg_flow_mag:
                    info['flow_mag'] = info['flow_mag'] / 2.
            
        # Multiply by coefficient
        if self.reg_flow_mag:
            info['flow_mag'] = self.reg_flow_mag * info['flow_mag']
            
        if self.flow_cycle_loss:
            cycle = self.get_cycle_consistency(pred, target)
            info['flow_cycle_loss'] = torch.norm(cycle, dim=1).mean()
            info['flow_cycle_loss'] = self.flow_cycle_loss * info['flow_cycle_loss']
            
        if self.device:
            loss = loss.to(old_device)
            
        if self.return_warped:
            return loss, info
        else:
            return loss

