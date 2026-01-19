'''
Utilities for flow prediction, including:
    torch modules for pwcnet, raft, and cv2 flow
    image warping
    flow clipping
'''
import os
import pathlib
import sys
import argparse
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
from PIL import Image
import cv2

# from flow_models.pyflow import pyflow

class PWCNet(nn.Module):
    def __init__(self, default=True):
        super(PWCNet, self).__init__()
        
        from flow_models.pwcnet import pwcnet

        self.default = default  # Choose checkpoint to use
        self.pwcnet = pwcnet.Network(default=default)

    def forward(self, im1, im2):
        '''
        Input: images \in [0,1]
        '''

        flow = pwcnet.estimate(im1, im2, self.pwcnet)
        return flow

class RAFT(nn.Module):
    def __init__(self, model='things'):
        super(RAFT, self).__init__()
        
        from flow_models.raft import raft

        if model == 'things':
            model = 'raft-things.pth'
        elif model == 'kitti':
            model = 'raft-kitti.pth'

        # TODO: Figure out how to do checkpoints
        raft_dir = pathlib.Path(__file__).parent.absolute()/'flow_models'/'raft'

        # Emulate arguments
        args = argparse.Namespace()
        args.model = raft_dir / model
        args.small = False
        args.mixed_precision = True
        args.alternate_corr = False
        #args.alternate_corr = True # TODO: This doesn't work :(

        flowNet = nn.DataParallel(raft.RAFT(args))
        flowNet.load_state_dict(torch.load(args.model, map_location='cpu'))
        self.flowNet = flowNet.module

    def forward(self, im1, im2):
        '''
        Input: images \in [0,1]
        '''

        # Normalize to [0, 255]
        im1 = im1 * 255
        im2 = im2 * 255

        # Estimate flow
        flow_low, flow_up = self.flowNet(im1, im2, iters=5, test_mode=True)

        return flow_up

class FBFlow(nn.Module):
    def __init__(self):
        super(FBFlow, self).__init__()
        # TODO: Allow flow parameters to be set with **kwargs

    def forward(self, im1, im2):
        batch_size = im1.shape[0]
        device = im1.device

        # Iterate through batches and calculate flows
        flows = []
        for b in range(batch_size):
            cvim1 = np.array(im1.cpu().detach()[0].permute(1,2,0))
            cvim2 = np.array(im2.cpu().detach()[0].permute(1,2,0))

            #cvim1 = cv2.cvtColor(cvim1, cv2.COLOR_BGR2GRAY) * 255
            #cvim2 = cv2.cvtColor(cvim2, cv2.COLOR_BGR2GRAY) * 255

            # TODO: HANDLE greyscale images
            cvim1 = cvim1[:,:,0]
            cvim2 = cvim2[:,:,0]

            flow = cv2.calcOpticalFlowFarneback(cvim1, cvim2, None,
                                                pyr_scale=.5,
                                                levels=8,
                                                winsize=40,
                                                iterations=3,
                                                poly_n=12,
                                                poly_sigma=1.5,
                                                flags=0)

            flow = torch.tensor(flow).permute(2,0,1).to(device).unsqueeze(0)
            flows.append(flow)


        flow = torch.cat(flows, dim=0)
        return flow

def clip_flow(flow, thresh, reduction_clip=False, scale_clip=None):
    # Either apply tanh soft thresholding (reduce = False)
    # OR apply hard tresholding, and set large flows to 0 (reduce = True)
    # OR if scale_clip==True, multiply flow by 1-\epsilon
    if scale_clip:
        epsilon = scale_clip
        return flow * (1 - epsilon)
        
    if not reduction_clip:
        thresh = float(thresh)
        eps = 1e-8
        scale = torch.tanh(flow.norm(dim=1) / thresh) * thresh / (flow.norm(dim=1) + eps)
    else:
        mask = flow.norm(dim=1) < thresh
        return mask.unsqueeze(1) * flow

    return scale.unsqueeze(1) * flow
    
def normalize_flow(flow, im):
    '''
    Normalize pixel-offset flow to absolute [-1, 1] flow
    input :
        flow : tensor (b, 2, h, w)
        im : tensor (b, c, h, w)
            This is just to get size of image
            (could be replaced by a size argument...)
    output :
        flow : tensor (b, h, w, 2) (for `F.grid_sample`)
    '''
    h = im.shape[2]
    w = im.shape[3]
    device = flow.device

    # Get base pixel coordinates (just "gaussian integers")
    base = torch.meshgrid(torch.arange(h), torch.arange(w))[::-1]
    base = torch.stack(base).float().to(device)
    size = torch.tensor([w, h]).float().to(device)

    # Convert to absolute coordinates
    flow = flow + base

    # Convert to [-1, 1] for grid_sample
    flow = -1 + 2.*flow/(-1 + size)[:,None,None]
    flow = flow.permute(0,2,3,1)
    
    return flow
    

def warp(im, flow, padding_mode='zeros'):
    '''
    requires absolute flow, normalized to [-1, 1]
        (see `normalize_flow` function)
    '''
    warped = F.grid_sample(im, flow, padding_mode=padding_mode, align_corners=True)

    return warped

