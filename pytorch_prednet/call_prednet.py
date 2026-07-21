import argparse
import os
from datetime import datetime
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
import pytorch_prednet.prednet as prednet_model
from tqdm import tqdm
from distutils.util import strtobool
from pytorch_prednet.dataset import ImageListDataset, ImageHDF5Dataset
from pytorch_prednet.corr_wise import CorrWise
from random import random

# return the sorted list of images in that folder
def make_list(images_dir, limit):
    temp_list = sorted(os.listdir(images_dir))
    if(limit>0):
        temp_list = temp_list[0:limit]
    image_list = [os.path.join(images_dir, im)  for im in temp_list]
    return image_list

def read_image(full_path, size, offset, c = 3):
    image = Image.open(full_path)

    if(c<3):
        image_array = np.asarray(image)
        image_array_w = np.reshape(image_array, (1, size[1], size[0]))
        # image_array_w = image_array.transpose(1, 0) 
       # image_array_w = np.reshape(image_array, (1, size[1],size[0]))
    else:
        image_array = np.asarray(image)
        image_array_w = image_array.transpose(2, 0, 1)

    image = image_array_w/255
    return image

def write_image(image, path):
    image *= 255
    image = image.transpose(1, 2, 0)
    size = image.shape
    image = image.astype(np.uint8)
    if size[2]>1:
        result = Image.fromarray(image)
    else:
        result = Image.fromarray(image[:,:,0], 'L')

    result.save(path)

# imagelist = [path, path, path]
def test_image_list(prednet, imagelist, output_dir, channels, size, offset, gpu, skip_save_frames=0, 
    extension_start=0, extension_duration=100, reset_each = False, step = 0, verbose = 1, reset_at = -1, input_len=-1, c = 3):

    # # ----
    # print("here")
    # img_dataset = ImageListDataset(img_size=size,
    #                                input_len=20, channels=channels)
    # img_dataset.load_images(img_paths=sequence_list, c_space="RGB")
    # data_loader = DataLoader(img_dataset, batch_size=1, shuffle=False, num_workers=1)
      
    # for i, data in enumerate(tqdm(data_loader, unit="batch")):
    #     for j in range(len(data)):
    #         for k in range(20):
    #             start_idx = 0
    #             x_batch = data[j, start_idx:k+2].view(1, k + 2 - start_idx, channels[0], size[1], size[0])
    #             with torch.no_grad():
    #                 with torch.amp.autocast('cuda',enabled=args.useamp):#with torch.cuda.amp.autocast(enabled=args.useamp):
    #                     pred, errors, eval_index = prednet(x_batch.to(device))
    #             file_name = 'result/test_' + str(k + (i * args.batchsize + j) * args.input_len ) + 'y_0'
    #             print("writing b ", file_name)
    #             write_image(pred[0].detach().cpu().numpy(), file_name,
    #                         img_dataset.mode, args.color_space)

    # # ----


    # xp = cuda.cupy if gpu >= 0 else np
    # 　this should be replaced
    device=torch.device("cpu")

    # todo
    # prednet.reset_state()
    loss = 0
    batchSize = 1
    # x_batch = np.ndarray((batchSize, channels[0], size[1], size[0]), dtype=np.float32)
    # x_batch = data[j, start_idx:k+2].view(1, k + 2 - start_idx, channels[0], size[1], size[0])
    x_batch = np.ndarray((batchSize, 2, channels[0], size[1], size[0]), dtype=np.float32)

    y_batch = np.ndarray((batchSize, 2, channels[0], size[1], size[0]), dtype=np.float32)
    # ? 
    useamp = True

    if reset_each:
        reset_at = 1

    # for j, data in enumerate(tqdm(data_loader, unit="batch")):
    for i in range(len(imagelist)):

        if input_len>0 and i>input_len:
            break

        x_batch[0, 0] =  torch.Tensor(read_image(imagelist[i], size, offset, c))
        x_batch =  torch.Tensor(x_batch)
        x_batch.to(device)
        # x_batch[0] = data[0, 0:i+2].view(1, i + 2, channels[0], size[1], size[0])

        if(i<len(imagelist)-1):

            with torch.no_grad():
                with torch.amp.autocast('cuda',enabled=useamp):
                    pred, errors, eval_index = prednet(x_batch.to(device))


        loss += errors
        loss = 0
        # if gpu >= 0: model.to_cpu() # should be to gpu

        if(i<len(imagelist)-1):
            if verbose == 1:
                print("step ", step," frame ", i)
        else:
            num = str(offset).zfill(10)
            new_filename = output_dir + '/' + num + '.png'
            if verbose == 1:
                print("writing ", new_filename)
            write_image(pred[0].detach().cpu().numpy(), new_filename)
        # if gpu >= 0: model.to_gpu()


        step = step + 1
        if step == 0  or (extension_start==0) or (step%extension_start>0):
            continue

        # if gpu >= 0: model.to_cpu() # cpu is typo?
        # why this
        # x_batch[0,0] = pred[0]#.detach()#.cpu() # .numpy()
        # if gpu >= 0: model.to_gpu()

    return step


# image list: non repeating list
def test_prednet_pytorch(initmodel, image_list, size, channels, gpu, output_dir="result", 
                skip_save_frames=0, extension_start=0, extension_duration=0, offset = [0,0], 
                reset_each = False, verbose = 1, reset_at = -1, input_len=-1, c_dim = 3, jitter = 0):


    # this should be replaced
    device = torch.device("cpu")
    prednet = prednet_model.PredNet(channels, device=device, diff_mode="pos_neg")
    prednet.to(device)
    prednet.eval()

    print('Load model from', initmodel)
    prednet.load_state_dict(torch.load(initmodel))
        
    repeat = 20
    step = 0

    # there should be resetting here
    for n, image in enumerate(image_list):
        if (jitter == 0):
            sequence_list = [image]*repeat
        else :
            sequence_list = jitter_image(image, repeat)

        offset = n
        step = test_image_list(prednet, sequence_list, output_dir, channels, size, offset,
                                gpu, skip_save_frames, extension_start, extension_duration,
                                reset_each, step, verbose, reset_at, input_len, c_dim)
        

def jitter_image(image, n_times):

    jitter_range = 10; # todo optimize
    sequence_list = [None]*repeat

    for i in range(n_times):
        # create blank image
        jimage = Image.new(image.mode, (image.size[0], image.size[1]))
        x = random.randint(-jitter_range, jitter_range)
        y = random.randint(-jitter_range, jitter_range)
        jimage.paste(image, (x,y))
        sequence_list[i] = jimage

    return sequence_list

        


def call_with_args(args):  
    if (not args.images_path) and (not args.sequences):
        print('Please specify images or sequences')
        exit()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    size = string_to_intarray(args.size)
    channels = string_to_intarray(args.channels)
    offset = string_to_intarray(args.offset)

    if args.gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if args.gpu >= 0 else np

    if args.images_path:
        temp_list = make_list(args.images_path, args.input_len)
        sequence_list = [temp_list]
    else:
        # read file
        temp_list = [line.rstrip('\n') for line in open(args.sequences)]
        # now read files in list
        # extract path of seq list file
        array = args.sequences.split('/')
        base_path = os.path.abspath(os.getcwd()) + "/" + '/'.join(array[:-1])
        sequence_list = [None]* len(temp_list)
        i = 0
        for path in temp_list:
            sequence_list[i] = [os.path.join(base_path,line.rstrip('\n')) for line in open(os.path.join(base_path,path))]
            i = i+1

    if args.period:
        input_len = args.period
    else:
        input_len = args.input_len

    if args.test == True:
        test_prednet(args.initmodel, sequence_list, size, channels, args.gpu, args.output_dir,
                    args.skip_save_frames, args.ext_t, args.ext, offset, args.reset_each, args.verbose, input_len)

    else:
        train_prednet(args.initmodel, sequence_list, args.gpu, size, channels,
                            offset, args.resume, args.bprop, args.output_dir, input_len, args.save, args.verbose)  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description='PredNet')
    parser.add_argument('--images_path', '-i', default='', help='Path to input images')
    parser.add_argument('--output_dir', '-out', default= "result", help='where to save predictions')
    parser.add_argument('--sequences', '-seq', default='', help='In text mode, Path to file with list of text files, that themselves contain lists of images')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--initmodel', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--size', '-s', default='160,120',
                        help='Size of target images. width,height (pixels)')
    parser.add_argument('--channels', '-c', default='3,48,96,192',
                        help='Number of channels on each layers')
    parser.add_argument('--offset', '-off', default='0,0',
                        help='Center offset of clipping input image (pixels)')
    parser.add_argument('--ext', '-e', default=0, type=int,
                        help='Extended prediction on test (frames)')
    parser.add_argument('--ext_t', default=0, type=int,
                        help='When to start extended prediction')
    parser.add_argument('--bprop', default=20, type=int,
                        help='Back propagation length (frames)')
    parser.add_argument('--save', default=10000, type=int,
                        help='Period of save model and state (frames)')
    parser.add_argument('--period', default=1000000, type=int,
                        help='maximum input length (legacy)')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--skip_save_frames', '-sikp', type=int, default=1, help='predictions will be saved every x steps')
    parser.add_argument('--input_len', default=-1, type=int,
                        help='maximum input length')
    parser.add_argument('--verbose', '-v', default=1, type=int,
                        help='Output progression logs (1) or not (0)')

    parser.set_defaults(test=False)
    parser.set_defaults(reset_each=False)
    args = parser.parse_args()

    call_with_args(args)

