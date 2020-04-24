import argparse
from chainer_prednet.PredNet.call_prednet import test_prednet
import cv2
import csv
from enum import IntEnum
import math
import numpy as np
from optical_flow.optical_flow import lucas_kanade
import os
from PIL import Image
import shutil
import shutil
import torch
from generate_illusion import tangent_ratio, plausibility_ratio

def predict(image_path, prediction_dir, model_name):
    print("Predicting illusions....")
    repeat = 5
    repeated_images_list = [image_path]*repeat
    w = 160
    h = 120
    half_h = int(h/2)
    size = [w,h]
    channels = [3,48,96,192]
    gpu = 0
    skip = 1
    extension_duration = 2
    # runs repeat x times on the input image, save in result folder
    test_prednet(initmodel = model_name, sequence_list = [repeated_images_list], size=size, 
                channels = channels, gpu = gpu, output_dir = prediction_dir, skip_save_frames=skip,
                extension_start = repeat, extension_duration = extension_duration,
                reset_at = repeat+extension_duration, verbose = 0
                )
    # calculate flows
    i = 0
    original_vectors = [None] * repeat
    # prediction_image_path = prediction_dir + "/" + str(i).zfill(10) + ".png"
    # print(prediction_image_path)
    index_0 = int(i*(repeat/skip)+3)
    index_1 = index_0+1
    prediction_0 = prediction_dir + "/" + str(index_0).zfill(10) + ".png"
    prediction_1 = prediction_dir + "/" + str(index_1).zfill(10) + ".png"
    print(prediction_0, prediction_1)
        # results = lucas_kanade(input_image, prediction_image_path, output_dir+"/flow/", save=True, verbose = 0)
    results = lucas_kanade(prediction_0, prediction_1, output_dir+"/flow/", save=True, verbose = 0)
    if results["vectors"]:
        original_vectors[i] = np.asarray(results["vectors"])
    else:
        original_vectors[i] = [[0,0,-1000,0]]

    ratio = plausibility_ratio(original_vectors) #TODO might not be needed?
    score_0 = ratio[0]
    good_vectors = ratio[1]
    limits = [0, half_h]
    score = tangent_ratio(good_vectors, limits)

    print("score", score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='optical flow tests')
    parser.add_argument('--model', '-m', default='', help='.model file')
    parser.add_argument('--output_dir', '-o', default='.', help='path of output diectory')
    parser.add_argument('--image', '-i', default='.', help='path of input image')

    args = parser.parse_args()
    output_dir = args.output_dir 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    predict(args.image, output_dir, args.model)
