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


# -1 -> clockwise
def tangent_ratio(vectors, limits = None):
    # we want to know the angle between
    # a radius of the circle at the center of the image
    # and the motion vectors

    # center
    w = 160
    h = 120
    c = [w/2.0, h/2.0]

    # scores
    direction = 0
    mean_alignment = 0

    count = 0
    for v in vectors:
        #if(v[0]!=106): continue #39

        # oh boy
        # v 
        v[0] = v[0] - c[0]
        v[1] = v[1] - c[1]
        v[2] = v[0] + v[2]
        v[3] = v[1] + v[3]

        print(v)
        # radius vector R from origin of V to image center
        r = [0, 0, v[0], v[1]]
        print(r)

        # offsets
        ro = [r[2]-r[0], r[3]-r[1]]
        vo = [v[2]-v[0], v[3]-v[1]]

        # check limits

        norm_r = np.sqrt(ro[0]*ro[0] + ro[1]*ro[1])
        norm_v = np.sqrt(vo[0]*vo[0] + vo[1]*vo[1])
        if not limits is None:
            if (norm_r<limits[0]) or (norm_r>limits[1]):
                continue

        if(norm_r*norm_v==0):
            count = count + 1
            continue

        # find angle between vectors by using dot product
        dot_p = ro[0]*vo[0] + ro[1]*vo[1]
        angle = math.acos(dot_p/(norm_r * norm_v))

        angle_d = angle*180/math.pi
        # this angle is ideally pi/2 or -pi/2
        score = (math.pi/2) - abs(angle)

        angle_d = score*180/math.pi
        # and the max difference is pi/2
        score = 1 - (abs(score)/ (math.pi/2))
        
        # we'd like them to all have the same alignment
        # use cross product to find ccw or cv
        cw = ro[0]*vo[1] - ro[1]*vo[0]
        if(cw>0):
            mean_alignment = mean_alignment + score 
        else:
            mean_alignment = mean_alignment - score
        count = count + 1

    if mean_alignment > 0:
        direction = 1
    elif mean_alignment < 0:
        direction = -1

    if count > 0:
        mean_alignment = mean_alignment/count

    return [direction, abs(mean_alignment)]

# returns ratio and vectors that are not unplausibly big
def plausibility_ratio(vectors):
    r = []
    for vector in vectors:
        norm = np.sqrt(vector[2]*vector[2] + vector[3]*vector[3])
        # print("norm", norm)
        if norm> 1: # or norm==0: 
            continue
        r.append(vector)

    ratio = len(r)/len(vectors)
    return [ratio, r]


def predict(image_path, prediction_dir, model_name):
    print("Predicting illusions....")
    repeat = 10
    repeated_images_list = [image_path]*repeat
    w = 160
    h = 120
    half_h = int(h/2)
    size = [w,h]
    channels = [3,48,96,192]
    gpu = 0
    # runs repeat x times on the input image, save in result folder
    test_prednet(initmodel = model_name, sequence_list = [repeated_images_list], size=size, 
                channels = channels, gpu = gpu, output_dir = prediction_dir, skip_save_frames=repeat,
                reset_each = True, verbose = 0
                )
    # calculate flows
    i = 0
    original_vectors = [None] * repeat
    prediction_image_path = prediction_dir + "/" + str(i).zfill(10) + ".png"
    print(prediction_image_path)
    results = lucas_kanade(image_path, prediction_image_path, output_dir+"/flow/", save=True, verbose = 0)
    print(results["vectors"])
    if results["vectors"]:
        original_vectors = np.asarray(results["vectors"])
        #print("original_vectors", original_vectors)
    else:
        original_vectors = [[0,0,-1000,0]]


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
