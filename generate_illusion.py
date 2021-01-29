import argparse
from chainer_prednet.PredNet.call_prednet import test_prednet
from chainer_prednet.utilities.mirror_images import mirror, mirror_multiple, TransformationType
import csv
import cv2
from enum import IntEnum
from google.colab import files
import math
import neat
import numpy as np
from optical_flow.optical_flow import lucas_kanade, draw_tracks, save_data
import os
from PIL import Image, ImageOps
from pytorch_neat.pytorch_neat.cppn import create_cppn
from pytorch_neat.pytorch_neat.multi_env_eval import MultiEnvEvaluator
from pytorch_neat.pytorch_neat.neat_reporter import LogReporter
from pytorch_neat.pytorch_neat.recurrent_net import RecurrentNet
from random import random
import shutil
import torch


# TODO enumerate illusion types
class StructureType(IntEnum):
    Bands = 0
    Circles = 1
    Free = 2
    CirclesFree = 3


# returns ratio and vectors that are not unplausibly big
def plausibility_ratio(vectors, limit):
    r = []
    for vector in vectors:
        norm = np.sqrt(vector[2]*vector[2] + vector[3]*vector[3])
        if norm> limit:
            continue
        r.append(vector)

    ratio = len(r)/len(vectors)
    return [ratio, r]

# returns mean of vectors norms weighted by their variances
# low variance = good
def strength_number(vectors, max_norm):
    v = np.asarray(vectors)
    mx = np.mean(abs(v[:,2]))
    my = np.mean(abs(v[:,3]))

    norms = np.sqrt(v[:,2]*v[:,2] + v[:,3]*v[:,3])
    v = np.var(norms)
    score = mx/max_norm # could be 1
    score = score * (1- min(v,1))
    return score


# returns [a,b]
# a = 1 if vectors rather aligned on x to the right;  -1 if to the left
# b = mean of projection on x axis (normalised)
def direction_ratio(vectors, limits = None):
    # print(vectors)
    mean_ratio = 0
    count = 0
    orientation = 0

    for v in vectors:
        # skip vectors that are outside the limits
        if not limits is None:
            if (v[1]<limits[0]) or (v[1]>limits[1]):
                continue

        # calculate x axis ratio
        # x length divided by norm
        norm_v = np.sqrt(v[2]*v[2] + v[3]*v[3])
        ratio = v[2]/norm_v
        mean_ratio = mean_ratio + ratio
        orientation = orientation + v[2]
        count = count + 1

    if count>0:
        mean_ratio = mean_ratio / count
    else:
        mean_ratio = 0

    if orientation>0:
        orientation = 1
    elif orientation<0:
        orientation = -1

    return [orientation, mean_ratio]

# calcuates the symmetry on the middle axis
def horizontal_symmetry_score(vectors, limits = [0,60]):
    # print(vectors)
    mean_ratio = 0
    count = 0
    orientation = 0
    middle = int(limits[1]/2)

    # matrix of mirrored vectors
    mirrored_vectors = np.zeros((len(vectors), 2))

    count = 0
    for v in vectors:
        # skip vectors that are outside the limits
        if (v[1]<limits[0]) or (v[1]>limits[1]):
            continue

        # normalize the vectors to offset model biases
        normalized_v = v / np.sqrt(v[2]*v[2] + v[3]*v[3])

        if (v[1]<middle):
            mirrored_vectors[count] = normalized_v[2:3]
        else:
            mirrored_vectors[count] = [-normalized_v[2],normalized_v[3]]
        
        count = count+1

    if (count==0):
        return 0

    # remove everything beyond count
    mirrored_vectors = mirrored_vectors[:count, :]

    var_x = np.var(mirrored_vectors[:,0])
    mean_x = abs(np.mean(mirrored_vectors[:,0]))
    mean_y = abs(np.mean(mirrored_vectors[:,1]))

    # max var is 1
    score = ((1 - var_x) + mean_x + (1-mean_y))/3
    # print("score", score)
    return score


# returns the agreement and disagreement betwen vectors
def swarm_score(vectors,w, h):
    max_distance = 100 #px
    distance_2 = 50
    score = 0
    n = len(vectors)

    # normalize vectors
    norm_vectors = np.array(vectors) 
    # print("vector array", norm_vectors)
    norms = np.sqrt(norm_vectors[:,2]*norm_vectors[:,2] + norm_vectors[:,3]*norm_vectors[:,3])
    norm_vectors[:,2] = norm_vectors[:,2]/norms
    norm_vectors[:,3] = norm_vectors[:,3]/norms
    #print("normalized", norm_vectors)
    temp = np.sqrt(norm_vectors[:,2]*norm_vectors[:,2] + norm_vectors[:,3]*norm_vectors[:,3])
    #print("norms", temp)
    angles = np.arccos(norm_vectors[:,2])

    for v_a in norm_vectors:
        # distance used as factor
        x = norm_vectors[:,0]-v_a[0]
        y = norm_vectors[:,1]-v_a[1]
        # [0 .. 1]
        distances = (np.multiply(x,x) + np.multiply(y,y))
        distance_factors = distances/(max_distance*max_distance)
        distance_factors = np.where(distance_factors > 1, 1, distance_factors)
        # 1 where vectors are close
        close = 1 - np.where(distance_factors < 1, 0, distance_factors)
        # close = 1-distance_factors

        # distance_factors = (np.multiply(x,x) + np.multiply(y,y))
        # distance_factors = np.where(distance_factors > distance_2*distance_2, distance_2*distance_2, distance_factors)
        # distance_factors = np.where(distance_factors < max_distance*max_distance, distance_2*distance_2, distance_factors)
        # far = 1 - (distance_factors/(distance_2*distance_2))
        # #print("far", far)

        # vectors orientation
        # alpha = acos(x)
        v_angle = math.acos(v_a[2])
        angle_diff = abs(angles-v_angle)
        angle_diff = angle_diff % 2*math.pi
        angle_diff = angle_diff/(2*math.pi)
        # v_agreement = np.multiply(close,abs(1-angle_diff))
        # v_discord = np.multiply(far,abs(angle_diff))

        # # optimize for a balance of extreme values
        # s1 = sum(v_agreement)/(2*math.pi*max_distance) 
        # s2 = sum(v_discord)/(2*math.pi*(distance_2- max_distance))
        # temp = s1*s2

        # oprimal deviation: completely opposite at 100 px away (distance factor  = 1)
        optimal = (v_angle + distance_factors*math.pi)%2*math.pi
        loss = close*abs(angles-optimal)
        temp = math.pi - (sum(loss)/n)
        score = score + (temp/math.pi)


    return score/n

# rotate all vectors to align their origin on x axis
# calculate the mean and variance of normalized vectors
# returns a high score if the variance is low (ie the vectors are symmetric)
# limits = radius limits
def rotation_symmetry_score(vectors, w, h, limits = None):

    # fill matrix of vectors
    rotated_vectors = np.zeros((len(vectors), 4))
    distances = np.zeros((len(vectors)))
    count = 0
    center = [w/2, h/2]
    for v in vectors:
        # change coordinates to center
        vc = [v[0]-center[0], v[1]-center[1]]
        distance = np.sqrt(vc[0]*vc[0] + vc[1]*vc[1])
        if not limits is None:
            if (distance<limits[0]) or (distance>limits[1]) or distance==0:
                continue

        rotated_vectors[count] =[vc[0],vc[1],v[2],v[3]]
        distances[count] = distance
        count = count+1

    if(count < 2):
        return 0

    # remove everything beyond count
    rotated_vectors = rotated_vectors[:count, :]
    distances = distances[:count]

    # normalise vectors
    norms = np.sqrt(rotated_vectors[:,2]*rotated_vectors[:,2] + rotated_vectors[:,3]*rotated_vectors[:,3])
    rotated_vectors[:,2] = rotated_vectors[:,2]/norms
    rotated_vectors[:,3] = rotated_vectors[:,3]/norms

    
    # rotate vectors clockwise to x axis
    # new_x = cos(a)x + sin(a)y, new_y = cos(a)y - sin(a)x
    # cos(a) = x/dist, sin a = y/dist
    # new_y = -sin(a)x + cos(a)y
    # vector origin is going to be [dist,0]
    # vector end coordinates
    x_1 = rotated_vectors[:,0] + rotated_vectors[:,2]
    y_1 = rotated_vectors[:,1] + rotated_vectors[:,3]

    rx_1 = (x_1*rotated_vectors[:,0] + y_1*rotated_vectors[:,1])/distances
    ry_1 = (-x_1*rotated_vectors[:,1] + y_1*rotated_vectors[:,0])/distances
    r_v = np.array([rx_1-distances, ry_1]).transpose()

    var_x = np.var(r_v[:,0])
    var_y = np.var(r_v[:,1])

    # max var is 1
    score = (1 - var_x)*(1 - var_x) + (1 - var_y)*(1 - var_y)
    score = score/2
    return score


# agreement inside the cell, + disagreement outside of it
def inside_outside_score(vectors, width, height):

    step = width/5 #px
    # build an array of vectors 
    w = int(width/step) + 1
    h = int(height/step) + 1
    flow_array = np.zeros((w, h, 2))
    count_array = np.ones((w, h))
    agreement_array = np.zeros((w, h, 2))
    norm_sum_array = np.zeros((w, h))

    # take the mean for vectors in the same cell, and calculate agreement score
    # vectors orientation 
    for index in range(0,len(vectors)):
        v = vectors[index]
        i = int(v[0]/step)
        j = int(v[1]/step)

        flow_array[i,j,0] += v[2]
        flow_array[i,j,1] += v[3]
        count_array[i,j] += 1 
        norm_v = np.sqrt(v[2]*v[2] + v[3]*v[3])
        norm_sum_array[i,j] += norm_v

    # not a real mean as the count started at 1
    flow_array[:,:,0] = flow_array[:,:,0]/count_array
    flow_array[:,:,1] = flow_array[:,:,1]/count_array
    norm_sum_array = norm_sum_array/count_array

    # now take the variance
    for index in range(0,len(vectors)):
        v = vectors[index]
        i = int(v[0]/step)
        j = int(v[1]/step)
        agreement_array[i,j,0] += (flow_array[i,j,0] - v[2])*(flow_array[i,j,0] - v[2])
        agreement_array[i,j,1] += (flow_array[i,j,1] - v[3])*(flow_array[i,j,1] - v[3])

    agreement_array[:,:,0] =  agreement_array[:,:,0]/count_array
    agreement_array[:,:,1] =  agreement_array[:,:,1]/count_array

    # take the sums
    score_agreement =  - (min(np.mean(agreement_array), 10))
    score_size = min(10, np.mean(norm_sum_array))

    # compare with other cells
    sum_d = 0
    for i in range(0,w):
        for j in range(0,h):
            vx = flow_array[i,j,0]
            vy = flow_array[i,j,1]
            if (vx!=0 or vy!=0):
            # normalize
                norm_v = np.sqrt(vx*vx + vy*vy)
                vx = vx/norm_v
                vy = vy/norm_v

            min_i = max(0,i-1)
            max_i = min(w,i+1)
            min_j = max(0,j-1)
            max_j = min(h,i+1)
            plus = 0
            minus = 0
            for x in range(min_i,max_i):
                for y in range(min_j,max_j):
                    if i == x and j == y:
                        continue

                    wx = flow_array[x,y,0]
                    wy = flow_array[x,y,1]
                    if (wx!=0 or wy!=0):
                        norm_w = np.sqrt(wx*wx + wy*wy)
                        wx = wx/norm_w
                        wy = wy/norm_w
                        # +1 for disagreement
                        dot = vx*wx + vy*wy
                        if dot >0:
                            plus += 1
                        else:
                            minus +=1
            sum_d += (min(2, plus) + min(2,minus))/4

    sum_d = sum_d/(w*h)
    sum_d = sum_d*10

    final_score = score_agreement + score_size + sum_d
    final_score = final_score/30
    return final_score



# calculate how parallel nearby patches are and how different they are from
# slightly further away patches
def divergence_convergence_score(vectors, width, height):

    step = height*4/len(vectors)

    score = 0
    step = 10 #px
    # build an array of vectors 
    w = int(width/step)
    h = int(height/step)
    flow_array = np.zeros((w, h, 2))

    # TODO: take the mean for vectors in the same cell
    # vectors orientation 
    for index in range(0,len(vectors)):
        v = vectors[index]
        i = int(v[0]/step)
        j = int(v[1]/step)
        norm_v = np.sqrt(v[2]*v[2] + v[3]*v[3])
        x = v[2]/norm_v
        y = v[3]/norm_v
        flow_array[i,j,0] = x
        flow_array[i,j,1] = y

    # calculate points
    for i in range(0,w):
        for j in range(0,h):
            xmin = max(i - 1, 0)
            xmax = min(i+1, w)
            ymin = max(j - 1, 0)
            ymax = min(j+1, h)
            loss = 0
            sum_vec = 0
            vx = flow_array[i,j,0]
            vy = flow_array[i,j,1]
            if vx == 0 and vy == 0:
                        continue

            plus = 0
            minus = 0

            sum_norm = 0
            for x in range(xmin, xmax):
                for y in range(ymin, ymax):
                    if flow_array[x,y,0] == 0 and flow_array[x,y,1] == 0:
                        continue

                    sum_vec += 1

                    dot = vx*flow_array[x,y,0] + vy*flow_array[x,y,1]
                    # aim for either completely different or completely same
                    loss = (abs(dot)-0.5)*(abs(dot)-0.5)
                    if (dot>0):
                        plus += dot
                    else:
                        minus -= dot
                    
                    # loss += (dot-0.5)*(dot-0.5)
                    # sum_vec += 1

            if(sum_vec>0):
                # there must be + and - in equal parts
                # print("plus, minus", plus, minus)
                loss = 1 - (plus - minus)/ (plus + minus)
                # high norms are better
                loss = loss * abs(vx+vy)
                score += loss
                # print("loss", loss, "score", score)

    return score


# limits: radius limits
# returns high scores if vectors are aligned on concentric circles
# [a,b]
# a = 1 if vectors rather aligned clockwise;  -1 if counterclockwise
# b = 1 if all vectors are tangent
# 1 -> clockwise
# -1 0-> counter clockwise
def tangent_ratio(vectors, w, h, limits = None):
    # we want to know the angle between
    # a radius of the circle at the center of the image
    # and the motion vectors

    # center
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

        # radius vector R from origin of V to image center
        r = [0, 0, v[0], v[1]]
        # offsets: change origin to vector origin
        ro = [r[2]-r[0], r[3]-r[1]]
        vo = [v[2]-v[0], v[3]-v[1]]

        # check limits
        norm_r = np.sqrt(ro[0]*ro[0] + ro[1]*ro[1])
        norm_v = np.sqrt(vo[0]*vo[0] + vo[1]*vo[1])

        if(norm_r*norm_v==0):
            count = count + 1
            continue

        # normalize 
        ro = ro/norm_r
        vo = vo/norm_v

        if not limits is None:
            if (norm_r<limits[0]) or (norm_r>limits[1]):
                continue

        # find angle between vectors by using dot product
        dot_p = ro[0]*vo[0] + ro[1]*vo[1] #  divide by (norm v * norm r) which is 1*1
        # sometimes slight errors

        if dot_p>1:
            dot_p = 1
        elif dot_p<-1:
            dot_p =-1

        angle = math.acos(dot_p)
        # this angle is ideally pi/2 or -pi/2
        score = (math.pi/2) - abs(angle)
        # and the max difference is pi/2
        score = 1 - (abs(score)/ (math.pi/2))
        
        # we'd like them to all have the same alignment
        # use cross product to find ccw or cv
        cw = ro[0]*vo[1] - ro[1]*vo[0]
        # maybe just add, if it's a flow fluke it will always be lower anyway
        # mean_alignment = mean_alignment + abs(score)
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

def get_vectors(image_path, model_name, w, h):
    skip = 1
    extension_duration = 2
    repeat = 20
    half_h = int(h/2)
    size = [w,h]
    channels = [3,48,96,192]
    gpu = 0

    output_dir = "test/" 
    prediction_dir = output_dir + "/prediction/"
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    
    repeated_images_list = [image_path]*repeat
    # print("list", repeated_images_list)

    # runs repeat x times on the input image, save in result folder
    test_prednet(initmodel = model_name, sequence_list = [repeated_images_list], size=size, 
                channels = channels, gpu = gpu, output_dir = prediction_dir, skip_save_frames=skip,
                extension_start = repeat, extension_duration = extension_duration,
                reset_at = repeat+extension_duration, verbose = 0
                )

    extended = prediction_dir + str(repeat+1).zfill(10) + "_extended.png"
    # calculate flows
    print("Calculating flows...", extended)
    vectors = [None] 

    results = lucas_kanade(image_path, extended, prediction_dir, save=True, verbose = 0, save_name = "flow.png")
    if results["vectors"]:
        vectors = np.asarray(results["vectors"])

    return vectors


# todo: use in get_grid
# fill carthesian grids with polar coordinates
# r_len = repetition length
# xx yy cartesian x and y, origin relative to whole grid
# x,y coordinates relative to center
# direction: 1 or -1
def fill_circle(x, y, xx, yy, max_radius, direction, structure=StructureType.Circles): #max diameter?
    r_total = np.sqrt(x*x + y*y)

    n_ratios = 10
    r_ratios = np.zeros(n_ratios)
    r_ratios[n_ratios-1] = 1

    for i in range(2,n_ratios+1):
        r_ratios[n_ratios-i] = r_ratios[n_ratios-i+1]*1.5

    r_ratios = r_ratios/r_ratios[0]

    # limit values to frame
    theta = 0
    r = -1
    if r_total <= max_radius/2:
        # it repeats every r_len
        radius = min(1, r_total/(max_radius/2))
        
        radius_index = 0
        for i in range(1,n_ratios-1):
            if radius > r_ratios[i]:
                r = (radius-r_ratios[i])/(r_ratios[i-1]-r_ratios[i])
                radius_index = n_ratios-i-1
                break;

        if structure == StructureType.Circles:
            # now structure theta values
            if x == 0:
                theta = math.pi/2.0
            else:
                theta = np.arctan(y*1.0/x)

            if x<0:
                theta = theta + math.pi

            r_index = radius_index 
            if r_index%2 == 1:
                # rotate
                theta = (theta + math.pi/4.0) 

            # focus on 1 small pattern
            theta = theta % (math.pi/6.0)

            if direction<0:
                theta = (math.pi/6.0) - theta

        elif structure == StructureType.CirclesFree:

            # now structure theta values
            if x == 0:
                theta = math.pi/2.0
            else:
                theta = np.arctan(y*1.0/x)

            if x<0:
                theta = theta + math.pi

            r_index = radius_index
            if r_index%2 == 1:
                # rotate
                theta = (theta + math.pi/4.0) 

            if direction<0:
                theta = - theta

        # keep some white space
        if (r>0.9) or (r<0.1):
            r = -1
            theta = 0
        else :
            #final normalization
            r = r/0.8

    return r, theta


def enhanced_image_grid(x_res, y_res, structure):

    x_mat = None
    y_mat = None
    scaling = 10

    num_points = x_res*y_res
    # coordinates of circle centers
    # 1: one row of circles at each third of the image
    c_rows = 3
    # 4 circles per row
    c_cols = 3
    y_step = (int) (y_res/c_cols)
    x_step = (int) (x_res/c_cols)

    # overlaid cicrles: 2 rows of 3 circles
    sub_rows = c_rows-1
    sub_cols = c_cols-1
    # coordinates
    centers = [None]*(c_rows*c_cols + sub_rows*sub_cols)

    for y in range(c_rows):
        for x in range(c_cols):
            index = y*c_cols + x
            centers[index] = [x_step*x + x_step/2, y_step*y + y_step/2]

    for y in range(sub_rows):
        for x in range(sub_cols):
            index = c_rows*c_cols + y*sub_cols + x
            centers[index] = [x_step*x + x_step, y_step*y + x_step]

    # radial repetition
    r_rep = 3
    r_len = int(y_step/(2*r_rep))
    x_range = np.linspace(-1*scaling, scaling, num = x_res)
    y_range = np.linspace(-1*scaling, scaling, num = y_res)

    y_mat = np.matmul(y_range.reshape((y_res, 1)), np.ones((1, x_res)))
    x_mat = np.matmul(np.ones((y_res, 1)), x_range.reshape((1, x_res)))

    for row in range(c_rows):
        for col in range(c_cols):
            index = row*c_cols + col
            direction = 1
            if index%2==0:
                direction = -1
            for xx in range(x_step):
                # shift coordinate to center of circle
                real_x = (col*x_step + xx)
                x =  real_x - centers[index][0]
                for yy in range(y_step):
                    real_y = (row*y_step + yy)
                    y =  real_y - centers[index][1]
                    r, theta = fill_circle(x, y, real_x, real_y, y_step, direction, structure)
                    x_mat[real_y,real_x] = r 
                    y_mat[real_y,real_x] = theta 


    # secondary layer of circles
    for row in range(sub_rows):
        for col in range(sub_cols):
            index = c_rows*c_cols + row*sub_rows + col
            direction = 1
            if index%2==0:
                direction = -1
            for xx in range(x_step):
                # shift coordinate to center 
                real_x = (col*x_step + xx) + (int) (x_step/2)
                x =  real_x - centers[index][0]
                for yy in range(y_step):
                    real_y = (row*y_step + yy) + (int) (y_step/2)
                    y =  real_y - centers[index][1]
                    r_total = np.sqrt(x*x + y*y)
                    if r_total < x_step/2:
                        r, theta = fill_circle(x, y, real_x, real_y, y_step, direction, structure)
                        x_mat[real_y,real_x] = r 
                        y_mat[real_y,real_x] = theta 

        
    return {"x_mat": x_mat, "y_mat": y_mat}

def create_grid(structure, x_res = 32, y_res = 32, scaling = 1.0):

    r_mat = None 
    x_mat = None
    y_mat = None
    num_points = x_res*y_res
   
    if structure == StructureType.Bands:
        y_rep = 4
        padding = 10
        total_padding = padding*(y_rep-1)
        y_len = int(y_res/y_rep) 
        sc = scaling/y_rep
        a = np.linspace(-1*sc, sc, num = y_len-padding)
        to_tile = np.concatenate((a,np.zeros((padding))))
        y_range = np.tile(to_tile, y_rep)
       # x_range = np.linspace(-1*scaling, scaling, num = x_res)

        x_rep = 5
        x_len = int(x_res/x_rep) 
        sc = scaling/x_rep
        a = np.linspace(-1*sc, sc, num = x_len)
        x_range = np.tile(a, x_rep)
        # reverse the x axis 
        x_reverse = np.ones((y_res, 1))
        start = y_len
        while start<y_res:
            # keep some white space
            # top of previous band
            m_start = max(0,start-padding)
            x_reverse[m_start:start] = np.zeros((start-m_start,1))

            # bottom of current band
            stop = min(y_res, start+y_len)
            m_start = max(stop - padding, 0) #max(0,start-padding)
            x_reverse[m_start:stop] = np.zeros((stop-m_start,1))
            x_reverse[start:stop] =  -x_reverse[start:stop]

            start = start+2*y_len

        x_mat = np.matmul(x_reverse, x_range.reshape((1, x_res)))
        y_mat = np.matmul(y_range.reshape((y_res, 1)), np.ones((1, x_res)))

        return {"x_mat": x_mat, "y_mat": y_mat} 

    elif structure == StructureType.Circles:
        #r_rep = 3
        #r_len = int(y_res/(2*r_rep))
        # r_len = [int(0.4*y_res/2), int(0.25*y_res/2) + int(0.15*y_res/2)]
        r_ratios = [0.6,0.3,0.1] 
        x_range = np.linspace(-1*scaling, scaling, num = x_res)
        y_range = np.linspace(-1*scaling, scaling, num = y_res)

 
        y_mat = np.matmul(y_range.reshape((y_res, 1)), np.ones((1, x_res)))
        x_mat = np.matmul(np.ones((y_res, 1)), x_range.reshape((1, x_res)))
        # x = r × cos( θ )
        # y = r × sin( θ )
        #radius_index = 0
        for xx in range(x_res):
            # center
            x = xx - (x_res/2)
            for yy in range(y_res):
                y = yy - (y_res/2)

                r,theta = fill_circle(x, y, xx, yy, y_res, 1)

                x_mat[yy,xx] = r 
                y_mat[yy,xx] = theta 
        return {"x_mat": x_mat, "y_mat": y_mat}

    elif structure == StructureType.CirclesFree:
        r_rep = 3
        r_len = int(y_res/(2*r_rep))
        x_range = np.linspace(-1*scaling, scaling, num = x_res)
        y_range = np.linspace(-1*scaling, scaling, num = y_res)

 
        y_mat = np.matmul(y_range.reshape((y_res, 1)), np.ones((1, x_res)))
        x_mat = np.matmul(np.ones((y_res, 1)), x_range.reshape((1, x_res)))

        # x = r × cos( θ )
        # y = r × sin( θ )
        for xx in range(x_res):
            # center
            x = xx - (x_res/2)
            for yy in range(y_res):
                y = yy - (y_res/2)
                r_total = np.sqrt(x*x + y*y)
                
                # limit values to frame
                r = min(r_total, y_res/2)
                # it repeats every r_len
                r = r % r_len
                # normalize
                r = r/r_len

                # now structure theta values
                theta = 0
                if r_total < y_res/2:
                    if x == 0:
                        theta = math.pi/2.0
                    else:
                        theta = np.arctan(y*1.0/x)

                    if x<0:
                        theta = theta + math.pi

                    r_index = int(r_total/r_len)
                    if r_index%2 == 1:
                        # rotate
                        theta = (theta + math.pi/4.0) 

                x_mat[yy,xx] = r 
                y_mat[yy,xx] = theta 

        return {"x_mat": x_mat, "y_mat": y_mat}

    elif structure == StructureType.Free:
        x_range = np.linspace(-1*scaling, scaling, num = x_res)
        y_range = np.linspace(-1*scaling, scaling, num = y_res)


        y_mat = np.matmul(y_range.reshape((y_res, 1)), np.ones((1, x_res)))
        x_mat = np.matmul(np.ones((y_res, 1)), x_range.reshape((1, x_res)))

        return {"x_mat": x_mat, "y_mat": y_mat}


    return {"input_0": x_mat, "input_1": y_mat, "input_2": r_mat} #, s_mat


def get_fidelity(input_image_path, prediction_image_path):
    input_image = np.array(Image.open(input_image_path).convert('RGB'))
    prediction = np.array(Image.open(prediction_image_path).convert('RGB'))

    err = np.sum((input_image.astype("float") - prediction.astype("float")) ** 2)
    err /= (float(input_image.shape[0] * input_image.shape[1])*255*255)
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return 1-err


# bg = background, 1 for white 0 for black
def get_image_from_cppn(inputs, genome, c_dim, w, h, scaling, config, s_val = 1, bg = 1, gradient = 1):
   
    # why twice???
    out_names = ["r0","g0","b0","r1","g1","b1"]
    leaf_names = ["x","y"]
    x_dat = inputs["x_mat"]
    y_dat = inputs["y_mat"]
    inp_x = torch.tensor(x_dat.flatten())
    inp_y = torch.tensor(y_dat.flatten())
   
    #or h w ??

    if(c_dim>1):
        image_array = np.zeros(((h,w,c_dim)))
        c = 0
        net_nodes = create_cppn(
            genome,
            config,
            leaf_names,
            out_names
        )

        for node_func in net_nodes:
            if(c>=3):
                break

            pixels = node_func(x=inp_x, y=inp_y)
            pixels_np = pixels.numpy()
            image_array[:,:, c] = np.reshape(pixels_np, (h,w))

            for x in range(h):
                for y in range(w):
                    if x_dat[x][y] == -1:
                        image_array[x, y, c] = bg #white or black
            c = c + 1

        # for no shading
        # img_data = np.array(np.round(image_array)*255.0, dtype=np.uint8)
        if gradient==0:
            image_array = np.round(image_array)

        img_data = np.array(image_array*255.0, dtype=np.uint8)
        image =  Image.fromarray(img_data)#, mode = "HSV")
    else:
        image_array = np.zeros(((h,w)))
        net_nodes = create_cppn(
            genome,
            config,
            leaf_names,
            out_names
        )
        node_func = net_nodes[0]
        pixels = node_func(x=inp_x, y=inp_y)
        pixels_np = pixels.numpy()
        # print(pixels_np.shape)
        #image_array = np.zeros(((w,h,c_dim))) # (warning 1) c_dim here should be 3 if using a color prednet model as black and white...
        # pixels_np = np.reshape(pixels_np, (w, h)) * 255.0
        pixels_np = np.reshape(pixels_np, (h, w)) 
        # same
        image_array = pixels_np
        for x in range(h):
            for y in range(w):
                if x_dat[x][y] == -1:
                    image_array[x,y] = bg

        if gradient == 0:
            image_array = np.round(image_array)

        img_data = np.array(image_array*255.0, dtype=np.uint8)
        image =  Image.fromarray(img_data , 'L')
        #Image.fromarray(np.reshape(img_data,(h,w,3))) 

    return image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


def cppn_evolution(population, repeat, structure, w, h, gpu, config, c_dim, gradient,
    output_dir, pertype_count, total_count, s_step):

    # latent space coarse graining (none)
    images_list = [None]*total_count
    repeated_images_list = [None]* (total_count + repeat)
    i = 0
    image_inputs = create_grid(structure, w, h, 10)
    for genome_id, genome in population:
        # traverse latent space
        j = 0
        for s in range(0,pertype_count):
            s_val = -1 + s*s_step
            index = i*pertype_count+j
            image_whitebg = get_image_from_cppn(image_inputs, genome, c_dim, w, h, 10, config,
                s_val = s_val, bg = 1, gradient=gradient)
            # image_blackbg = get_image_from_cppn(image_inputs, genome, c_dim, w, h, 10, config, s_val = s_val, bg = 0)

            # save  image
            image_name = output_dir + "images/" + str(index).zfill(10) + ".png"
            image_whitebg.save(image_name, "PNG")
            # image_name = output_dir + "images/" + str(index).zfill(10) + "_black.png"
            # image_blackbg.save(image_name, "PNG")

            image = np.asarray(Image.open(image_name))

            images_list[index] = image_name
            repeated_images_list[index*repeat:(index+1)*repeat] = [image_name]*repeat

            j = j+1
        i = i + 1

    return images_list, repeated_images_list, image_inputs


def get_flows(images_list, model_name, repeated_images_list, size, channels, gpu, output_dir,
    repeat, c_dim, total_count):

    prediction_dir = output_dir + "/prediction/"
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    print("Predicting illusions...")
    skip = 1
    extension_duration = 2 #2
    # runs repeat x times on the input image, save in result folder
    test_prednet(initmodel = model_name, sequence_list = [repeated_images_list], size=size, 
                channels = channels, gpu = gpu, output_dir = prediction_dir, skip_save_frames=skip,
                extension_start = repeat, extension_duration = extension_duration,
                reset_at = repeat+extension_duration, verbose = 0, c_dim = c_dim
                )
    # calculate flows
    print("Calculating flows...")
    i = 0
    original_vectors = [None] * total_count
    for input_image in images_list:
        index_0 = int(i*(repeat/skip)+ repeat-1)
        index_1 = index_0+extension_duration-1
        prediction_0 = prediction_dir + str(index_0).zfill(10) + ".png"
        prediction_1 = prediction_dir + str(index_1).zfill(10) + "_extended.png"

        save_name = output_dir + "/images/" + str(i).zfill(10) + "_f.png"
        results = lucas_kanade(prediction_0, prediction_1, output_dir+"/flow/", save=True, verbose = 0, save_name = save_name)
        if results["vectors"]:
            original_vectors[i] = np.asarray(results["vectors"])
        else:
            original_vectors[i] = [[0,0,-1000,0]]
        i = i + 1

    return original_vectors


def get_flows_mean(images_list, size,  output_dir, c_dim):

    prediction_dir = output_dir + "/prediction/"
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    # neighborhood
    step = 10
    # actually neighborhood size
    cell_size = step*2 + 1
    x_cells = math.ceil(size[0]/cell_size)
    y_cells = math.ceil(size[1]/cell_size)
    original_vectors = [None] * len(images_list)
    # number of recognized colors
    # grain = 255

    # print("weights")
    # print(weights)
    print("Averaging images, calculatng flows")
    # average each cell with its neighbors and save the image
    index = 0
    for input_path in images_list:
        
        # bg = white by default
        bg = 0 # 255
        if c_dim == 3:
            image  = Image.open(input_path)
            new_image = Image.new('RGB', (x_cells*cell_size, y_cells*cell_size), (bg, bg, bg))
        else:
            image  = Image.open(input_path).convert("L")
            #print(image.size)
            # new_image = Image.new('L', (x_cells*cell_size, y_cells*cell_size), (bg))
            new_image = Image.new('L', (size), (bg))

        frame = cv2.imread(input_path)
        # bilateral filter: http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html
        #filters from https://qiita.com/stnk20/items/c36bef359a8f92d058b0 cv2.GaussianBlur(frame,(3,3),0)
        #average_image = cv2.bilateralFilter(frame, 15, 80, 80) #cv2.GaussianBlur(frame,(3,3),0)  # 


        # my own filter
        new_image.paste(image)#, corner)
        new_image = np.array(new_image)

        average_image = new_image #np.zeros((size[1], size[0]))
        post_grain_image = np.zeros((size[1], size[0]))

        # calculate weights based on distance
        # weight matrix for averages
        center = step
        weights = np.ones((cell_size, cell_size))
        for x in range(cell_size):
            for y in range(cell_size):
                if (x == step and y == step):
                    weights[x, y] = 2 # why
                else:
                    # weight = 1 for distance = 1
                    distance_sq = (x-step)*(x-step) + (y-step)*(y-step)
                    weights[x, y] = 1/distance_sq


        # # we do everything the opposite order of "size"
        # # blur: take weighted average (does it need to be weighted?)
        # for x in range(size[1]):
        #     for y in range(size[0]):
        #         # cell neighborhood
        #         x0 = max(0, x - step)
        #         y0 = max(0, y - step)
        #         x1 = min(size[1], x + step + 1) #subsetting leaves last number out
        #         y1 = min(size[0], y + step + 1)

        #         wx0 = step-(x-x0)
        #         wy0 = step-(y-y0)
        #         wx1 = x1-x+step
        #         wy1 = y1-y+step
        #         sub_weights = weights[wx0:wx1, wy0:wy1]

        #         # weighted average
        #         if c_dim == 3:
        #             # inverted x and y
        #             pixel = np.mean(new_image[y0:y1, x0:x1, :])
        #             average_image[y,x,:] = pixel
        #         else:                    
        #             pixel = np.sum(np.multiply(sub_weights, new_image[x0:x1, y0:y1]))
        #             factor = np.sum(sub_weights)
        #             pixel = pixel/factor
        #             # now bias the coarse graining towards average color
        #             # local color coarse graining
        #             # mean_color = np.mean(new_image[x0:x1, y0:y1])
        #             # if pixel <= mean_color:
        #             #     if mean_color == 0: # and pixel == 0
        #             #         pixel = 0
        #             #     else:
        #             #         pixel = (pixel/mean_color)*(255/2)
        #             # else:
        #             #     #print(pixel, mean_color)
        #             #     if mean_color == 255: # this branch is true sometimes because of float errors, px>255
        #             #         pixel = 255
        #             #     else:                            
        #             #         pixel = (255/2)+(pixel/(255-mean_color))*(255/2)

        #             # pixel = int(pixel)
        #             average_image[x,y] = pixel


        # local color coarse graining
        c_step = step
        for x in range(size[1]):
            for y in range(size[0]):
                pixel = average_image[x,y]
                # cell neighborhood
                x0 = max(0, x - c_step)
                y0 = max(0, y - c_step)
                x1 = min(size[1], x + c_step + 1) #subsetting leaves last number out
                y1 = min(size[0], y + c_step + 1)

                # wx0 = step-(x-x0)
                # wy0 = step-(y-y0)
                # wx1 = x1-x+step
                # wy1 = y1-y+step
                # sub_weights = weights[wx0:wx1, wy0:wy1]

                # sensitivity will be be higher around this mean color
                # mean_color = np.sum(np.multiply(sub_weights, average_image[x0:x1, y0:y1]))                
                # factor = np.sum(sub_weights)
                # mean_color = mean_color/factor

                mean_color = np.mean(average_image[x0:x1, y0:y1])
                distance_sq = 0.2*(mean_color-pixel)*(mean_color-pixel)

                if pixel >= mean_color:
                    pixel = min(255,mean_color+distance_sq)
                else:
                    pixel = max(0,mean_color-distance_sq)




                # if pixel <= mean_color:
                #     if mean_color == 0: # and pixel == 0
                #         pixel = 0
                #     else:
                #         pixel = (pixel/mean_color)*(255/2.0)
                # else:
                #     #print(pixel, mean_color)
                #     if mean_color == 255: # this branch is true sometimes because of float errors, px>255
                #         pixel = 255
                #     else:                            
                #         pixel = (255/2.0)+(pixel/(255-mean_color))*(255/2.0)

                pixel = int(pixel)
                post_grain_image[x,y] = pixel

        # # global color coarse graining (need pixel not to be int at firs)
        # # to have as many bins on each side of the average
        # mean_color = np.mean(new_image)
        # col_grain = np.mean(new_image) 
        # for x in range(size[1]):
        #     for y in range(size[0]):
        #         pixel = average_image[x,y]
        #         if pixel <= mean_color:
        #             pixel = (pixel/col_grain)*(255/2)
        #         else:
        #             pixel = (255/2)+(pixel/(255-col_grain))*(255/2) div by 0
        #         average_image[x,y] = int(pixel)


        # save
        average_image_path = output_dir + "prediction/" + str(index).zfill(10) + ".png"
        if c_dim == 3:
            av = Image.fromarray(average_image)
        else:
            av = Image.fromarray(post_grain_image)
            av = av.convert("L")

        av.save(average_image_path)

        # calculate flows
        save_name = output_dir + "/images/" + str(index).zfill(10) + "_f.png"
        results = lucas_kanade(input_path, average_image_path, output_dir+"/flow/", save=True, verbose = 0, save_name = save_name)
        if results["vectors"]:
            original_vectors[index] = np.asarray(results["vectors"])
        else:
            original_vectors[index] = [[0,0,-1000,0]]

        index = index + 1

    return original_vectors

def calculate_scores(population_size, structure, original_vectors, s_step=2):
    pertype_count = int((2/s_step))

    scores = [None] * population_size
    for i in range(0, population_size):
        final_score = -100
        temp_index = -1
        mean_score = 0
        # traverse latent space
        for j in range(0,int(2/s_step)):
            index = i*pertype_count+j
            score = 0
            score_d = 0

            if structure == StructureType.Bands:
                ratio = plausibility_ratio(original_vectors[index], 0.15) 
                score_0 = ratio[0]
                good_vectors = ratio[1]

                if(len(good_vectors)>0): 
                    y = 0                
                    count = 0
                    stripes = 4
                    step = h/stripes
                    score_direction = 0
                    discord = 0
                    orientation = 0

                    score_direction = horizontal_symmetry_score(good_vectors, [0, step*2])
                    
                    # bonus for strength
                    #score_strength = strength_number(good_vectors)
                    score_d = score_direction#*min(1,score_strength)

            elif structure == StructureType.Circles or structure == StructureType.CirclesFree :
                max_strength = 0.3 # 0.4
                ratio = plausibility_ratio(original_vectors[index], max_strength) 
                score_0 = ratio[0]
                good_vectors = ratio[1]
                min_vectors = ((2*math.pi) / (math.pi/4.0))*3
                #print("min_vectors", min_vectors, len(good_vectors))

                if(len(good_vectors)>min_vectors): 
                    # get tangent scores
                    score_direction = 0
                    limits = [0, h/2]
                    # temp = h/(2*3)
                    # limits = [temp*2, temp*3]
                    score_direction = rotation_symmetry_score(good_vectors, w, h, limits)
                    score_strength = strength_number(good_vectors,max_strength)
                    score_number = min(1, len(good_vectors)/(160*120/100))
                    score_d = 0.*score_direction + 0.3*score_strength #+ 0.3*score_number
                    # print(i, "score_direction", score_direction, "score_strength", score_strength, "final", score_d)

            elif structure == StructureType.Free:
                max_strength = 0.4
                ratio = plausibility_ratio(original_vectors[index], max_strength) 
                good_vectors = ratio[1]

                if(len(good_vectors)>0): 
                    score_strength = strength_number(good_vectors,max_strength)
                    score_number = min(len(good_vectors),15)/15
                    score_s = swarm_score(good_vectors, w, h)
                    # print("swarm_score", score_s)
                    score_d = 0.5*score_s + 0.1*score_strength + 0.4*score_number
            else:
                score_d = inside_outside_score(good_vectors, w, h)
            
            
            score = score + score_d

            if score>final_score:
                final_score = score
                temp_index = index
        
        m =  score/pertype_count
        scores[i] =[i, m]

    print("scores",scores)

    return scores


def detailed_scores(population_size, structure, original_vectors):
    scores_direction = [None] * population_size
    scores_strength = [None] * population_size
    scores_number = [None] * population_size

    for i in range(0, population_size):
        final_score = -100
        temp_index = -1
        mean_score = 0
        # traverse latent space
        index = i
        score = 0
        score_d = 0

        if structure == StructureType.Circles or structure == StructureType.CirclesFree :
            max_strength = 0.4 # 0.4
            ratio = plausibility_ratio(original_vectors[index], max_strength) 
            score_0 = ratio[0]
            good_vectors = ratio[1]
            #min_vectors = ((2*math.pi) / (math.pi/4.0))*3
            #print("min_vectors", min_vectors, len(good_vectors))

            if(len(good_vectors)>0): 
                # get tangent scores
                score_direction = 0
                limits = [0, h/2]
                # temp = h/(2*3)
                # limits = [temp*2, temp*3]
                scores_direction[i] = rotation_symmetry_score(good_vectors, w, h, limits)
                scores_strength[i] = strength_number(good_vectors,max_strength)
                scores_number[i] = min(1, len(good_vectors)/(160*120/400))
                #score_d = 0.*score_direction + 0.3*score_strength #+ 0.3*score_number
                # print(i, "score_direction", score_direction, "score_strength", score_strength, "final", score_d)
            else:
                scores_direction[i] = 0
                scores_strength[i] = 0
                scores_number[i] = 0

    return scores_direction, scores_strength, scores_number

# population:  [id, net]
def get_fitnesses_neat(structure, population, model_name, config, w, h, channels,
    id=0, c_dim=3, best_dir = ".", gradient = 1, gpu = 0):

    print("Calculating fitnesses of populations: ", len(population))
    output_dir = "temp/" 
    repeat = 20
    half_h = int(h/2)
    size = [w,h]
    skip = 1
    
    if not os.path.exists(output_dir + "images/"):
        os.makedirs(output_dir + "images/")

    s_step = 2
    pertype_count = int((2/s_step))
    total_count = len(population)*pertype_count
    images_list, repeated_images_list, image_inputs = cppn_evolution(population, repeat,  structure, w, h, gpu, config, c_dim, gradient,
        output_dir, pertype_count, total_count, s_step)

    # using prednet
    # original_vectors = get_flows(images_list, model_name, repeated_images_list, size, channels, gpu, output_dir,
    # repeat, c_dim, total_count)

    # using mean
    original_vectors = get_flows_mean(images_list, size, output_dir, c_dim)

    scores = calculate_scores(len(population), structure, original_vectors, s_step)

    i = 0
    best_score = 0
    best_illusion = 0
    best_genome = None
    for genome_id, genome in population:
        genome.fitness = scores[i][1]
        if (scores[i][1]> best_score):
            best_illusion = i
            best_score = scores[i][1]
            best_genome = genome
        i = i+1

    # save best illusion
    image_name = output_dir + "/images/" + str(best_illusion).zfill(10) + ".png"
    print("best", image_name, best_illusion)
    move_to_name = best_dir + "/best.png"
    shutil.copy(image_name, move_to_name)
    index = int(best_illusion*(repeat/skip) + repeat-1)
    image_name = output_dir + "/images/" + str(best_illusion).zfill(10) + "_f.png"
    move_to_name = best_dir + "/best_flow.png"
    shutil.copy(image_name, move_to_name)

    image_blackbg = get_image_from_cppn(image_inputs, best_genome, c_dim, w, h, 10, config,
        s_val = -1, bg = 0, gradient=gradient)
    image_name = best_dir + "/best_black_bg.png"
    image_blackbg.save(image_name, "PNG")
    
    # create enhanced image
    e_w = 800
    e_h = 800
    e_grid = enhanced_image_grid(e_w, e_h, structure)
    image = get_image_from_cppn(e_grid, population[best_illusion][1], c_dim, e_w, e_h, 10, config,
        s_val = -1, bg = 1, gradient=gradient)

    image_name = best_dir + "/enhanced.png"
    image.save(image_name)

def get_random_pixels(w, h, c_dim):
    img_data = np.random.rand(w,h,c_dim) 
    img_data = np.round(img_data*255.0)
    
    return img_data

def fix_values(pixel):
    for i in range(pixel.shape[0]):
        if(pixel[i]<0):
            pixel[i] = -pixel[i]
        elif pixel[i]>255:
            pixel[i] = 255*2 - pixel[i]

def mutate_genome(genome):
    n = max(0, genome["number_mutation"] + round(random()))
    r =  max(0, genome["range_mutation"] + round(random()))
    rad =  max(0, genome["radius"] + round(random()))

    mutated_genome = {"number_mutation": n, "range_mutation": r, "radius": rad}
    return mutated_genome

def mutate_pixels(input_image, c_dim, genome):
    mutated = np.copy(input_image).reshape((input_image.shape[0],input_image.shape[1],c_dim))

    #mutate genome first
    mutated_genome = mutate_genome(genome)
    number_mutation = mutated_genome["number_mutation"]
    range_mutation = mutated_genome["range_mutation"]
    radius = mutated_genome["radius"]

    #print(n)
    #generate a number of random starting points
    points = np.random.rand(number_mutation,2)

    points[:,0] = points[:,0]*(mutated.shape[0])
    points[:,1] = points[:,1]*(mutated.shape[1])
    points = points.astype(int)

    
    for c in range(number_mutation):
        x = points[c,0]-radius
        y = points[c,1]-radius
        x_stop = x + radius
        y_stop = y + radius

        if(x<0):
            x=0
        if(y<0):
            y=0

        if(x_stop>mutated.shape[0]):
            x_stop = mutated.shape[0]
        if(y_stop>mutated.shape[1]):
            y_stop = mutated.shape[1]

        old_pixels = mutated[x:x_stop,y:y_stop,:]
        
        if c_dim == 3:
            mutation = range_mutation - np.random.rand(3)*range_mutation*2
        else:
            mutation = range_mutation - np.random.rand(1)*range_mutation*2

        new_pixels = old_pixels + mutation

        for ix in range(x_stop-x):
            for iy in range(y_stop-y):
                fix_values(new_pixels[ix,iy,:])

        mutated[x:x_stop,y:y_stop,:] = new_pixels

    if c_dim == 3:
        image =  Image.fromarray(np.array(mutated,dtype=np.uint8))
    else:
        mutated = mutated.reshape((mutated.shape[0],mutated.shape[1]))
        image =  Image.fromarray(np.array(mutated,dtype=np.uint8), 'L')

    return image, mutated_genome
    

def pixel_evolution(population_size, output_dir, model_name, channels, c_dim, structure, start_image, gpu=0):
    output_dir = "temp/" 
    repeat = 20
    half_h = int(h/2)
    size = [w,h]
    mutation_rate = 0.15
    best_dir = output_dir
    skip = 1
    half_population = int(population_size/2)

    radius = 10
    n = 50 #int(np.round(w*h/(radius*2*radius*2)))
    winning_genome = {"number_mutation": n, "range_mutation": 50, "radius": radius}

    if not os.path.exists(output_dir + "images/"):
        os.makedirs(output_dir + "images/")

    repeated_images_list = [None]* (repeat)
    images_list = [None]*(population_size)
    mutated_genomes = [None]*population_size

    
    best_score = 0
    best_best_score = 0
    best_illusion = 0
    best_coverage = 0
    best_best_coverage = 0

    if start_image!="":
        #best_image_pixels = Image.open(start_image).convert('RGB')
        if c_dim==3:
            image = Image.open(start_image).convert('RGB')
        else:
             image = Image.open(start_image).convert('L')
        best_image_pixels = np.array(image)
        save_to_name = best_dir + "/best.png"
        shutil.copy(start_image, save_to_name)
    else:
        img_data = np.ones((h,w,c_dim)) 
        best_image_pixels = np.round(img_data*255.0)
        if c_dim == 3:
            image = Image.fromarray(np.array(best_image_pixels,dtype=np.uint8))
        else:
            best_image_pixels_temp = best_image_pixels.reshape((h,w))
            image = Image.fromarray(np.array(best_image_pixels_temp,dtype=np.uint8), 'L')
        save_to_name = best_dir + "/best.png"
        image.save(save_to_name, "PNG")

    generation = 0
    species_genomes_0 = winning_genome
    species_genomes_1 = winning_genome
    secondary_image = best_image_pixels
    while True:
        print("generation", generation)
        generation = generation+1
        
        for i in range(population_size):
            if(i==0) and (generation==1):
                image_modified = image
            else:
                # divide all images into 2 species?
                if(i<half_population):
                    # image_modified, mutated_genomes[i] = mutate_pixels(best_image_pixels, c_dim, winning_genome)
                    image_modified, mutated_genomes[i] = mutate_pixels(best_image_pixels, c_dim, species_genomes_0)
                    # dont' mutate the genome yet
                    mutated_genomes[i] = species_genomes_0
                else:
                    image_modified, mutated_genomes[i] = mutate_pixels(secondary_image, c_dim, species_genomes_1)
                    # dont' mutate the genome yet
                    mutated_genomes[i] = species_genomes_1

            # save  image
            image_name = output_dir + "images/" + str(i).zfill(10) + ".png"
            image_modified.save(image_name, "PNG")
            # image_name = output_dir + "images/" + str(index).zfill(10) + "_black.png"
            # image_blackbg.save(image_name, "PNG")

            image = np.asarray(Image.open(image_name))

            images_list[i] = image_name
            repeated_images_list[i*repeat:(i+1)*repeat] = [image_name]*repeat

        original_vectors = get_flows(images_list, model_name, repeated_images_list, size, channels, gpu, output_dir,
        repeat, c_dim, population_size)

        #scores = calculate_scores(population_size, structure, original_vectors)

        scores_direction, scores_strength, scores_number = detailed_scores(population_size, structure, original_vectors)

        # calculate failure coverage
        score_d0 = np.var(scores_direction[0:half_population])
        score_d1 = np.var(scores_direction[half_population:population_size])
        score_s0 = np.var(scores_strength[0:half_population])
        score_s1 = np.var(scores_strength[half_population:population_size])
        score_n0 = np.var(scores_number[0:half_population])
        score_n1 = np.var(scores_number[half_population:population_size])

        score_0 = score_d0+score_s0+score_n0
        score_1 = score_d1+score_s1+score_n1
        print("scores_direction", scores_direction)
        print("scores_strength", scores_strength)
        print("scores_number", scores_number)
        print("failure coverage", score_0, score_1)

        # take highest variance
        best_coverage = score_1
        if score_0 >= score_1:
            pstart = 0
            best_coverage = score_0;
        else:
            pstart = half_population

        best_score = 0
        best_genome = winning_genome
        for i in range(pstart, pstart+half_population):
            score = scores_direction[i] + scores_strength[i]
            if (score >= best_score):
                best_illusion = i
                best_score = score 

                #best_genome = mutated_genomes[i]
                # prevent overrite that seems to happen

        
        move_to_name = "temp.png"
        shutil.copy(images_list[best_illusion], move_to_name)
        print("moved", images_list[best_illusion], "temp.png")
        move_to_name = "temp_f.png"
        image_name = output_dir + "/images/" + str(best_illusion).zfill(10) + "_f.png"
        shutil.copy(image_name, move_to_name)

        print("best_illusion", best_illusion, "best_score", best_score , "best_best_score", best_best_score)
        print("best_best_coverage", best_best_coverage)

        #if(best_score>=best_best_score):
        if best_coverage >= best_best_coverage and best_illusion>=half_population:

            print("new best coverage")
            print("******changing best illusion")
            best_best_score = best_score
            winning_genome = best_genome
            best_best_coverage = best_coverage

            # save best illusion
            image_name = "temp.png"
            print("best", image_name, best_illusion)
            move_to_name = best_dir + "/best.png"

            Image.open(image_name).save(move_to_name)
            print("moved", image_name, move_to_name)

            image_name = "temp_f.png"
            move_to_name = best_dir + "/best_flow.png"
            # shutil.copy(image_name, move_to_name)
            Image.open(image_name).save(move_to_name)
            print("moved", image_name, move_to_name)
            #files.download(move_to_name)

            best_image_pixels = secondary_image

        secondary_image = np.asarray(Image.open(images_list[best_illusion]))

        print("best score", best_best_score)
        print("winning_genome", winning_genome)

def neat_illusion(output_dir, model_name, config_path, structure, w, h, channels, c_dim =3, checkpoint = None, gradient=1, gpu = 0):
    repeat = 6
    limit = 1
    half_h = int(h/2)
    size = [w,h]

    best_dir = output_dir 
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    def eval_genomes(genomes, config):
        get_fitnesses_neat(structure, genomes, model_name, config, w, h, channels,
            c_dim=c_dim, best_dir=best_dir, gradient=gradient, gpu = gpu)

    checkpointer = neat.Checkpointer(100)

    # Create the population, which is the top-level object for a NEAT run.
    if not checkpoint:
        p = neat.Population(config)
    else:
        p = checkpointer.restore_checkpoint(checkpoint)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(checkpointer)

    # Run for up to x generations.
    winner = p.run(eval_genomes, 300)


def string_to_intarray(string_input):
    array = string_input.split(',')
    for i in range(len(array)):
        array[i] = int(array[i])

    return array

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate illusions')
    parser.add_argument('--model', '-m', default='', help='.model file')
    parser.add_argument('--output_dir', '-o', default='.', help='path of output diectory')
    parser.add_argument('--structure', '-s', default=0, type=int, help='Type of illusion. 0: Bands; 1: Circles; 2: Free form')
    parser.add_argument('--config', '-cfg', default="", help='path to the NEAT config file')
    parser.add_argument('--checkpoint', '-cp', help='path of checkpoint to restore')
    parser.add_argument('--size', '-wh', help='big or small', default="small")
    parser.add_argument('--color_space', '-c', help='1 for greyscale, 3 for rgb', default=3, type=int)
    # [1,16,32,64]
    # 3,48,96,192
    parser.add_argument('--channels', '-ch', default='3,48,96,192', help='Number of channels on each layers')
    parser.add_argument('--gradient', '-g', default=1, type=int, help='1 to use gradients, 0 for pure colors')
    parser.add_argument('--pixels', '-px', default=-1, type=int, help='-1 to use cppn, 0 for pixel evolution')
    parser.add_argument('--start', default="", help='pixel image to use as starting point')


    args = parser.parse_args()
    output_dir = args.output_dir 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    w = 160
    h = 120
    if args.size == "big":
        w = 640

        h = 480

    config = args.config

    if config == "":
        config = os.path.dirname(__file__)
        print(config)
        if args.structure == StructureType.Bands:
            config += "/neat_configs/bands.txt"
        elif args.structure == StructureType.Circles or args.structure == StructureType.CirclesFree:
            config += "/neat_configs/circles.txt"
        elif args.structure == StructureType.Free:
            config += "/neat_configs/free.txt"
        else :
            config += "/neat_configs/default.txt"
        
    print("config", config)
    print("gradient", args.gradient)

    gpu = -1

    if(args.pixels<0):
        neat_illusion(output_dir, args.model,config, args.structure, w, h, string_to_intarray(args.channels),
        args.color_space, args.checkpoint, args.gradient, gpu)
    else:
        population_size = 20
        pixel_evolution(population_size, output_dir, args.model,string_to_intarray(args.channels),
        args.color_space, args.structure, args.start, gpu)

