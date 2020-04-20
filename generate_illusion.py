import argparse
from chainer_prednet.PredNet.call_prednet import test_prednet
from chainer_prednet.utilities.mirror_images import mirror, mirror_multiple, TransformationType
import cv2
import csv
from enum import IntEnum
import math
import neat
import numpy as np
from optical_flow.optical_flow import lucas_kanade
import os
from PIL import Image
from pytorch_neat.pytorch_neat.cppn import create_cppn
from pytorch_neat.pytorch_neat.multi_env_eval import MultiEnvEvaluator
from pytorch_neat.pytorch_neat.neat_reporter import LogReporter
from pytorch_neat.pytorch_neat.recurrent_net import RecurrentNet
from random import random, randrange
import shutil
import shutil
import torch


# TODO enumerate illusion types
class StructureType(IntEnum):
    Bands = 0
    Circles = 1
    Free = 2

# high score if vectors pass the mirror test
def illusion_score(vectors, flipped=False, mirrored=False):
    # check vector alignements
    comp_x = 0
    count = 0
    for vector in vectors:
        # normalize
        norm = np.sqrt(vector[2]*vector[2] + vector[3]*vector[3])

        # print("norm", norm)
        if norm> 0.15 or norm==0: 
            continue

        if mirrored:
            comp_x = comp_x + (-vector[2]/norm)
        else:
            comp_x = comp_x + vector[2]/norm

    # minimize comp_y, maximize comp_x
    score = comp_x
    return score

# returns ratio and vectors that are not unplausibly big
def plausibility_ratio(vectors):
    r = []
    for vector in vectors:
        norm = np.sqrt(vector[2]*vector[2] + vector[3]*vector[3])
        if norm> 0.15: # or norm==0: 
            continue
        r.append(vector)

    ratio = len(r)/len(vectors)
    return [ratio, r]

#returns mean of vectors norms
def strength_number(vectors):
    sum_v = 0
    total_v = 0

    for vector in vectors:
        norm = np.sqrt(vector[2]*vector[2] + vector[3]*vector[3])
        sum_v = sum_v + norm
        total_v = total_v +1
    
    return sum_v/total_v

# returns the mirroring score (lower == better) 
def mirroring_score(vectors, m_vectors):
    # print("vectors", vectors)
    sum_v = [0,0]
    for vector in vectors:
        sum_v = [sum_v[0] + vector[2], sum_v[1] + vector[3]]

    sum_mv = [0,0]
    for vector in m_vectors:
        sum_mv = [sum_mv[0] + vector[2], sum_mv[1] + vector[3]]

    s0x = sum_v[0] + sum_mv[0]
    s0y = sum_v[1] + sum_mv[1]

    return abs(s0x) + abs(s0y)

# return the mirrored score on x and y, 
# the global strength of all plausible vectors, 
# and the ratio of plausible vectors vs too big vectors
def combined_illusion_score(vectors, m_vectors):
    # check vector alignements
    sum_v = [0,0]
    total_v = 0
    for vector in vectors:
        norm = np.sqrt(vector[2]*vector[2] + vector[3]*vector[3])
        if norm> 0.15 or norm==0: 
            continue
        sum_v = [sum_v[0] + vector[2], sum_v[1] + vector[3]]
        total_v = total_v +1

    sum_mv = [0,0]
    total_mv = 0
    for vector in m_vectors:
        norm = np.sqrt(vector[2]*vector[2] + vector[3]*vector[3])
        if norm> 0.15 or norm==0: 
            continue
        sum_mv = [sum_mv[0] + vector[2], sum_mv[1] + vector[3]]
        total_mv = total_mv +1

    s0x = sum_v[0] + sum_mv[0]
    s0y = sum_v[1] + sum_mv[1]
    s1 = abs(sum_v[0]) +  abs(sum_v[1]) +  abs(sum_mv[0]) +  abs(sum_mv[1])
    s2 = total_v + total_mv
    if s2 == 0:
        s2 = 0.01
    else:
        s2 = s2 / (len(vectors) + len(m_vectors))

    return [s0x + s0y, s1, s2]

# returns [a,b]
# a = 1 if vectors rather aligned on x to the right;  -1 if to the left
# b = mean of projection on x axis (normalised)

def direction_ratio(vectors, limits = None):
    # print(vectors)
    mean_ratio = 0
    count = 0
    orientation = 0
    # make sure that all vectors are on x axis
    for v in vectors:
        if not limits is None:
            if (v[1]<limits[0]) or (v[1]>limits[1]):
                continue
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
# [ratio of tangent, ratio of alignment]
# [a,b]
# a = 1 if vectors rather aligned clockwise;  -1 if counterclockwise
# b = mean of projection on tangent (normalised)
def tangent_ratio(vectors, limits = None):
    w = 160
    h = 120
    direction = 0
    mean_alignment = 0
    # if beta = angle between radius and current vector
    # ratio of projection of V on tangent / ||V|| = sin(beta)
    # ratio = sin(arcos(R*V/||V||*||R||)) = sqrt(1- a^2)
    count = 0
    for v in vectors:
        # radius vector R from image center to origin of V
        r = [c[0], c[1], v[0]-c[0], v[1]-c[1]]
        norm_r = np.sqrt(r[2]*r[2] + r[3]*r[3])
        norm_v = np.sqrt(v[2]*v[2] + v[3]*v[3])
        if not limits is None:
            if (norm_r<limits[0]) or (norm_r>limits[1]):
                continue

        # global_sum = [global_sum[0] + v[2], global_sum[1]+v[3]]
        # abs_sum = [abs_sum[0] + abs(v[2]), abs_sum[1]+ abs(v[3])]
        # sum_norm = sum_norm + norm_v
        # projection of vectors on each other a = V*R / ||V||*||R||
        a = r[2] * v[2] + r[3] * v[3]
        # was this cause of NaN?
        if(norm_r*norm_v==0):
            a = 0
        else:
            a = a/(norm_r * norm_v)
        mean_alignment = mean_alignment + a
        # need the sign of the angle for orientation of vector
        # if(a>0):
        #     # ratio
        #     ratio = np.sqrt(1 - a*a)
        #     mean_ratio = mean_ratio + ratio
        count = count + 1

    if mean_alignment > 0:
        direction = 1
    elif mean_alignment <0:
        direction = -1

    if count > 0:
        mean_alignment = mean_alignment/count

    return [direction, mean_alignment]



# returns a high score if vectors are aligned on concentric circles
# [ratio of tangent, ratio of alignment]
def circle_tangent_ratio(vectors, limits = None):
    w = 160
    h = 120
    c = [w/2.0, h/2.0]
    mean_ratio = 0
    global_sum= [0,0]
    abs_sum = [0,0]
    sum_norm = 0
    # if beta = angle between radius and current vector
    # ratio of projection of V on tangent / ||V|| = sin(beta)
    # ratio = sin(arcos(R*V/||V||*||R||)) = sqrt(1- a^2)
    count = 0
    for v in vectors:
        # radius vector R from image center to origin of V
        r = [c[0], c[1], v[0]-c[0], v[1]-c[1]]
        norm_r = np.sqrt(r[2]*r[2] + r[3]*r[3])
        norm_v = np.sqrt(v[2]*v[2] + v[3]*v[3])
        if not limits is None:
            if (norm_r<limits[0]) or (norm_r>limits[1]):
                continue

        global_sum = [global_sum[0] + v[2], global_sum[1]+v[3]]
        abs_sum = [abs_sum[0] + abs(v[2]), abs_sum[1]+ abs(v[3])]
        sum_norm = sum_norm + norm_v
        # projection of vectors on each other a = V*R / ||V||*||R||
        a = r[2] * v[2] + r[3] * v[3]
        a = a/(norm_r * norm_v)
        # need the sign of the angle for orientation of vector
        if(a>0):
            # ratio
            ratio = np.sqrt(1 - a*a)
            mean_ratio = mean_ratio + ratio
        count = count + 1

    if count > 0:
        mean_ratio = mean_ratio/count
    else:
        mean_ratio = 0

    if sum_norm == 0:
        s_sum = 1000
    else:
        s_sum = abs(global_sum[0]/abs_sum[0]) + abs(global_sum[1]/abs_sum[1])
        s_sum = s_sum/2

    return [mean_ratio,s_sum]


def generate_random_image(w, h):
    image = np.random.randint(256, size=(w, h, 3))
    return np.uint8(image)

def random_modify(image_path):
    image = np.array(Image.open(image_path).convert('RGB'))

    w = image.shape[0]
    h = image.shape[1]
    c_range = 50

    for x in range(0,500):
        i = randrange(w)
        j = randrange(h)
        color = randrange(3)
        sign = random()

        pixel = image[i,j]
        if sign>=0.5:
            pixel[color] = pixel[color] + randrange(c_range)
            if pixel[color] > 255 : pixel[color] = 255
        else:
            pixel[color] = pixel[color] - randrange(c_range) 
            if pixel[color] < 0  : pixel[color] = 0

    return image


def create_grid(structure, x_res = 32, y_res = 32, scaling = 1.0):

    r_mat = None 
    x_mat = None
    y_mat = None
    num_points = x_res*y_res
   
    if structure == StructureType.Bands:
        y_rep = 5
        y_len = int(y_res/y_rep) 
        sc = scaling/y_rep
        a = np.linspace(-1*sc, sc, num = y_len)
        y_range = np.tile(a, y_rep)
        x_range = np.linspace(-1*scaling, scaling, num = x_res)

        x_reverse = np.ones((y_res, 1))
        start = 0
        while start<y_res:
            stop = min(y_res, start+y_len)
            x_reverse[start:stop] =  -x_reverse[start:stop]
            start = start+2*y_len

        x_mat = np.matmul(x_reverse, x_range.reshape((1, x_res)))
        y_mat = np.matmul(y_range.reshape((y_res, 1)), np.ones((1, x_res)))
        x_mat = np.tile(x_mat.flatten(), 1).reshape(1, num_points, 1)
        y_mat = np.tile(y_mat.flatten(), 1).reshape(1, num_points, 1)

        return {"x_mat": x_mat, "y_mat": y_mat} 

    elif structure == StructureType.Circles:
        r_rep = 3
        r_len = int(y_res/(2*r_rep))
        x_range = np.linspace(-1*scaling, scaling, num = x_res)
        y_range = np.linspace(-1*scaling, scaling, num = y_res)

 
        y_mat = np.matmul(y_range.reshape((y_res, 1)), np.ones((1, x_res)))
        x_mat = np.matmul(np.ones((y_res, 1)), x_range.reshape((1, x_res)))

        # r_sign = np.ones((int(y_res/2),1))
        # start = 0
        # while start<int(y_res/2):
        #     stop = min(r_rep*r_len, start+r_len)
        #     r_sign[start:stop] =  -r_sign[start:stop]
        #     start = start+2*r_len

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

                    # focus on 1 small pattern
                    theta = theta % (math.pi/6.0)

                x_mat[yy,xx] = r 
                y_mat[yy,xx] = theta 

        return {"x_mat": x_mat, "y_mat": y_mat}

    return {"input_0": x_mat, "input_1": y_mat, "input_2": r_mat} #, s_mat

def fully_connected(input, out_dim, with_bias = True, mat = None):
    if mat is None:
        mat = np.random.standard_normal(size = (input.shape[1], out_dim)).astype(np.float32)

    result = np.matmul(input, mat)

    if with_bias == True:
        bias = np.random.standard_normal(size =(1, out_dim)).astype(np.float32)
        result += bias * np.ones((input.shape[0], 1), dtype = np.float32)

    return result

def get_fidelity(input_image_path, prediction_image_path):
    input_image = np.array(Image.open(input_image_path).convert('RGB'))
    prediction = np.array(Image.open(prediction_image_path).convert('RGB'))

    err = np.sum((input_image.astype("float") - prediction.astype("float")) ** 2)
    err /= (float(input_image.shape[0] * input_image.shape[1])*255*255)
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return 1-err

def get_image_from_cppn(structure, genome, c_dim, w, h, config, s_val = 1):

    scaling = 10
   
    # why twice???
    out_names = ["r0","g0","b0","r1","g1","b1"]

    inputs = create_grid(structure, w, h, scaling)
    # x_dat, y_dat, r_dat  = create_grid(w, h, scaling)
    # s_dat = s_val*s_dat

    # if structure == StructureType.Bands:
    leaf_names = ["x","y"]
    x_dat = inputs["x_mat"]
    y_dat = inputs["y_mat"]
    inp_x = torch.tensor(x_dat.flatten())
    inp_y = torch.tensor(y_dat.flatten())
    # else :
    #     leaf_names = ["x","y","s","r"]
    
    # inp_s = torch.tensor(s_dat.flatten())
    # inp_minus_s = torch.tensor(-s_dat.flatten())
    #reverse x
    #inp_r = torch.tensor(r_dat.flatten())

    if(c_dim>1):
            image_array = np.zeros(((h,w,3)))
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

                # if structure == StructureType.Bands:
                pixels = node_func(x=inp_x, y=inp_y)
                # else: 
                #     pixels = node_func(x=inp_x, y=inp_y, s = inp_s, r = inp_r)
                
                pixels_np = pixels.numpy()
                # pixels = node_func(x=inv_x, y=inp_y, s = inp_s)
                # reverse_pixels_np = pixels.numpy()
                # for x_slice in range(0,x_rep):
                #     start = x_slice*x_subwidth
                #     image_array[0:half_h, start:(start+x_subwidth), c] = np.reshape(pixels_np, (half_h,x_subwidth))
                #     image_array[half_h:h, start:(start+x_subwidth), c] = np.reshape(reverse_pixels_np, (half_h,x_subwidth))

                image_array[0:h, 0:w, c] = np.reshape(pixels_np, (h,w))

                c = c + 1
            img_data = np.array(image_array*255.0, dtype=np.uint8)
            image =  Image.fromarray(img_data)#, mode = "HSV")
            #image = image.convert(mode="RGB")
    else:
        net_nodes = create_cppn(
            genome,
            config,
            leaf_names,
            out_names
        )
        node_func = net_nodes[0]
        pixels = node_func(x=inp_x, y=inp_y, s = inp_s, r = inp_r)
        pixels_np = pixels.numpy()
        image_array = np.zeros(((w,h,3)))
        pixels_np = np.reshape(pixels_np, (w, h)) * 255.0
        image_array[:,:,0] = pixels_np
        image_array[:,:,1] = pixels_np
        image_array[:,:,2] = pixels_np
        img_data = np.array(image_array, dtype=np.uint8)
        image =  Image.fromarray(np.reshape(img_data,(h,w,3)))

    return image

# population:  [id, net]
def get_fitnesses_neat(structure, population, model_name, config, id=0, c_dim=3, best_dir = "."):
    print("Calculating fitnesses of populations: ", len(population))
    output_dir = "temp/" 
    repeat = 10
    w = 160
    h = 120
    half_h = int(h/2)
    size = [w,h]
    channels = [3,48,96,192]
    gpu = 0

    prediction_dir = output_dir + "/prediction/"
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    if not os.path.exists(output_dir + "images/"):
        os.makedirs(output_dir + "images/")

    # latent space coarse graining (none)
    s_step = 2
    pertype_count = int((2/s_step))
    total_count = len(population)*pertype_count
    images_list = [None]*total_count
    repeated_images_list = [None]* (total_count + repeat)
    i = 0
    for genome_id, genome in population:
        # traverse latent space
        j = 0
        for s in range(0,pertype_count):
            s_val = -1 + s*s_step
            index = i*pertype_count+j
            image = get_image_from_cppn(structure, genome, c_dim, w, h, config, s_val = s_val)

            image_name = output_dir + "images/" + str(index).zfill(10) + ".png"
            images_list[index] = image_name
            repeated_images_list[index*repeat:(index+1)*repeat] = [image_name]*repeat
            image.save(image_name, "PNG")
            j = j+1
        i = i + 1

    print("Predicting illusions....")
    # runs repeat x times on the input image, save in result folder
    test_prednet(initmodel = model_name, sequence_list = [repeated_images_list], size=size, 
                channels = channels, gpu = gpu, output_dir = prediction_dir, skip_save_frames=repeat,
                reset_each = True, verbose = 0
                )
    # calculate flows
    i = 0
    original_vectors = [None] * total_count
    for input_image in images_list:
        prediction_image_path = prediction_dir + str(i).zfill(10) + ".png"
        results = lucas_kanade(input_image, prediction_image_path, output_dir+"/flow/", save=True, verbose = 0)
        if results["vectors"]:
            original_vectors[i] = np.asarray(results["vectors"])
        else:
            original_vectors[i] = [[0,0,-1000,0]]
        i = i + 1

    # # mirror images
    # print("Mirror images")
    # mirror_multiple(output_dir + "images/" , output_dir + "mirrored_images/" , TransformationType.MirrorAndFlip)
    # # mirrored flows
    # i = 0
    # mirrored_vectors = [None] * total_count
    # for input_image in images_list:
    #     prediction_image_path = prediction_dir + str(i).zfill(10) + ".png"
    #     results = lucas_kanade(input_image, prediction_image_path, output_dir+"/flow/", save=True, verbose = 0)
    #     if results["vectors"]:
    #         original_vectors[i] = np.asarray(results["vectors"])
    #     else:
    #         original_vectors[i] = [[0,0,-1000,0]]
    #     i = i + 1


    # calculate score
    #radius_limits = [20,50]
    scores = [None] * len(population)
    for i in range(0, len(population)):
        final_score = -100
        temp_index = -1
        mean_score = 0
        # traverse latent space
        for j in range(0,int(2/s_step)):
            index = i*pertype_count+j
            score = 0
            if(len(original_vectors[index])>0):
                # bonus
                # score = score + 0.1
                ratio = plausibility_ratio(original_vectors[index]) #TODO might not be needed?
                score_0 = ratio[0]
                good_vectors = ratio[1]

                if(len(good_vectors)>0): 
                    # score = score + 0.5*min(len(good_vectors),5)
                #     step = h/2
                #     y = 0                
                #     count = 0
                #     score_2 = [None]*2
                #     while y<h:
                #         limit = [y, y+step]
                #         score_2[count] =  direction_ratio(good_vectors, limit)
                #         y = y + step
                #         count = count + 1

                #     # bonus points
                #     if(score_2[0]*score_2[1]<0):
                #         # is the ideal number of vectors
                #         temp = 24 - len(good_vectors)
                #         if(temp==0):
                #             n_dist = 1
                #         else:
                #             n_dist = 1/temp*temp
                #         score = score + n_dist*(abs(score_2[0]) + abs(score_2[1]))/2
                #         mean_score = mean_score + score
                # if score>final_score:
                #     final_score = score
                #     temp_index = index

                    if structure == StructureType.Bands:
                        y = 0                
                        count = 0
                        stripes = 5
                        step = h/stripes
                        score_direction = 0
                        discord = 0
                        orientation = 0
                        while y<h:
                            limits = [y, y+step]
                            dir_ratio =  direction_ratio(good_vectors, limits)
                            #check mirroring
                            if(count==1):
                                if not(orientation>0 and dir_ratio[0]<0):
                                    score_direction = 0
                                    break
                            orientation = dir_ratio[0]

                            factor = 1
                            if count % 2 == 1:
                                factor = -1
                            score_direction = score_direction + factor*dir_ratio[1]

                            y = y + step
                            count = count + 1
                        
                        score_direction = score_direction / stripes
                        # bonus for strength
                        score_strength = strength_number(good_vectors)
                        score_d= score_direction*score_strength

                    elif structure == StructureType.Circles:
                        # get tangent scores
                        r = 0                
                        count = 0
                        stripes = 3
                        step = h/stripes
                        score_direction = 0
                        discord = 0
                        orientation = 0
                        while r<h/2:
                            limits = [r, r+step]
                            dir_ratio =  tangent_ratio(good_vectors, limits)                    
                            score_direction = score_direction + abs(dir_ratio[1])
                            r = r + step
                            count = count + 1
                        
                        score_direction = score_direction / stripes
                        # bonus for strength
                        score_strength = strength_number(good_vectors)
                        score_d= score_direction*score_strength

                    else:
                        score_d = inside_outside_score(good_vectors, w, h)

                    # divergence_convergence_score(good_vectors, w, h)

                    score = score + score_d

                if score>final_score:
                    final_score = score
                    temp_index = index
        
        m =  score/pertype_count
        scores[i] =[i, m]

    print("scores",scores)
    i = 0
    best_score = 0
    best_illusion = 0
    for genome_id, genome in population:
        genome.fitness = scores[i][1]
        if (scores[i][1]> best_score):
            best_illusion = i
            best_score = scores[i][1]
        i = i+1

    # save best illusion
    image_name = images_list[best_illusion] #output_dir + "/images/" + str(best_illusion).zfill(10) + ".png"
    move_to_name = best_dir + "/best.png"
    shutil.copy(image_name, move_to_name)
    print("best", image_name, best_illusion)
    image_name = output_dir + "/flow/" + str(best_illusion).zfill(10) + ".png"
    move_to_name = best_dir + "/best_flow.png"
    shutil.copy(image_name, move_to_name)


def neat_illusion(output_dir, model_name, config_path, structure, checkpoint = None):
    repeat = 6
    limit = 1
    w = 160
    h = 120
    half_h = int(h/2)
    size = [w,h]
    channels = [3,48,96,192]
    gpu = 0
    c_dim = 3

    best_dir = output_dir
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    def eval_genomes(genomes, config):
        get_fitnesses_neat(structure, genomes, model_name, config, c_dim=c_dim, best_dir=best_dir)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='optical flow tests')
    parser.add_argument('--model', '-m', default='', help='.model file')
    parser.add_argument('--output_dir', '-o', default='.', help='path of output diectory')
    parser.add_argument('--structure', '-s', default=0, type=int, help='Type of illusion. 0: Bands; 1: Circles; 2: Free form')
    parser.add_argument('--config', '-cfg', default="", help='path to the NEAT config file')
    parser.add_argument('--checkpoint', '-cp', help='path of checkpoint to restore')


    args = parser.parse_args()
    output_dir = args.output_dir 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config = args.config

    if config == "":
        config = os.path.dirname(__file__)
        print(config)
        if args.structure == StructureType.Bands:
            config += "/neat_configs/bands.txt"
        elif args.structure == StructureType.Circles:
            config += "/neat_configs/circles.txt"
        else :
            config += "/neat_configs/default.txt"
        
    print("config", config)
    neat_illusion(output_dir, args.model,config, args.structure, args.checkpoint)

