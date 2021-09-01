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



# rotates an image by `angle` until it reaches 360 degrees
# save the rotations and return the list of file names
# index is the file numer name for saving
# angle = clockwise rotation
def full_rotation(image, angle, output_dir, index=0):
    rotations = int(360/angle)
    rotated_images_list = [None]*rotations

    image_name = output_dir + "/" + str(index).zfill(3) + ".png"
    rotated_images_list[0] = image_name 

    for i in range(1,rotations):
        rotated_image = image.rotate(-angle*i, expand=True)
        image_name = output_dir + "/" + str(index+i).zfill(3) + ".png"
        rotated_image.save(image_name)
        rotated_images_list[i] = image_name

    return rotated_images_list


# repeats a list to fill another bigger list
# returns filled-up list
# total_repeat = expected size of filled-up list
def get_repeat_rotation_list(partial_list, total_repeat):
    rotations = len(partial_list)
    full_list = [None]*total_repeat

    # number of time to repeat a full rotation
    rep_rotations = int(total_repeat/rotations)
    end_index = rep_rotations*rotations
    full_list[0:end_index] = partial_list*rep_rotations
    # remaining partial rotation
    partial_rotation = total_repeat % rotations
    full_list[end_index:end_index+partial_rotation] = partial_list[0:partial_rotation]
    return full_list

# returns input patterns and list of rotated images
# todo: cleanup
def cppn_patterns(population, repeat, structure, w, h, gpu, config, c_dim, gradient,
    output_dir, pertype_count, total_count, s_step):

    #rotate image by 90 degrees
    angle = 90
    rotations = int(360/angle)
    images_list = [None]*total_count

    index = 0
    image_inputs = create_grid(structure, w, h, 10)
    for genome_id, genome in population:
        # do not use genome_id as it increases with time
        j = 0
        image_whitebg = get_image_from_cppn(image_inputs, genome, c_dim, w, h, 10, config, bg = 1, gradient=gradient)
        # image_blackbg = get_image_from_cppn(image_inputs, genome, c_dim, w, h, 10, config, s_val = s_val, bg = 0)

        # save image
        pattern_folder = output_dir + "images/" + str(j).zfill(3)  + "/"
        # create folder
        if not os.path.exists(pattern_folder):
            os.makedirs(pattern_folder)


        image_name = pattern_folder + str(0).zfill(3) + ".png"
        image_whitebg.save(image_name, "PNG")
        image = np.asarray(Image.open(image_name))
        # save rotated images and get file names
        rotated_images_list = full_rotation(image_whitebg, angle, pattern_folder, 0)

        # repeat rotated names as necessary
        repeated_list = get_repeat_rotation_list(rotated_images_list, repeat)
        images_list[index:index+repeat] = repeated_list

        index += repeat
        j = j+1

    return images_list, image_inputs


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
    images_list, image_inputs = cppn_patterns(population, repeat,  structure, w, h, gpu, config, c_dim, gradient,
        output_dir, pertype_count, total_count, s_step)

    # calculate fitnesses using Prednet
    # 1 get absolute color differences
    # 2 get divergence from "black/grey/white" ie inter-rgb-difference magnitude
    
    # using mean
    #original_vectors = get_flows_mean(images_list, size, output_dir, c_dim)

#! todo  
    # calculate how much the colors diverge
    #scores = calculate_scores(len(population), structure, original_vectors, s_step)

    i = 0
    best_score = 0
    best_illusion = 0
    best_genome = None
    for genome_id, genome in population:
        genome.fitness = 0.01 #scores[i][1]
        # if (scores[i][1]> best_score):
        #     best_illusion = i
        #     best_score = scores[i][1]
        #     best_genome = genome
        # i = i+1

    # # save best illusion
    # image_name = output_dir + "/images/" + str(best_illusion).zfill(10) + ".png"
    # print("best", image_name, best_illusion)
    # move_to_name = best_dir + "/best.png"
    # shutil.copy(image_name, move_to_name)
    # index = int(best_illusion*(repeat/skip) + repeat-1)
    # image_name = output_dir + "/images/" + str(best_illusion).zfill(10) + "_f.png"
    # move_to_name = best_dir + "/best_flow.png"
    # shutil.copy(image_name, move_to_name)

    # image_blackbg = get_image_from_cppn(image_inputs, best_genome, c_dim, w, h, 10, config,
    #     s_val = -1, bg = 0, gradient=gradient)
    # image_name = best_dir + "/best_black_bg.png"
    # image_blackbg.save(image_name, "PNG")
    


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

