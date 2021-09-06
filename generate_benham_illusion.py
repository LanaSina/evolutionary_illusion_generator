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


def create_grid(structure, x_res = 32, y_res = 32, scaling = 1.0):

    r_mat = None 
    x_mat = None
    y_mat = None
    num_points = x_res*y_res
    max_radius = (h/2) - 5
   
    # just some nice circles
    x_range = np.linspace(-1*scaling, scaling, num = x_res)
    y_range = np.linspace(-1*scaling, scaling, num = y_res)


    y_mat = np.matmul(y_range.reshape((y_res, 1)), np.ones((1, x_res)))
    x_mat = np.matmul(np.ones((y_res, 1)), x_range.reshape((1, x_res)))

    for xx in range(x_res):
        # center
        x = xx - (x_res/2)
        for yy in range(y_res):
            y = yy - (y_res/2)
            r_total = np.sqrt(x*x + y*y)
            
            # limit values to frame
            r = min(r_total, y_res/2)
            # normalize
            r = r/max_radius

            # now structure theta values
            theta = 0
            if r_total < y_res/2:
                if x == 0:
                    theta = math.pi/2.0
                else:
                    theta = np.arctan(y*1.0/x)

                if x<0:
                    theta = theta + math.pi

             # keep some white space
            if (r>0.9) or (r<0.1):
                r = -1
                theta = 0
            else :
                #final normalization
                r = r/0.8

            x_mat[yy,xx] = r 
            y_mat[yy,xx] = theta 

    return {"x_mat": x_mat, "y_mat": y_mat}


# bg = background, 1 for white 0 for black
def get_image_from_cppn(inputs, genome, c_dim, w, h, scaling, config, s_val = 1, bg = 1, gradient = 1):
   
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

    return image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])



# rotates an image by `angle` until it reaches 360 degrees
# save the rotations and return the list of file names
# index is the file numer name for saving
# angle = clockwise rotation
def full_rotation(image, angle, output_dir, index=0, c_dim=1):
    rotations = int(360/angle)
    rotated_images_list = [None]*rotations

    image_name = output_dir + "/" + str(index).zfill(3) + ".png"
    rotated_images_list[0] = image_name 
    white_color = (255,255,255)
    if c_dim == 1:
        white_color = 255

    for i in range(1,rotations):
        rotated_image = image.rotate(-angle*i, expand=False, fillcolor=white_color)
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
    # todo
    # todo: the  images are  off-center
    # todo: radial patterns
    image_inputs = create_grid(structure, w, h, 10) # x and y inverted
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
        rotated_images_list = full_rotation(image_whitebg, angle, pattern_folder, 0, c_dim)

        # repeat rotated names as necessary
        repeated_list = get_repeat_rotation_list(rotated_images_list, repeat)
        images_list[index:index+repeat] = repeated_list

        index += repeat
        j = j+1

    return images_list, image_inputs


def get_mean_radius_value(image_array):
    #probably make masks and average that


def radius_color_difference(images_list, model_name, repeated_images_list, size, channels, gpu, output_dir,
    repeat, c_dim, total_count):

    prediction_dir = output_dir + "/prediction/"
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    print("Predicting illusions...")
    # what?
    skip = 1
    # how many frames to predict
    extension_duration = 2 #2
    # runs repeat x times on the input image, save in result folder
    test_prednet(initmodel = model_name, sequence_list = [repeated_images_list], size=size, 
                channels = channels, gpu = gpu, output_dir = prediction_dir, skip_save_frames=skip,
                extension_start = repeat, extension_duration = extension_duration,
                reset_at = repeat+extension_duration, verbose = 0, c_dim = c_dim
                )

    # calculate color differences
    print("Calculating color differences...")
    
    # images to compare
    # for input_image in images_list:
    #     # todo: this is not the same anlge! 
    #     original_image = np.asarray(Image.open(input_image)) warning see up
    #     index_0 = int(i*(repeat/skip)+ repeat-1)
    #     index_1 = index_0+extension_duration-1
    #     predicted_image_path = prediction_dir + str(index_1).zfill(10) + "_extended.png"
    #     predicted_image = np.asarray(Image.open(predicted_image_path))

    #     # compare by radius
    #     diff = get_mean_radius_value(original_image-predicted_image)


    #     index_0 = int(i*(repeat/skip)+ repeat-1)
    #     index_1 = index_0+extension_duration-1
    #     prediction_0 = prediction_dir + str(index_0).zfill(10) + ".png"
    #     prediction_1 = prediction_dir + str(index_1).zfill(10) + "_extended.png"

    #     save_name = output_dir + "/images/" + str(i).zfill(10) + "_f.png"




    # original_vectors = [None] * total_count
    # for input_image in images_list:
    #     index_0 = int(i*(repeat/skip)+ repeat-1)
    #     index_1 = index_0+extension_duration-1
    #     prediction_0 = prediction_dir + str(index_0).zfill(10) + ".png"
    #     prediction_1 = prediction_dir + str(index_1).zfill(10) + "_extended.png"

    #     save_name = output_dir + "/images/" + str(i).zfill(10) + "_f.png"
    #     results = lucas_kanade(prediction_0, prediction_1, output_dir+"/flow/", save=True, verbose = 0, save_name = save_name)
    #     if results["vectors"]:
    #         original_vectors[i] = np.asarray(results["vectors"])
    #     else:
    #         original_vectors[i] = [[0,0,-1000,0]]
    #     i = i + 1

    # return original_vectors


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
    # 1 get absolute color differences per radius
    radius_color_difference(images_list, size, output_dir, c_dim)
    color_diff = radius_color_difference(images_list, model_name, repeated_images_list, size, channels, gpu, output_dir,
        repeat, c_dim, total_count)

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
    

def mutate_genome(genome):
    n = max(0, genome["number_mutation"] + round(random()))
    r =  max(0, genome["range_mutation"] + round(random()))
    rad =  max(0, genome["radius"] + round(random()))

    mutated_genome = {"number_mutation": n, "range_mutation": r, "radius": rad}
    return mutated_genome


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

