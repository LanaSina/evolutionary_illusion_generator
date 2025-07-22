import argparse
from chainer_prednet.PredNet.call_prednet import test_prednet
from chainer_prednet.utilities.mirror_images import mirror, mirror_multiple, TransformationType
import csv
import cv2
from enum import IntEnum
from google.colab.patches import cv2_imshow
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
from random import random, randrange
import shutil
import torch
from fitness_calculator import *


# TODO enumerate illusion types
class StructureType(IntEnum):
    Bands = 0
    Circles = 1
    Free = 2
    CirclesFree = 3
    Circles5Colors = 4


# todo: use in get_grid
# fill carthesian grids with polar coordinates
# r_len = repetition length
# xx yy cartesian x and y, origin relative to whole grid
# x,y coordinates relative to center
# direction: 1 or -1
def fill_circle(x, y, xx, yy, max_radius, direction, structure=StructureType.Circles):  # max diameter?
    r_total = np.sqrt(x * x + y * y)

    n_ratios = 10
    r_ratios = np.zeros(n_ratios)
    r_ratios[n_ratios - 1] = 1

    for i in range(2, n_ratios + 1):
        r_ratios[n_ratios - i] = r_ratios[n_ratios - i + 1] * 1.5

    r_ratios = r_ratios / r_ratios[0]

    # limit values to frame
    theta = 0
    r = -1
    if r_total <= max_radius / 2:
        # it repeats every r_len
        radius = min(1, r_total / (max_radius / 2))

        radius_index = 0
        for i in range(1, n_ratios - 1):
            if radius > r_ratios[i]:
                r = (radius - r_ratios[i]) / (r_ratios[i - 1] - r_ratios[i])
                if direction < 0:
                    r = 1-r
                radius_index = n_ratios - i - 1
                break;

        if structure == StructureType.Circles:
            # now structure theta values
            if x == 0:
                theta = math.pi / 2.0
            else:
                theta = np.arctan(y * 1.0 / x)

            if x < 0:
                theta = theta + math.pi

            r_index = radius_index
            if r_index % 2 == 1:
                # rotate
                theta = (theta + math.pi / 4.0)

                # focus on 1 small pattern
            theta = theta % (math.pi / 6.0)

            if direction < 0:
                theta = (math.pi / 6.0) - theta
 

        elif structure == StructureType.CirclesFree:

            # now structure theta values
            if x == 0:
                theta = math.pi / 2.0
            else:
                theta = np.arctan(y * 1.0 / x)

            if x < 0:
                theta = theta + math.pi

            r_index = radius_index
            if r_index % 2 == 1:
                # rotate
                theta = (theta + math.pi / 4.0)

            if direction < 0:
                # theta = - theta
                theta = (math.pi / 6.0) - theta


        # keep some white space
        if (r > 0.9) or (r < 0.1):
            r = -1
            theta = 0
        else:
            # final normalization
            r = r / 0.8

    return r, theta


# creates big image with several circles
def enhanced_image_grid(x_res, y_res, structure):
    x_mat = None
    y_mat = None
    scaling = 10

    num_points = x_res * y_res
    # coordinates of circle centers
    # 1: one row of circles at each third of the image
    c_rows = 3
    # 4 circles per row
    c_cols = 3
    y_step = (int)(y_res / c_cols)
    x_step = (int)(x_res / c_cols)

    # overlaid cicrles: 2 rows of 3 circles
    sub_rows = c_rows - 1
    sub_cols = c_cols - 1
    # coordinates
    centers = [None] * (c_rows * c_cols + sub_rows * sub_cols)

    for y in range(c_rows):
        for x in range(c_cols):
            index = y * c_cols + x
            centers[index] = [x_step * x + x_step / 2, y_step * y + y_step / 2]

    for y in range(sub_rows):
        for x in range(sub_cols):
            index = c_rows * c_cols + y * sub_cols + x
            centers[index] = [x_step * x + x_step, y_step * y + x_step]


    y_mat = np.ones((y_res, x_res))*-1
    x_mat = np.ones((y_res, x_res))*-1


    for row in range(c_rows):
        for col in range(c_cols):
            index = row * c_cols + col
            direction = 1
            if index % 2 == 0:
                direction = -1
            for xx in range(x_step):
                # shift coordinate to center of circle
                real_x = (col * x_step + xx)
                x = real_x - centers[index][0]
                for yy in range(y_step):
                    real_y = (row * y_step + yy)
                    y = real_y - centers[index][1]
                    r, theta = fill_circle(x, y, real_x, real_y, y_step, direction, structure)
                    x_mat[real_y, real_x] = r
                    y_mat[real_y, real_x] = theta

                    # secondary layer of circles
    for row in range(sub_rows):
        for col in range(sub_cols):
            index = c_rows * c_cols + row * sub_rows + col
            direction = 1
            if index % 2 == 0:
                direction = -1
            for xx in range(x_step):
                # shift coordinate to center 
                real_x = (col * x_step + xx) + (int)(x_step / 2)
                x = real_x - centers[index][0]
                for yy in range(y_step):
                    real_y = (row * y_step + yy) + (int)(y_step / 2)
                    y = real_y - centers[index][1]
                    r_total = np.sqrt(x * x + y * y)
                    if r_total < x_step / 2:
                        r, theta = fill_circle(x, y, real_x, real_y, y_step, direction, structure)
                        x_mat[real_y, real_x] = r
                        y_mat[real_y, real_x] = theta

    return {"x_mat": x_mat, "y_mat": y_mat}


def create_grid(structure, x_res=32, y_res=32, scaling=1.0):
    r_mat = None
    x_mat = None
    y_mat = None
    num_points = x_res * y_res

    if structure == StructureType.Bands:
        y_rep = 4
        padding = 10
        total_padding = padding * (y_rep - 1)
        y_len = int(y_res / y_rep)
        sc = scaling / y_rep
        a = np.linspace(-1 * sc, sc, num=y_len - padding)
        to_tile = np.concatenate((a, np.zeros((padding))))
        y_range = np.tile(to_tile, y_rep)

        x_rep = 10
        x_len = int(x_res / x_rep)
        sc = scaling / x_rep
        a = np.linspace(-1 * sc, sc, num=x_len)
        x_range = np.tile(a, x_rep)
        # reverse the x axis 
        # todo: ,1 not needed
        x_reverse = np.ones((y_res, 1))
        start = y_len
        while start < y_res:
            # keep some white space
            # top of previous band
            m_start = max(0, start - padding)
            x_reverse[m_start:start] = np.zeros((start - m_start, 1))

            # bottom of current band
            stop = min(y_res, start + y_len)
            m_start = max(stop - padding, 0)  # max(0,start-padding)
            x_reverse[m_start:stop] = np.zeros((stop - m_start, 1))
            x_reverse[start:stop] = -x_reverse[start:stop]
            start = start + 2 * y_len

        x_mat = np.matmul(x_reverse, x_range.reshape((1, x_res)))
        y_mat = np.matmul(y_range.reshape((y_res, 1)), np.ones((1, x_res)))
        x_mat = np.tile(x_mat.flatten(), 1).reshape(1, num_points, 1)
        y_mat = np.tile(y_mat.flatten(), 1).reshape(1, num_points, 1)

        return {"x_mat": x_mat, "y_mat": y_mat}

    elif structure == StructureType.Circles or structure == StructureType.Circles5Colors:
        r_ratios = [0.6, 0.3, 0.1]
        x_range = np.linspace(-1 * scaling, scaling, num=x_res)
        y_range = np.linspace(-1 * scaling, scaling, num=y_res)

        y_mat = np.matmul(y_range.reshape((y_res, 1)), np.ones((1, x_res)))
        x_mat = np.matmul(np.ones((y_res, 1)), x_range.reshape((1, x_res)))
        # x = r × cos( θ )
        # y = r × sin( θ )
        for xx in range(x_res):
            # center
            x = xx - (x_res / 2)
            for yy in range(y_res):
                y = yy - (y_res / 2)

                r, theta = fill_circle(x, y, xx, yy, y_res, 1)

                x_mat[yy, xx] = r
                y_mat[yy, xx] = theta
        return {"x_mat": x_mat, "y_mat": y_mat}

    elif structure == StructureType.CirclesFree:
        r_rep = 3
        r_len = int(y_res / (2 * r_rep))
        x_range = np.linspace(-1 * scaling, scaling, num=x_res)
        y_range = np.linspace(-1 * scaling, scaling, num=y_res)

        y_mat = np.matmul(y_range.reshape((y_res, 1)), np.ones((1, x_res)))
        x_mat = np.matmul(np.ones((y_res, 1)), x_range.reshape((1, x_res)))

        # x = r × cos( θ )
        # y = r × sin( θ )
        for xx in range(x_res):
            # center
            x = xx - (x_res / 2)
            for yy in range(y_res):
                y = yy - (y_res / 2)
                r_total = np.sqrt(x * x + y * y)

                # limit values to frame
                r = min(r_total, y_res / 2)
                # it repeats every r_len
                r = r % r_len
                # normalize
                r = r / r_len

                # now structure theta values
                theta = 0
                if r_total < y_res / 2:
                    if x == 0:
                        theta = math.pi / 2.0
                    else:
                        theta = np.arctan(y * 1.0 / x)

                    if x < 0:
                        theta = theta + math.pi

                    r_index = int(r_total / r_len)
                    if r_index % 2 == 1:
                        # rotate
                        theta = (theta + math.pi / 4.0)

                x_mat[yy, xx] = r
                y_mat[yy, xx] = theta

        return {"x_mat": x_mat, "y_mat": y_mat}

    elif structure == StructureType.Free:
        x_range = np.linspace(-1 * scaling, scaling, num=x_res)
        y_range = np.linspace(-1 * scaling, scaling, num=y_res)

        y_mat = np.matmul(y_range.reshape((y_res, 1)), np.ones((1, x_res)))
        x_mat = np.matmul(np.ones((y_res, 1)), x_range.reshape((1, x_res)))

        return {"x_mat": x_mat, "y_mat": y_mat}

    return {"input_0": x_mat, "input_1": y_mat, "input_2": r_mat}  # , s_mat


def get_fidelity(input_image_path, prediction_image_path):
    input_image = np.array(Image.open(input_image_path).convert('RGB'))
    prediction = np.array(Image.open(prediction_image_path).convert('RGB'))

    err = np.sum((input_image.astype("float") - prediction.astype("float")) ** 2)
    err /= (float(input_image.shape[0] * input_image.shape[1]) * 255 * 255)

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return 1 - err


import colorsys
def get_equilum_image_from_cppn(inputs, genome, c_dim, w, h, config, bg=1, gradient=1):
    out_names = []  # ["r0","g0","b0","r1","g1","b1"]
    leaf_names = ["x", "y"]
    x_dat = inputs["x_mat"]
    y_dat = inputs["y_mat"]
    inp_x = torch.tensor(x_dat.flatten())
    inp_y = torch.tensor(y_dat.flatten())

    image_array = np.zeros(((h, w, c_dim)))
    c = 0
    net_nodes = create_cppn(
        genome,
        config,
        leaf_names,
        out_names
    )

    # 3 nodes, one for each of h,s,v
    for node_func in net_nodes:
        # an array with values between 0 and 1
        pixels = node_func(x=inp_x, y=inp_y)
        pixels_np = pixels.numpy()
        image_array[:, :, c] = np.reshape(pixels_np, (h, w))
        for x in range(h):
            for y in range(w):
                if x_dat[x][y] == -1:
                    image_array[x, y, c] = bg  # white or black
        c = c + 1

    image_array = colorsys.hsv_to_rgb(image_array)
    img_data = np.array(image_array, dtype=np.uint8)

    image = Image.fromarray(img_data)

    return image


# bg = background, 1 for white 0 for black
# returns PIL image
def get_image_from_cppn(inputs, genome, c_dim, w, h, config, bg=1, gradient=1):
    out_names = []  # ["r0","g0","b0","r1","g1","b1"]
    leaf_names = ["x", "y"]
    x_dat = inputs["x_mat"]
    y_dat = inputs["y_mat"]
    inp_x = torch.tensor(x_dat.flatten())
    inp_y = torch.tensor(y_dat.flatten())

    # color images
    if c_dim > 1:
        image_array = np.zeros(((h, w, c_dim)))
        c = 0
        net_nodes = create_cppn(
            genome,
            config,
            leaf_names,
            out_names
        )

        if gradient == 1:
            # 3 nodes, one for each of r,g,b
            for node_func in net_nodes:
                # an array with values between 0 and 1
                pixels = node_func(x=inp_x, y=inp_y)
                pixels_np = pixels.numpy()
                image_array[:, :, c] = np.reshape(pixels_np, (h, w))
                for x in range(h):
                    for y in range(w):
                        if x_dat[x][y] == -1:
                            image_array[x, y, c] = bg  # white or black
                c = c + 1
            img_data = np.array(image_array * 255.0, dtype=np.uint8)
        else:
            node_func = net_nodes[0]
            pixels = node_func(x=inp_x, y=inp_y)
            pixels_np = pixels.numpy()
            image_array = np.reshape(pixels_np, (h, w))

            # 0 to 4
            # black, white, r, g, or b=255
            color_data = np.array(image_array * 4.0, dtype=np.uint8)
            color_data = np.round(color_data)
            img_data = np.zeros((h, w, 3))
            # fill each channel
            # white
            img_data[:, :, 0] = np.where(color_data == 0, 255, img_data[:, :, 0])
            img_data[:, :, 1] = np.where(color_data == 0, 255, img_data[:, :, 1])
            img_data[:, :, 2] = np.where(color_data == 0, 255, img_data[:, :, 2])
            # rgb
            img_data[:, :, 0] = np.where(color_data == 1, 255, img_data[:, :, 0])
            img_data[:, :, 1] = np.where(color_data == 2, 255, img_data[:, :, 1])
            img_data[:, :, 2] = np.where(color_data == 3, 255, img_data[:, :, 2])

            # fill background
            for x in range(h):
                for y in range(w):
                    if x_dat[x][y] == -1:
                        img_data[x, y] = [bg*255, bg*255, bg*255]  # white or black

            img_data = np.array(img_data, dtype=np.uint8)

        image = Image.fromarray(img_data)
    # grayscale
    else:
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

        image_array = pixels_np
        for x in range(h):
            for y in range(w):
                if x_dat[x][y] == -1:
                    image_array[x, y] = bg

        # for no gradients
        if gradient == 0:
            image_array = np.round(image_array)

        img_data = np.array(image_array * 255.0, dtype=np.uint8)
        image = Image.fromarray(img_data, 'L')

    return image


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def pil_to_cv2(image, c_dim):
    image_np = np.asarray(image)
    if c_dim == 3:
        open_cv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        open_cv_image = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

    return open_cv_image


# population:  [id, net]
def get_fitnesses_neat(structure, population, model_name, config, w, h, channels,
                       id=0, c_dim=3, best_dir=".", gradient=1):
    print("Calculating fitnesses of populations: ", len(population))
    output_dir = "temp/"
    repeat = 20
    half_h = int(h / 2)
    size = [w, h]
    gpu = 0

    prediction_dir = output_dir + "/prediction/"
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    if not os.path.exists(output_dir + "images/"):
        os.makedirs(output_dir + "images/")

    # latent space coarse graining (none)
    s_step = 2
    pertype_count = int((2 / s_step))
    total_count = len(population) * pertype_count
    images_list = [None] * total_count
    repeated_images_list = [None] * (total_count + repeat)
    i = 0
    image_inputs = create_grid(structure, w, h, 10)
    for genome_id, genome in population:
        # traverse latent space
        j = 0
        for s in range(0, pertype_count):
            s_val = -1 + s * s_step
            index = i * pertype_count + j


            # equiluminance
            #image_whitebg = get_equilum_image_from_cppn(image_inputs, genome, c_dim, w, h, config, gradient=gradient) #  get_image_from_cppn
            # image_blackbg = ..., bg = 0)

            image_whitebg = get_image_from_cppn(image_inputs, genome, c_dim, w, h, config, gradient=gradient) #  get_image_from_cppn


            # save  image
            image_name = output_dir + "images/" + str(index).zfill(10) + ".png"
            image_whitebg.save(image_name, "PNG")
            # image_name = output_dir + "images/" + str(index).zfill(10) + "_black.png"
            # image_blackbg.save(image_name, "PNG")

            images_list[index] = image_name
            repeated_images_list[index * repeat:(index + 1) * repeat] = [image_name] * repeat

            j = j + 1
        i = i + 1

    print("Predicting illusions...")
    skip = 1
    extension_duration = 2  # 2
    # runs repeat x times on the input image, save in result folder
    test_prednet(initmodel=model_name, sequence_list=[repeated_images_list], size=size,
                 channels=channels, gpu=gpu, output_dir=prediction_dir, skip_save_frames=skip,
                 extension_start=repeat, extension_duration=extension_duration,
                 reset_at=repeat + extension_duration, verbose=0, c_dim=c_dim
                 )
    # calculate flows
    print("Calculating flows...")
    i = 0
    original_vectors = [None] * total_count
    for input_image in images_list:
        index_0 = int(i * (repeat / skip) + repeat - 1)
        index_1 = index_0 + extension_duration - 1
        prediction_0 = prediction_dir + str(index_0).zfill(10) + ".png"
        prediction_1 = prediction_dir + str(index_1).zfill(10) + "_extended.png"

        save_name = output_dir + "/images/" + str(i).zfill(10) + "_f.png"
        results = lucas_kanade(prediction_0, prediction_1, output_dir + "/flow/", save=True, verbose=0,
                               save_name=save_name)
        if results["vectors"]:
            original_vectors[i] = np.asarray(results["vectors"])
        else:
            original_vectors[i] = [[0, 0, -1000, 0]]
        i = i + 1

    # calculate score
    scores = [None] * len(population)
    for i in range(0, len(population)):
        final_score = -100
        # traverse latent space
        # is this a mean score per family?
        for j in range(0, int(2 / s_step)): # this is currenly 1
            index = i * pertype_count + j
            score = 0
            score_d = 0

            if structure == StructureType.Bands:
                ratio = plausibility_ratio(original_vectors[index], 0.15)
                score_0 = ratio[0]
                good_vectors = ratio[1]

                if (len(good_vectors) > 0):
                    stripes = 4
                    step = h / stripes
                    score_direction = horizontal_symmetry_score(good_vectors, [0, step * 2])

                    # bonus for strength
                    score_d = score_direction  # *min(1,score_strength)

            elif structure == StructureType.Circles \
                    or structure == StructureType.CirclesFree \
                    or structure == StructureType.Circles5Colors:
                max_strength = 0.3  # 0.4
                ratio = plausibility_ratio(original_vectors[index], max_strength)
                score_0 = ratio[0]
                good_vectors = ratio[1]
                min_vectors = ((2 * math.pi) / (math.pi / 4.0)) * 3

                if (len(good_vectors) > min_vectors):
                    # get tangent scores
                    limits = [0, h / 2]
                    score_direction = rotation_symmetry_score(good_vectors, w, h, limits, images_list[index])
                    score_strength = strength_number(good_vectors, max_strength)
                    score_d = 0.7 * score_direction + 0.3 * score_strength

            elif structure == StructureType.Free:
                max_strength = 0.4
                ratio = plausibility_ratio(original_vectors[index], max_strength)
                good_vectors = ratio[1]

                if (len(good_vectors) > 0):
                    score_strength = strength_number(good_vectors, max_strength)
                    score_number = min(len(good_vectors), 15) / 15
                    score_s = swarm_score(good_vectors)
                    score_d = 0.5 * score_s + 0.1 * score_strength + 0.4 * score_number
            else:
                score_d = inside_outside_score(good_vectors, w, h)

            score = score + score_d

            if score > final_score:
                final_score = score
                temp_index = index

        m = score / pertype_count
        scores[i] = [i, m]

    print("scores", scores)
    i = 0
    best_score = 0
    best_illusion = 0
    best_genome = None
    for genome_id, genome in population:
        genome.fitness = scores[i][1]
        if (scores[i][1] >= best_score):
            best_illusion = i
            best_score = scores[i][1]
            best_genome = genome
        # if (scores[i][1]==0):
        #     # save a control
        #     print("0 fitness", i)
        #     image_name = output_dir + "/images/" + str(i).zfill(10) + ".png"
        #     move_to_name = best_dir + "/0-fitness.png"
        #     shutil.copy(image_name, move_to_name)
        #     index = int(i * (repeat / skip) + repeat - 1)
        #     image_name = output_dir + "/images/" + str(i).zfill(10) + "_f.png"
        #     move_to_name = best_dir + "/0-fitness_flow.png"
        #     shutil.copy(image_name, move_to_name)
        #     # create enhanced image
        #     e_w = 800
        #     e_h = 800
        #     e_grid = enhanced_image_grid(e_w, e_h, structure)
        #     image = get_image_from_cppn(e_grid, population[i][1], c_dim, e_w, e_h, config, bg=1, gradient=gradient)
        #     image_name = best_dir + "/0-fitness_enhanced.png"
        #     image.save(image_name)

        i = i + 1

    # save best illusion
    print("best", best_score, image_name, best_illusion)
    image_name = output_dir + "/images/" + str(best_illusion).zfill(10) + ".png"
    move_to_name = best_dir + "/best.png"
    shutil.copy(image_name, move_to_name)
    index = int(best_illusion * (repeat / skip) + repeat - 1)
    image_name = output_dir + "/images/" + str(best_illusion).zfill(10) + "_f.png"
    move_to_name = best_dir + "/best_flow.png"
    shutil.copy(image_name, move_to_name)
    # show in colab
    cv2_imshow(cv2.imread(move_to_name))

    image_blackbg = get_image_from_cppn(image_inputs, best_genome, c_dim, w, h, config, bg=0, gradient=gradient)
    image_name = best_dir + "/best_black_bg.png"
    image_blackbg.save(image_name, "PNG")

    # create enhanced image
    e_w = 800
    e_h = 800
    e_grid = enhanced_image_grid(e_w, e_h, structure)
    image = get_image_from_cppn(e_grid, population[best_illusion][1], c_dim, e_w, e_h, config, bg=1, gradient=gradient)
    image_name = best_dir + "/enhanced.png"
    image.save(image_name)
    # show in colab
    cv2_imshow(cv2.imread(image_name))


def neat_illusion(output_dir, model_name, config_path, structure, w, h, channels, c_dim=3, checkpoint=None, gradient=1):
    repeat = 6
    limit = 1
    half_h = int(h / 2)
    size = [w, h]
    gpu = 0

    best_dir = output_dir
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    def eval_genomes(genomes, config):
        get_fitnesses_neat(structure, genomes, model_name, config, w, h, channels,
                           c_dim=c_dim, best_dir=best_dir, gradient=gradient)

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
    winner = p.run(eval_genomes, 100)


def string_to_intarray(string_input):
    array = string_input.split(',')
    for i in range(len(array)):
        array[i] = int(array[i])

    return array


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate illusions')
    parser.add_argument('--model', '-m', default='', help='.model file')
    parser.add_argument('--output_dir', '-o', default='.', help='path of output diectory')
    parser.add_argument('--structure', '-s', default=0, type=int,
                        help='Type of illusion. 0: Bands; 1: Circles; 2: Free form')
    parser.add_argument('--config', '-cfg', default="", help='path to the NEAT config file')
    parser.add_argument('--checkpoint', '-cp', help='path of checkpoint to restore')
    parser.add_argument('--size', '-wh', help='big or small', default="small")
    parser.add_argument('--color_space', '-c', help='1 for greyscale, 3 for rgb', default=3, type=int)
    # [1,16,32,64]
    # 3,48,96,192
    parser.add_argument('--channels', '-ch', default='3,48,96,192', help='Number of channels on each layers')
    parser.add_argument('--gradient', '-g', default=1, type=int, help='1 to use gradients, 0 for pure colors')

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
            if args.color_space > 1:
                if args.gradient == 1:
                    config += "/neat_configs/circles.txt"
                else:
                    config += "/neat_configs/circles_bw.txt"
            else:
                config += "/neat_configs/circles_bw.txt"
        elif args.structure == StructureType.Free:
            config += "/neat_configs/free.txt"
        else:
            config += "/neat_configs/default.txt"

    print("config", config)
    print("gradient", args.gradient)
    neat_illusion(output_dir, args.model, config, args.structure, w, h, string_to_intarray(args.channels),
                  args.color_space, args.checkpoint, args.gradient)
