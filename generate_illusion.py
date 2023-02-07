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


# TODO enumerate illusion types
class StructureType(IntEnum):
    Bands = 0
    Circles = 1
    Free = 2
    CirclesFree = 3
    # Circles5Colors = 4


# returns ratio and vectors that are not unplausibly big
def plausibility_ratio(vectors, limit):
    r = []
    for vector in vectors:
        norm = np.sqrt(vector[2] * vector[2] + vector[3] * vector[3])
        if norm > limit:
            continue
        r.append(vector)

    ratio = len(r) / len(vectors)
    return [ratio, r]


# returns mean of vectors norms weighted by their variances
# low variance = good
def strength_number(vectors, max_norm):
    v = np.asarray(vectors)
    mx = np.mean(abs(v[:, 2]))
    my = np.mean(abs(v[:, 3]))

    norms = np.sqrt(v[:, 2] * v[:, 2] + v[:, 3] * v[:, 3])
    v = np.var(norms)
    score = mx / max_norm  # could be 1
    score = score * (1 - min(v, 1))
    return score


# returns [a,b]
# a = 1 if vectors rather aligned on x to the right;  -1 if to the left
# b = mean of projection on x axis (normalised)
def direction_ratio(vectors, limits=None):
    # print(vectors)
    mean_ratio = 0
    count = 0
    orientation = 0

    for v in vectors:
        # skip vectors that are outside the limits
        if not limits is None:
            if (v[1] < limits[0]) or (v[1] > limits[1]):
                continue

        # calculate x axis ratio
        # x length divided by norm
        norm_v = np.sqrt(v[2] * v[2] + v[3] * v[3])
        ratio = v[2] / norm_v
        mean_ratio = mean_ratio + ratio
        orientation = orientation + v[2]
        count = count + 1

    if count > 0:
        mean_ratio = mean_ratio / count
    else:
        mean_ratio = 0

    if orientation > 0:
        orientation = 1
    elif orientation < 0:
        orientation = -1

    return [orientation, mean_ratio]


# calcuates the symmetry on the middle axis
def horizontal_symmetry_score(vectors, limits=[0, 60]):
    # print(vectors)
    mean_ratio = 0
    count = 0
    orientation = 0
    middle = int(limits[1] / 2)

    # matrix of mirrored vectors
    mirrored_vectors = np.zeros((len(vectors), 2))

    count = 0
    for v in vectors:
        # skip vectors that are outside the limits
        if (v[1] < limits[0]) or (v[1] > limits[1]):
            continue

        # normalize the vectors to offset model biases
        normalized_v = v / np.sqrt(v[2] * v[2] + v[3] * v[3])

        if (v[1] < middle):
            mirrored_vectors[count] = normalized_v[2:3]
        else:
            mirrored_vectors[count] = [-normalized_v[2], normalized_v[3]]

        count = count + 1

    if count == 0:
        return 0

    # remove everything beyond count
    mirrored_vectors = mirrored_vectors[:count, :]

    var_x = np.var(mirrored_vectors[:, 0])
    mean_x = abs(np.mean(mirrored_vectors[:, 0]))
    mean_y = abs(np.mean(mirrored_vectors[:, 1]))

    # max var is 1
    score = ((1 - var_x) + mean_x + (1 - mean_y)) / 3
    # print("score", score)
    return score


# returns the agreement and disagreement betwen vectors
def swarm_score(vectors):
    max_distance = 100  # px
    distance_2 = 50
    score = 0
    n = len(vectors)

    # normalize vectors
    norm_vectors = np.array(vectors)
    # print("vector array", norm_vectors)
    norms = np.sqrt(norm_vectors[:, 2] * norm_vectors[:, 2] + norm_vectors[:, 3] * norm_vectors[:, 3])
    norm_vectors[:, 2] = norm_vectors[:, 2] / norms
    norm_vectors[:, 3] = norm_vectors[:, 3] / norms
    temp = np.sqrt(norm_vectors[:, 2] * norm_vectors[:, 2] + norm_vectors[:, 3] * norm_vectors[:, 3])
    angles = np.arccos(norm_vectors[:, 2])

    for v_a in norm_vectors:
        # distance used as factor
        x = norm_vectors[:, 0] - v_a[0]
        y = norm_vectors[:, 1] - v_a[1]
        # [0 .. 1]
        distances = (np.multiply(x, x) + np.multiply(y, y))
        distance_factors = distances / (max_distance * max_distance)
        distance_factors = np.where(distance_factors > 1, 1, distance_factors)
        # 1 where vectors are close
        close = 1 - np.where(distance_factors < 1, 0, distance_factors)

        # vectors orientation
        # alpha = acos(x)
        v_angle = math.acos(v_a[2])
        # optimal deviation: completely opposite at 100 px away (distance factor  = 1)
        optimal = (v_angle + distance_factors * math.pi) % 2 * math.pi
        loss = close * abs(angles - optimal)
        temp = math.pi - (sum(loss) / n)
        score = score + (temp / math.pi)

    return score / n


# rotate all vectors to align their origin on x axis
# calculate the mean and variance of normalized vectors
# returns a high score if the variance is low (ie the vectors are symmetric)
# limits = radius limits
def rotation_symmetry_score(vectors, w, h, limits=None, original_filename="temp.png"):
    # fill matrix of vectors
    rotated_vectors = np.zeros((len(vectors), 4))
    distances = np.zeros((len(vectors)))
    count = 0
    center = [w / 2, h / 2]
    for v in vectors:
        # change coordinates to center
        vc = [v[0] - center[0], v[1] - center[1]]
        distance = np.sqrt(vc[0] * vc[0] + vc[1] * vc[1])
        if not limits is None:
            if (distance < limits[0]) or (distance > limits[1]) or distance == 0:
                continue

        rotated_vectors[count] = [vc[0], vc[1], v[2], v[3]]
        distances[count] = distance
        count = count + 1

    if (count < 2):
        return 0

    # remove everything beyond count
    rotated_vectors = rotated_vectors[:count, :]
    distances = distances[:count]

    # normalise vectors
    norms = np.sqrt(rotated_vectors[:, 2] * rotated_vectors[:, 2] + rotated_vectors[:, 3] * rotated_vectors[:, 3])
    rotated_vectors[:, 2] = rotated_vectors[:, 2] / norms
    rotated_vectors[:, 3] = rotated_vectors[:, 3] / norms

    # rotate vectors clockwise to x axis
    # new_x = cos(a)x + sin(a)y, new_y = cos(a)y - sin(a)x
    # cos(a) = x/dist, sin a = y/dist
    # new_y = -sin(a)x + cos(a)y
    # vector origin is going to be [dist,0]
    # vector end coordinates
    x_1 = rotated_vectors[:, 0] + rotated_vectors[:, 2]
    y_1 = rotated_vectors[:, 1] + rotated_vectors[:, 3]

    rx_1 = (x_1 * rotated_vectors[:, 0] + y_1 * rotated_vectors[:, 1]) / distances
    ry_1 = (-x_1 * rotated_vectors[:, 1] + y_1 * rotated_vectors[:, 0]) / distances
    r_v = np.array([rx_1 - distances, ry_1]).transpose()

    var_x = np.var(r_v[:, 0])
    var_y = np.var(r_v[:, 1])

    # max var is 1
    score = (1 - var_x) * (1 - var_x) + (1 - var_y) * (1 - var_y)
    score = score / 2
    return score


# agreement inside the cell, + disagreement outside of it
def inside_outside_score(vectors, width, height):
    step = width / 5  # px
    # build an array of vectors 
    w = int(width / step) + 1
    h = int(height / step) + 1
    flow_array = np.zeros((w, h, 2))
    count_array = np.ones((w, h))
    agreement_array = np.zeros((w, h, 2))
    norm_sum_array = np.zeros((w, h))

    # take the mean for vectors in the same cell, and calculate agreement score
    # vectors orientation 
    for index in range(0, len(vectors)):
        v = vectors[index]
        i = int(v[0] / step)
        j = int(v[1] / step)

        flow_array[i, j, 0] += v[2]
        flow_array[i, j, 1] += v[3]
        count_array[i, j] += 1
        norm_v = np.sqrt(v[2] * v[2] + v[3] * v[3])
        norm_sum_array[i, j] += norm_v

    # not a real mean as the count started at 1
    flow_array[:, :, 0] = flow_array[:, :, 0] / count_array
    flow_array[:, :, 1] = flow_array[:, :, 1] / count_array
    norm_sum_array = norm_sum_array / count_array

    # now take the variance
    for index in range(0, len(vectors)):
        v = vectors[index]
        i = int(v[0] / step)
        j = int(v[1] / step)
        agreement_array[i, j, 0] += (flow_array[i, j, 0] - v[2]) * (flow_array[i, j, 0] - v[2])
        agreement_array[i, j, 1] += (flow_array[i, j, 1] - v[3]) * (flow_array[i, j, 1] - v[3])

    agreement_array[:, :, 0] = agreement_array[:, :, 0] / count_array
    agreement_array[:, :, 1] = agreement_array[:, :, 1] / count_array

    # take the sums
    score_agreement = - (min(np.mean(agreement_array), 10))
    score_size = min(10, np.mean(norm_sum_array))

    # compare with other cells
    sum_d = 0
    for i in range(0, w):
        for j in range(0, h):
            vx = flow_array[i, j, 0]
            vy = flow_array[i, j, 1]
            if (vx != 0 or vy != 0):
                # normalize
                norm_v = np.sqrt(vx * vx + vy * vy)
                vx = vx / norm_v
                vy = vy / norm_v

            min_i = max(0, i - 1)
            max_i = min(w, i + 1)
            min_j = max(0, j - 1)
            max_j = min(h, i + 1)
            plus = 0
            minus = 0
            for x in range(min_i, max_i):
                for y in range(min_j, max_j):
                    if i == x and j == y:
                        continue

                    wx = flow_array[x, y, 0]
                    wy = flow_array[x, y, 1]
                    if (wx != 0 or wy != 0):
                        norm_w = np.sqrt(wx * wx + wy * wy)
                        wx = wx / norm_w
                        wy = wy / norm_w
                        # +1 for disagreement
                        dot = vx * wx + vy * wy
                        if dot > 0:
                            plus += 1
                        else:
                            minus += 1
            sum_d += (min(2, plus) + min(2, minus)) / 4

    sum_d = sum_d / (w * h)
    sum_d = sum_d * 10

    final_score = score_agreement + score_size + sum_d
    final_score = final_score / 30
    return final_score


# calculate how parallel nearby patches are and how different they are from
# slightly further away patches
def divergence_convergence_score(vectors, width, height):
    step = height * 4 / len(vectors)

    score = 0
    step = 10  # px
    # build an array of vectors 
    w = int(width / step)
    h = int(height / step)
    flow_array = np.zeros((w, h, 2))

    # TODO: take the mean for vectors in the same cell
    # vectors orientation 
    for index in range(0, len(vectors)):
        v = vectors[index]
        i = int(v[0] / step)
        j = int(v[1] / step)
        norm_v = np.sqrt(v[2] * v[2] + v[3] * v[3])
        x = v[2] / norm_v
        y = v[3] / norm_v
        flow_array[i, j, 0] = x
        flow_array[i, j, 1] = y

    # calculate points
    for i in range(0, w):
        for j in range(0, h):
            xmin = max(i - 1, 0)
            xmax = min(i + 1, w)
            ymin = max(j - 1, 0)
            ymax = min(j + 1, h)
            loss = 0
            sum_vec = 0
            vx = flow_array[i, j, 0]
            vy = flow_array[i, j, 1]
            if vx == 0 and vy == 0:
                continue

            plus = 0
            minus = 0

            sum_norm = 0
            for x in range(xmin, xmax):
                for y in range(ymin, ymax):
                    if flow_array[x, y, 0] == 0 and flow_array[x, y, 1] == 0:
                        continue

                    sum_vec += 1

                    dot = vx * flow_array[x, y, 0] + vy * flow_array[x, y, 1]
                    # aim for either completely different or completely same
                    loss = (abs(dot) - 0.5) * (abs(dot) - 0.5)
                    if (dot > 0):
                        plus += dot
                    else:
                        minus -= dot

                    # loss += (dot-0.5)*(dot-0.5)
                    # sum_vec += 1

            if (sum_vec > 0):
                # there must be + and - in equal parts
                # print("plus, minus", plus, minus)
                loss = 1 - (plus - minus) / (plus + minus)
                # high norms are better
                loss = loss * abs(vx + vy)
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
def tangent_ratio(vectors, w, h, limits=None):
    # we want to know the angle between
    # a radius of the circle at the center of the image
    # and the motion vectors

    # center
    c = [w / 2.0, h / 2.0]

    # scores
    direction = 0
    mean_alignment = 0

    count = 0
    for v in vectors:
        # if(v[0]!=106): continue #39

        # oh boy
        # v 
        v[0] = v[0] - c[0]
        v[1] = v[1] - c[1]
        v[2] = v[0] + v[2]
        v[3] = v[1] + v[3]

        # radius vector R from origin of V to image center
        r = [0, 0, v[0], v[1]]
        # offsets: change origin to vector origin
        ro = [r[2] - r[0], r[3] - r[1]]
        vo = [v[2] - v[0], v[3] - v[1]]

        # check limits
        norm_r = np.sqrt(ro[0] * ro[0] + ro[1] * ro[1])
        norm_v = np.sqrt(vo[0] * vo[0] + vo[1] * vo[1])

        if (norm_r * norm_v == 0):
            count = count + 1
            continue

        # normalize 
        ro = ro / norm_r
        vo = vo / norm_v

        if not limits is None:
            if (norm_r < limits[0]) or (norm_r > limits[1]):
                continue

        # find angle between vectors by using dot product
        dot_p = ro[0] * vo[0] + ro[1] * vo[1]  # divide by (norm v * norm r) which is 1*1
        # sometimes slight errors

        if dot_p > 1:
            dot_p = 1
        elif dot_p < -1:
            dot_p = -1

        angle = math.acos(dot_p)
        # this angle is ideally pi/2 or -pi/2
        score = (math.pi / 2) - abs(angle)
        # and the max difference is pi/2
        score = 1 - (abs(score) / (math.pi / 2))

        # we'd like them to all have the same alignment
        # use cross product to find ccw or cv
        cw = ro[0] * vo[1] - ro[1] * vo[0]
        # maybe just add, if it's a flow fluke it will always be lower anyway
        # mean_alignment = mean_alignment + abs(score)
        if (cw > 0):
            mean_alignment = mean_alignment + score
        else:
            mean_alignment = mean_alignment - score
        count = count + 1

    if mean_alignment > 0:
        direction = 1
    elif mean_alignment < 0:
        direction = -1

    if count > 0:
        mean_alignment = mean_alignment / count

    return [direction, abs(mean_alignment)]


def get_vectors(image_path, model_name, w, h):
    skip = 1
    extension_duration = 2
    repeat = 20
    half_h = int(h / 2)
    size = [w, h]
    channels = [3, 48, 96, 192]
    gpu = 0

    output_dir = "test/"
    prediction_dir = output_dir + "/prediction/"
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    repeated_images_list = [image_path] * repeat
    # print("list", repeated_images_list)

    # runs repeat x times on the input image, save in result folder
    test_prednet(initmodel=model_name, sequence_list=[repeated_images_list], size=size,
                 channels=channels, gpu=gpu, output_dir=prediction_dir, skip_save_frames=skip,
                 extension_start=repeat, extension_duration=extension_duration,
                 reset_at=repeat + extension_duration, verbose=0
                 )

    extended = prediction_dir + str(repeat + 1).zfill(10) + "_extended.png"
    # calculate flows
    print("Calculating flows...", extended)
    vectors = [None]

    results = lucas_kanade(image_path, extended, prediction_dir, save=True, verbose=0, save_name="flow.png")
    if results["vectors"]:
        vectors = np.asarray(results["vectors"])

    return vectors


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
                theta = - theta

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


            image_whitebg = get_equilum_image_from_cppn(image_inputs, genome, c_dim, w, h, config, gradient=gradient) #  get_image_from_cppn
            # image_blackbg = ..., bg = 0)

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
        for j in range(0, int(2 / s_step)):
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
        if (scores[i][1] > best_score):
            best_illusion = i
            best_score = scores[i][1]
            best_genome = genome
        i = i + 1

    # save best illusion
    print("best", image_name, best_illusion)
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
