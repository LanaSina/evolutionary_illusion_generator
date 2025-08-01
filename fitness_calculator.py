import numpy as np
import os
from optical_flow.optical_flow import lucas_kanade, draw_tracks, save_data
from chainer_prednet.PredNet.call_prednet import test_prednet
from chainer_prednet.utilities.mirror_images import mirror, mirror_multiple, TransformationType
from enum import IntEnum
import math


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


def get_vectors(image_path, model_name, channels, w, h):
    skip = 1
    extension_duration = 2
    repeat = 20
    half_h = int(h / 2)
    size = [w, h]
    c_dim = channels[0]
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
                 reset_at=repeat + extension_duration, c_dim=c_dim, verbose=0
                 )

    extended = prediction_dir + str(repeat + 1).zfill(10) + "_extended.png"
    # calculate flows
    print("Calculating flows...", extended)
    vectors = [None]

    results = lucas_kanade(image_path, extended, prediction_dir, save=True, verbose=0, save_name="flow.png")
    if results["vectors"]:
        vectors = np.asarray(results["vectors"])

    return vectors


def calculate_fitness(structure, vectors, image_path, w, h):
   
    if structure == StructureType.Bands:
        ratio = plausibility_ratio(vectors, 0.15)
        score_0 = ratio[0]
        good_vectors = ratio[1]

        if (len(good_vectors) > 0):
            stripes = 4
            step = h / stripes
            score_direction = horizontal_symmetry_score(good_vectors, [0, step * 2])

            # bonus for strength
            score_d = score_direction  # *min(1,score_strength)

    elif structure == StructureType.Circles \
            or structure == StructureType.CirclesFree:
        max_strength = 0.3  # 0.4
        ratio = plausibility_ratio(vectors, max_strength)
        score_0 = ratio[0]
        good_vectors = ratio[1]
        min_vectors = 24 #((2 * math.pi) / (math.pi / 4.0)) * 3

        if (len(good_vectors) > min_vectors):
            # get tangent scores
            limits = [0, h / 2]
            score_direction = rotation_symmetry_score(good_vectors, w, h, limits, image_path)
            score_strength = strength_number(good_vectors, max_strength)
            score_d = 0.7 * score_direction + 0.3 * score_strength

    elif structure == StructureType.Free:
        max_strength = 0.4
        ratio = plausibility_ratio(vectors, max_strength)
        good_vectors = ratio[1]

        if (len(good_vectors) > 0):
            score_strength = strength_number(good_vectors, max_strength)
            score_number = min(len(good_vectors), 15) / 15
            score_s = swarm_score(good_vectors)
            score_d = 0.5 * score_s + 0.1 * score_strength + 0.4 * score_number
    else:
        score_d = inside_outside_score(good_vectors, w, h)

    return score_d

