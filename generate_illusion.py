import argparse
import cv2
import csv
import numpy as np
from ..optical_flow.optical_flow import lucas_kanade
import os
from PIL import Image
from PredNet.call_prednet import test_prednet
from random import random, randrange
from utilities.mirror_images import mirror, mirror_multiple, TransformationType

from pytorch_neat.pytorch_neat.cppn import create_cppn
from pytorch_neat.pytorch_neat.multi_env_eval import MultiEnvEvaluator
from pytorch_neat.pytorch_neat.neat_reporter import LogReporter
from pytorch_neat.pytorch_neat.recurrent_net import RecurrentNet
import neat
import torch


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
        #comp_y = comp_y + abs(vector[3])/norm

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

# returns 1 if vectors all aligned on x to the right; 
# -1 if to the left
def direction_ratio(vectors, limits = None):
    # print(vectors)
    mean_ratio = 0
    count = 0
    # make sure that all vectors are on x axis
    for v in vectors:
        if not limits is None:
            if (v[1]<limits[0]) or (v[1]>limits[1]):
                continue
        # x length divided by norm
        norm_v = np.sqrt(v[2]*v[2] + v[3]*v[3])
        ratio = v[2]/norm_v
        mean_ratio = mean_ratio + ratio
        count = count + 1

    if count>0:
        mean_ratio = mean_ratio / count
    else:
        mean_ratio = 0

    return mean_ratio



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
        #s_sum = abs(global_sum[0]) + abs(global_sum[1])

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


def create_grid(x_res = 32, y_res = 32, scaling = 1.0):

    num_points = x_res*y_res
    # repeat x a few times
    # rep = 5
    # nx = int(160/rep)
    # sc = scaling/rep
    # a = np.linspace(-1*sc, sc, num = nx)
    # x_range = np.tile(a, rep)
    x_range = np.linspace(-1*scaling, scaling, num = x_res)
    y_range = np.linspace(-1*scaling, scaling, num = y_res)
    x_mat = np.matmul(np.ones((y_res, 1)), x_range.reshape((1, x_res)))
    y_mat = np.matmul(y_range.reshape((y_res, 1)), np.ones((1, x_res)))
    r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)

    # s_mat_1 = y_mat < 0
    # s_mat_1 = s_mat_1.astype(int)
    # s_mat_m1 = y_mat >= 0
    # s_mat_m1 = s_mat_m1.astype(int)
    # s_mat = s_mat_1 - s_mat_m1
    s_mat = np.ones((num_points))

    x_mat = np.tile(x_mat.flatten(), 1).reshape(1, num_points, 1)
    y_mat = np.tile(y_mat.flatten(), 1).reshape(1, num_points, 1)
    r_mat = np.tile(r_mat.flatten(), 1).reshape(1, num_points, 1)
    s_mat = np.tile(s_mat.flatten(), 1).reshape(1, num_points, 1)

    return x_mat, y_mat, r_mat, s_mat

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

def get_image_from_cppn(genome, c_dim, w, h, config, s_val = 1):
    half_h = int(h/2)
    scaling = 4
    leaf_names = ["x","y","s"]
    out_names = ["r0","g0","b0","r1","g1","b1"]
    x_rep = 5
    x_subwidth = int(160/x_rep)
    x_dat, y_dat, r_dat, s_dat = create_grid(x_subwidth, half_h, scaling)
    s_dat = s_val*s_dat

    inp_x = torch.tensor(x_dat.flatten())
    inp_y = torch.tensor(y_dat.flatten())
    inp_s = torch.tensor(s_dat.flatten())
    inp_minus_s = torch.tensor(-s_dat.flatten())
    #reverse
    x0 = x_dat[:,::-1,:].flatten()
    inv_x = torch.tensor(x0.flatten())

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

                pixels = node_func(x=inp_x, y=inp_y, s = inp_s)
                pixels_np = pixels.numpy()
                pixels = node_func(x=inv_x, y=inp_y, s = inp_s)
                reverse_pixels_np = pixels.numpy()
                for x_slice in range(0,x_rep):
                    start = x_slice*x_subwidth
                    image_array[0:half_h, start:(start+x_subwidth), c] = np.reshape(pixels_np, (half_h,x_subwidth))
                    image_array[half_h:h, start:(start+x_subwidth), c] = np.reshape(reverse_pixels_np, (half_h,x_subwidth))

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
        pixels = node_func(x=inp_x, y=inp_y, s = inp_s)
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
def get_fitnesses_neat(population, model_name, config, id=0, c_dim=3):
    print("fitnesses of ", len(population))
    output_dir = "temp" + str(id) + "/"
    repeat = 10
    w = 160
    h = 120
    half_h = int(h/2)
    size = [w,h]
    channels = [3,48,96,192]
    gpu = 0

    prediction_dir = output_dir + "/original/prediction/"
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    # mirror_dir = output_dir + "mirrored/"
    # if not os.path.exists(mirror_dir+ "flow/"):
    #     os.makedirs(mirror_dir +"flow/")
    # mirror_images_dir = mirror_dir+ "images/"
    # if not os.path.exists(mirror_images_dir):
    #     os.makedirs(mirror_images_dir)

    if not os.path.exists(output_dir + "images/"):
        os.makedirs(output_dir + "images/")

    s_step = 0.2
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
            image = get_image_from_cppn(genome, c_dim, w, h, config, s_val = s_val)

            image_name = output_dir + "images/" + str(index).zfill(10) + ".png"
            images_list[index] = image_name
            repeated_images_list[index*repeat:(index+1)*repeat] = [image_name]*repeat
            image.save(image_name, "PNG")
            j = j+1
        i = i + 1

    # runs repeat x times on the input image, save in result folder
    test_prednet(initmodel = model_name, images_list = repeated_images_list, size=size, 
                channels = channels, gpu = gpu, output_dir = prediction_dir, skip_save_frames=repeat,
                reset_each = True,
                )
    # calculate flows
    i = 0
    original_vectors = [None] * total_count
    #fidelity = [None] * len(population)
    for input_image in images_list:
        prediction_image_path = prediction_dir + str(i).zfill(10) + ".png"
        results = lucas_kanade(input_image, prediction_image_path, output_dir+"/original/flow/", save=True)
        #fidelity[i] = get_fidelity(input_image, prediction_image_path)
        if results["vectors"]:
            original_vectors[i] = np.asarray(results["vectors"])
        else:
            original_vectors[i] = [[0,0,-1000,0]]
        i = i + 1

    # #mirror images
    # mirror_multiple(output_dir + "images/", mirror_images_dir, TransformationType.MirrorAndFlip)
    # #print("mirror images finished")
    # temp_list = sorted(os.listdir(mirror_images_dir))
    # temp_list = temp_list[0:len(images_list)]
    # mirror_images_list = [mirror_images_dir + im for im in temp_list]
    # repeated_mirror_list = [mirror_images_dir + im for im in temp_list for i in range(repeat) ]

    # # predict
    # test_prednet(initmodel = model_name, images_list = repeated_mirror_list, size=size, 
    #             channels = channels, gpu = gpu, output_dir = mirror_dir + "prediction/", skip_save_frames=repeat,
    #             reset_each = True
    #             )
    # # calculate flow
    # i = 0
    # mirrored_vectors = [None] * len(population)
    # for input_image in mirror_images_list:
    #     print(input_image)
    #     prediction_image_path = mirror_dir + "prediction/" + str(i).zfill(10) + ".png"
    #     print(prediction_image_path)
    #     results = lucas_kanade(input_image, prediction_image_path, output_dir+"/mirrored/flow/", save=True)
    #     if results["vectors"]:
    #         mirrored_vectors[i] = np.asarray(results["vectors"])
    #     else:
    #         mirrored_vectors[i] = [[0,0,-1000,0]]
    #     i = i + 1

    # calculate score
    #radius_limits = [20,50]
    scores = [None] * len(population)
    for i in range(0, len(population)):
        #score = combined_illusion_score(original_vectors[i], mirrored_vectors[i])
        final_score = -100
        temp_index = -1
        mean_score = 0
        for j in range(0,int(2/s_step)):
            index = i*pertype_count+j
            score = 0
            if(len(original_vectors[index])>0):
                # bonus
                score = score + 0.1
                ratio = plausibility_ratio(original_vectors[index])
                score_0 = ratio[0]
                good_vectors = ratio[1]

                if(len(good_vectors)>0): 
                    score = score + 0.1
                    step = h/2
                    y = 0                
                    count = 0
                    score_2 = [None]*2
                    while y<h:
                        limit = [y, y+step]
                        score_2[count] = direction_ratio(good_vectors, limit)
                        y = y + step
                        count = count + 1

                    # bonus points
                    if(score_2[0]*score_2[1]<0):
                        # is the ideal number of vectors
                        temp = 24 - len(good_vectors)
                        if(temp==0):
                            n_dist = 1
                        else:
                            n_dist = 1/temp*temp
                        score = score + n_dist*(abs(score_2[0]) + abs(score_2[1]))/2
                        mean_score = mean_score + score
                if score>final_score:
                    final_score = score
                    temp_index = index
        
        print("index ", temp_index, " score ", final_score)
        scores[i] =[i, mean_score/pertype_count]

    print("scores",scores)
    i = 0
    for genome_id, genome in population:
        genome.fitness = scores[i][1]
        i = i+1


def neat_illusion(input_image, output_dir, model_name, checkpoint = None):
    repeat = 6
    limit = 1
    w = 160
    h = 120
    half_h = int(h/2)
    size = [w,h]
    channels = [3,48,96,192]
    gpu = 0
    c_dim = 3

    best_dir = output_dir + "best/"
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "chainer_prednet/neat.cfg")

    def eval_genomes(genomes, config):
        get_fitnesses_neat(genomes, model_name, config, c_dim=c_dim)

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

    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    # image = get_image_from_cppn(winner, c_dim, w, h, config)

    # image.save("best_illusion.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='optical flow tests')
    parser.add_argument('--input', '-i', default='', help='Path to the directory which countains the input_images directory')
    parser.add_argument('--model', '-m', default='', help='.model file')
    parser.add_argument('--output_dir', '-o', default='.', help='path of output diectory')
    parser.add_argument('--checkpoint', '-c', help='path of checkpoint to restore')

    args = parser.parse_args()
    output_dir = args.output_dir 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    neat_illusion(args.input, output_dir, args.model, args.checkpoint)

