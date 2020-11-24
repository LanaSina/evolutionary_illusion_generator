import argparse
from chainer_prednet.PredNet.call_prednet import test_prednet
import math
import numpy as np
from optical_flow.optical_flow import lucas_kanade, draw_tracks, save_data
import os

# returns the agreement and disagreement betwen vectors
def swarm_score(vectors):
    max_distance = 100 #px
    distance_2 = 50
    w = 160
    h = 120
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
        ocsv = ','.join(map(str, optimal))
        dcsv = ','.join(map(str, distances))
        #print("optimal", ocsv)
        print("distances", dcsv)
        loss = close*abs(angles-optimal)
        lcsv = ','.join(map(str, loss))
        print("loss", lcsv)
        temp = math.pi - (sum(loss)/n)
        score = score + (temp/math.pi)


    return score/n


def get_vectors(image_path, model_name):
    skip = 1
    extension_duration = 2
    repeat = 20
    w = 160
    h = 120
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tests')
    parser.add_argument('--model', '-m', default='', help='.model file')
    parser.add_argument('--input_image', '-i', default='')
    args = parser.parse_args()

    vectors = get_vectors(args.input_image, args.model)
    score = swarm_score(vectors)

    print("score", score)
