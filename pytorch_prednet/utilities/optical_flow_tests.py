import argparse
import csv
import cv2
import os
from PredNet.call_prednet import call_prednet
from generate_img_func import generate_imagelist
import numpy as np
from optical_flow.optical_flow import lucas_kanade, draw_tracks
from utilities.mirror_images import mirror_multiple, TransformationType

# This file runs a mirrored flow analysis to find which images contain illusions

def run_prednet(input_path, model_name, limit, repeat, output_dir):
    print("run prednet")
    l = limit*10
    class PrednetArgs:
        images = input_path
        initmodel = model_name
        input_len = l
        ext = 0
        ext_t = -1
        bprop = 20
        save = 10000
        period = 1000000
        test = True
        skip_save_frames = repeat

        sequences = ''
        gpu = 0 # -1
        root = "."
        resume = ''
        size = '160,120'
        channels = '3,48,96,192'
        offset = "0,0"

    # only save last image
    # %run 'chainer_prednet/PredNet/main.py' --images 'imported' --initmodel 'fpsi_500000_20v.model' --input_len 10 --test 
    prednet_args = PrednetArgs()
    call_prednet(prednet_args, output_dir)

def make_img_list(input_path, limit, repeat):
    print("create image list")
    # 'chainer_prednet/generate_imagelist.py' 'imported/' '1' -tr 10
    parser = argparse.ArgumentParser(description='generate_imagelist args')
    class ImglistArgs:
        data_dir = input_path
        n_images = limit
        rep = repeat
        total_rep = 1

    imagelist_args = ImglistArgs()
    generate_imagelist(imagelist_args)

# return true if there are some strong vectors in there
def strong_vectors(vectors, threshold):

    if(len(vectors)==0):
        return False
    # data is rows of [x, y, dx, dy]
    if (sum(np.abs(vectors[:,2]))>threshold):
        return True
    if (sum(np.abs(vectors[:,3]))>threshold):
        return True  
    return False

#save(results, mirrored_results, filename, output_path="."):
def  save(img, mirror_img, good_vectors, filename, output_path="."):
    v = good_vectors["original"]
    for i in range(0,len(v)):
        draw_tracks(img, v[i][0], v[i][1], v[i][2], v[i][3])

    v = good_vectors["mirrored"]
    for i in range(0,len(v)):
        draw_tracks(mirror_img, v[i][0], v[i][1], v[i][2], v[i][3])

    name = filename.split("/")
    name = name[len(name)-1]
    temp = name.split(".")


    output_path_long = output_path + "/original/" 
    output_file = output_path_long+ temp[0] + ".png"
    print("saving", output_file)
    cv2.imwrite(output_file, img)   
    output_file = output_path_long + "/csv/" + temp[0] +".csv" 
    with open(output_file, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(good_vectors["original"])

    output_path_long = output_path + "/mirrored/"
    output_file = output_path_long + temp[0] + ".png"
    print("saving", output_file)
    cv2.imwrite(output_file, mirror_img)  
    output_file = output_path_long + "/csv/" + temp[0] +".csv" 
    with open(output_file, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(good_vectors["mirrored"])
   

# returns true if one direction seems to have a motion illusion
def mirror_test(vectors, mirrored_vectors, mtype, threshold):
    v = []
    v_m = []

    # sum quarter by quarter
    w = 160
    h = 120
    step = 40 #px
    x = 0
    while x<w :
        xx = x
        x = x + step
        y = 0
        subset_cond = ((vectors[:,0] >= xx) & (vectors[:,0] < xx +step))
        subset_x = vectors[subset_cond]
        if(len(subset_x) == 0):
            continue

        # subset depends on transformation type
        if (TransformationType(mtype) == TransformationType.Mirror or TransformationType(mtype) == TransformationType.MirrorAndFlip):
            subset_cond = ((mirrored_vectors[:,0] <= w - xx) & (mirrored_vectors[:,0] > (w-xx-step)))
        else:
            subset_cond = ((mirrored_vectors[:,0] >= xx) & (mirrored_vectors[:,0] < xx +step))

        subset_xm = mirrored_vectors[subset_cond]
        if(len(subset_xm) == 0):
            continue

        while y<h :
            yy = y
            y = y + step

            # select vecotrs on a cell
            subset_cond = ((subset_x[:,1] >= yy) & (subset_x[:,1] < yy + step))
            subset_y = subset_x[subset_cond]
            if(len(subset_y) == 0):
                continue

            if (TransformationType(mtype) == TransformationType.Flip or TransformationType(mtype) == TransformationType.MirrorAndFlip):
                subset_cond = ((subset_xm[:,1] <= (h-yy)) & (subset_xm[:,1] > (h-yy-step)))
            else:
                subset_cond = ((subset_xm[:,1] >= yy) & (subset_xm[:,1] < yy + step))

            subset_ym = subset_xm[subset_cond]
            if(len(subset_ym) == 0):
                continue

            # take the mean direction on original image
            # check x and y separately because of model bias
            if(TransformationType(mtype) == TransformationType.Mirror or TransformationType(mtype) == TransformationType.MirrorAndFlip):
                #print("dx means", np.mean(np.abs(subset_y[:,2])), np.mean(np.abs(subset_ym[:,2])))
                if np.mean(np.abs(subset_y[:,2])) > threshold or np.mean(np.abs(subset_ym[:,2])) > threshold:
                    # vmean = np.mean(subset_y[:,2]) + np.mean(subset_ym[:,2])
                    # if np.abs(vmean)<threshold :
                    sign = np.mean(subset_y[:,2]) * np.mean(subset_ym[:,2])
                    if sign<0 :
                        print("mirror_test passed on x ")
                        v.extend(subset_y)
                        v_m.extend(subset_ym)
                        continue

            if(TransformationType(mtype) == TransformationType.Flip or TransformationType(mtype) == TransformationType.MirrorAndFlip):
                # print("dy means", np.mean(subset_y[:,3]), np.mean(subset_ym[:,3]))
                if np.mean(np.abs(subset_y[:,3])) > threshold or np.mean(np.abs(subset_ym[:,3])) > threshold:
                    # vmean = np.mean(subset_y[:,3]) + np.mean(subset_ym[:,3])
                    # if np.abs(vmean)<threshold :
                    sign = np.mean(subset_y[:,3]) * np.mean(subset_ym[:,3])
                    if sign<0 :
                        print("mirror_test passed on y ")
                        v.extend(subset_y)
                        v_m.extend(subset_ym)
                        #return True

    results = {"original":v, "mirrored":v_m}
    return results

# stype is boolean
def compare_flow(input_image_dir, output_dir, limit, stype, mtype):
    threshold = 0.05

    # calculate optical flow compared to input
    print("calculate optical flow")
    if not os.path.exists(output_dir+"/original/"):
        os.makedirs(output_dir+"/original/")
    if not os.path.exists(output_dir+"/mirrored/"):
        os.makedirs(output_dir+"/mirrored/")
    if not os.path.exists(output_dir+"/csv"):
        os.makedirs(output_dir+"/csv")

    input_image_list = sorted(os.listdir(input_image_dir))
    prediction_image_dir = "result"
    prediction_image_list = sorted(os.listdir(prediction_image_dir))
    mirrored_image_dir = "mirrored/input_images"
    mirrored_image_list = sorted(os.listdir(mirrored_image_dir))
    mirrored_prediction_image_dir = "mirrored_result"
    mirrored_prediction_image_list = sorted(os.listdir(prediction_image_dir))

    # python optical_flow.py test_20y_0.jpg test_20y_1.jpg -s 0 -l 1 -cc yellow -lc red -s 2 -l 2 -vs 60.0
    for i in range(0,limit):

        # results for original image 
        original_image = input_image_list[i]
        # original input
        original_image_path = os.path.join(input_image_dir, original_image)
        # prediction
        prediction_image_path = prediction_image_dir + "/" + prediction_image_list[i] 
        results = lucas_kanade(original_image_path, prediction_image_path, output_dir+"/original/", save=stype)
        results["vectors"] = np.asarray(results["vectors"])

        # reject too small vectors
        if (not strong_vectors(results["vectors"], threshold)):
            print("no strong vectors in original image", i)
            continue
        
        # results for mirrored image 
        mirrored_image = mirrored_image_list[i]
        # original input
        mirrored_image_path = os.path.join(mirrored_image_dir, mirrored_image)
        # prediction
        mirrored_prediction_image_path = mirrored_prediction_image_dir + "/" + mirrored_prediction_image_list[i] 
        mirrored_results = lucas_kanade(mirrored_image_path, mirrored_prediction_image_path, output_dir+"/mirrored/", save=stype)
        mirrored_results["vectors"] = np.asarray(mirrored_results["vectors"])

        if (not strong_vectors(mirrored_results["vectors"], threshold)):
            print("no strong vectors in mirrored image", i)
            continue

        # analyse the vectors
        good_vectors = mirror_test(results["vectors"], mirrored_results["vectors"], mtype, threshold)
        if (len(good_vectors["original"])>0):
            # save files and images
            if (not stype):
                #save(results, mirrored_results, original_image, output_dir)
                save(results["image"], mirrored_results["image"], good_vectors, original_image, output_dir)


# process images as static images
def predict_static(input_path, output_dir, model_name, limit, repeat=10, mtype=0, stype=0):
    input_image_dir = input_path + "/input_images/"
    input_image_list = sorted(os.listdir(input_image_dir))
    if limit==-1:
        limit = len(input_image_list)

    # predict original images
    make_img_list(input_path, limit, repeat)
    run_prednet(input_path, model_name, limit, repeat, "result")
    # predict mirrored images
    # "chainer_prednet/utilities/mirror_images.py" -i "imported/input_images" -o "mirrored"
    mirror_images_path = "mirrored"
    mirror_images_dir = "mirrored/input_images"
    if not os.path.exists(mirror_images_dir):
        os.makedirs(mirror_images_dir)
    mirror_multiple(input_image_dir, mirror_images_dir, limit, TransformationType(mtype))
    make_img_list(mirror_images_path, limit, repeat)
    run_prednet(mirror_images_path, model_name, limit, repeat, "mirrored_result")

    # now compare image by image
    save_type = True
    if (stype == 1):
        save_type = False
    compare_flow(input_image_dir, output_dir, limit, save_type, mtype)


parser = argparse.ArgumentParser(description='optical flow tests')
parser.add_argument('--input', '-i', default='', help='Path to the directory which countains the input_images directory')
parser.add_argument('--model', '-m', default='', help='.model file')
parser.add_argument('--output_dir', '-o', default='.', help='path of output diectory')
parser.add_argument('--limit', '-l', type=int, default=-1, help='max number of images')
parser.add_argument('--repeat', '-r', type=int, default=10, help='number of times to repeat image before calculating flow')
parser.add_argument('--mirror_type', '-mt', type=int, default=0, help='0 for mirroring, 1 for flipping, 2 for mirrored and flipped')
parser.add_argument('--save_type', '-st', type=int, default=0, help='0 for saving all images, 1 for saving only detected illusions')

args = parser.parse_args()
output_dir = args.output_dir 
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

predict_static(args.input,output_dir, args.model, args.limit, args.repeat, args.mirror_type, args.save_type)
