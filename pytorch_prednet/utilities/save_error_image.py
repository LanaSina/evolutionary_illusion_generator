import argparse
import numpy as np
import os
from PIL import Image


# return an image with only overpredicted colors
def color_diff(input_image_path, prediction_path, output_dir):
    # create image with only the strongest predicted in r,g and b
    input_image = np.array(Image.open(input_image_path).convert('RGB'))
    prediction = np.array(Image.open(prediction_path).convert('RGB'))

    diff = 1.0*input_image - prediction

    plus_error = np.zeros(prediction.shape)
    minus_error = np.zeros(prediction.shape)

    # for i in range(0,prediction.shape[0]):
    #     for j in range(0,prediction.shape[1]):

    #         for c in range(0,3):
    #             if( diff[i, j, c] > 0 ):
    #                 plus_error[i, j, c] = diff[i, j, c]*2
    #             else :
    #                 minus_error[i, j, c] = -diff[i, j, c]*2
    

    combined = np.ones(prediction.shape)
    combined = combined*128 

    # save it
    # image_array = Image.fromarray(plus_error.astype('uint8'), 'RGB')
    # name = output_dir + "_in-pre.png"
    # image_array.save(name)
    # print("saved image ", name)
    # image_array = Image.fromarray(minus_error.astype('uint8'), 'RGB')
    # name = output_dir + "_pre-in.png"
    # image_array.save(name)
    # print("saved image ", name)

    # plus_error = combined + plus_error
    # image_array = Image.fromarray(plus_error.astype('uint8'), 'RGB')
    # name = output_dir + "_in-pre_offset.png"
    # image_array.save(name)
    # print("saved image ", name)

    # minus_error = combined + minus_error
    # image_array = Image.fromarray(minus_error.astype('uint8'), 'RGB')
    # name = output_dir + "_pre-in_offset.png"
    # image_array.save(name)
    # print("saved image ", name)

    combined = combined+ diff
    image_array = Image.fromarray(combined.astype('uint8'), 'RGB')
    name = output_dir + "_combi.png"
    image_array.save(name)
    print("saved image ", name)


def save_errors(input_path, prediction_path, output_dir):
    input_list = sorted(os.listdir(input_path))
    prediction_list = sorted(os.listdir(prediction_path))
    n = len(input_list)

    for i in range(0,n-1):
        input_image_path = input_path + "/" + input_list[i+1]#next input!!
        input_image = np.array(Image.open(input_image_path).convert('RGB'))
        prediction_image_path = prediction_path + "/" + prediction_list[i]
        prediction = np.array(Image.open(prediction_image_path).convert('RGB'))

        diff = 1.0*input_image - prediction
        mse = (np.square(input_image - prediction)).mean(axis=None)
        print("mse ", mse)

        combined = np.ones(prediction.shape)
        combined = combined*128 

        combined = combined + diff
        image_array = Image.fromarray(combined.astype('uint8'), 'RGB')
        name = output_dir + "/" + prediction_list[i]
        image_array.save(name)
        print("saved image ", name)

def save_error(input_path, prediction_path, output_dir):
    input_image_path = input_path
    input_image = np.array(Image.open(input_image_path).convert('RGB'))
    prediction_image_path = prediction_path
    prediction = np.array(Image.open(prediction_image_path).convert('RGB'))

    diff = 1.0*input_image - prediction
    mse = (np.square(input_image - prediction)).mean(axis=None)
    print("mse ", mse)

    combined = np.ones(prediction.shape)
    combined = combined*128 

    combined = combined + diff
    image_array = Image.fromarray(combined.astype('uint8'), 'RGB')
    name = output_dir + "/error.png"
    image_array.save(name)
    print("saved image ", name)

# save places where right error is < left error
# ie the illusion-learning model is correct
def save_errors_left(input_path, prediction_path, output_dir, rep):
    input_list = sorted(os.listdir(input_path))
    prediction_list = sorted(os.listdir(prediction_path))
    n = len(input_list)
    w = 160

    count = 0
    for i in range(0,n):
        if((i+1)%rep!=0):
            continue

        input_image_path = input_path + "/" + input_list[count+1]
        # print(input_image_path)
        # create image with only the strongest predicted in r,g and b
        input_image = np.array(Image.open(input_image_path).convert('RGB'))
        prediction_image_path = prediction_path + "/" + prediction_list[i]
        prediction = np.array(Image.open(prediction_image_path).convert('RGB'))

        #error
        diff = 1.0*input_image - prediction
        # left - right error
        # lr_diff = np.zeros(prediction.shape)
        # lr_diff[:,0:int(w/2),:] = diff[:,0:int(w/2),:] - diff[:,int(w/2):w,:]

        combined = np.ones(prediction.shape)
        combined = combined*128 + diff

        # # combined = combined + lr_diff
        # # image_array = Image.fromarray(combined.astype('uint8'), 'RGB')
        # # name = output_dir + "/" + prediction_list[i]
        # # image_array.save(name)
        # # print("saved image ", name)

        # # enhanced = np.zeros(prediction.shape)

        # for k in range(0,combined.shape[0]):
        #     for l in range(0,int(w/2)): 
        #         for c in range(0,3):
        #             if lr_diff[k,l,c] > 0:
        #                 combined[k,l,c] = combined[k,l,c] + lr_diff[k,l,c]*5
        #                 #enhanced[k,l,c] = lr_diff[k,l,c]
        #             # else:
        #                 #enhanced[k,l,c] = input_image [k,l,c]

        image_array = Image.fromarray(combined.astype('uint8'), 'RGB')
        if(rep==1):
            name = output_dir + "/" + prediction_list[i]
        else:
            name = output_dir + "/" + str(count).zfill(10) + ".png"
        
        count = count + 1
        image_array.save(name)
        print("saved image ", name)

parser = argparse.ArgumentParser(description='image_analysis')
parser.add_argument('--input', '-i', default='', help='Path to input image or directory')
parser.add_argument('--prediction', '-p', default='', help='Path to predicted image or diectory')
parser.add_argument('--output_dir', '-o', default='', help='path of output diectory')
parser.add_argument('--rep', '-r', type=int, default=1, help='number of images to skip (eg 5 to skip 0..3, 5..8')


args = parser.parse_args()
output_dir = args.output_dir #"image_analysis/averages/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#save_errors_left(args.input, args.prediction, output_dir, args.rep)
save_error(args.input, args.prediction, output_dir)

