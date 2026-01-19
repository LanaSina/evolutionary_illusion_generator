import argparse
import numpy as np
import os
from PIL import Image
import random


# counts non-null pixels in an image
def count(input_image_path):
    input_image = np.array(Image.open(input_image_path).convert('RGB'))
    sum_ = (input_image).mean(axis=2)          
    print("non null pixels", np.count_nonzero(sum_))

# select random pixels in an image, fills the rest with random values
def save(input_image_path, output_dir, n):
    input_image = np.array(Image.open(input_image_path).convert('RGB'))
    new_image = np.random.rand(input_image.shape[0], input_image.shape[1], input_image.shape[2])*256 #np.zeros(input_image.shape).astype('uint8')
    new_image =  new_image.astype('uint8')

    for index in range(0,n):
        i = random.randint(0, input_image.shape[0]-1)
        j = random.randint(0, input_image.shape[1]-1)
        new_image[i,j] = input_image[i,j]

    image_array = Image.fromarray(new_image.astype('uint8'), 'RGB')
    name = output_dir + "/pixels_" + str(n) + ".png"
    image_array.save(name)
    print("saved image ", name)

parser = argparse.ArgumentParser(description='image_analysis')
parser.add_argument('--input', '-i', default='', help='Path to 1st input directory')
parser.add_argument('--output_dir', '-o', default='', help='path of output diectory')
parser.add_argument('--type', '-t', type=int, default=0, help='0 for count, 1 for generate')
parser.add_argument('--n', '-n', type=int, default=100, help='number of pixels to keep')


args = parser.parse_args()

if args.type == 0:
    count(args.input)
else:
    output_dir = args.output_dir #"image_analysis/averages/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save(args.input, output_dir, args.n)