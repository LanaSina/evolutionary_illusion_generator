import argparse
from mirror_images import mirror
import os

parser = argparse.ArgumentParser(description='optical flow tests')
parser.add_argument('--input', '-i', default='', help='Path to the directory which countains the input_images directory')
parser.add_argument('--output_dir', '-o', default='mirrored', help='Images will be saved in output_dir/input_images')
parser.add_argument('--limit', '-l', type=int, default=-1, help='max number of images')

args = parser.parse_args()
output_dir = args.output_dir+"/input_images" 
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

mirror(args.input, output_dir, args.limit)