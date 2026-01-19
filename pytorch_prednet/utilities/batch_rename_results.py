import argparse
import os

parser = argparse.ArgumentParser(description='image_analysis')
parser.add_argument('image_dir', help='Path to prednet output images')
parser.add_argument('--padding', "--p",  default=4, type=int, help='Number of padding 0s')
args = parser.parse_args()

path = args.image_dir
for filename in os.listdir(path):
	num = filename.split('_')[1].split('y')[0]
	temp = filename.split(".")
	ext = temp[len(temp)-1]
	print(num)
	num = num.zfill(args.padding)
	new_filename = num + "." + ext
	os.rename(os.path.join(path, filename), os.path.join(path, new_filename))