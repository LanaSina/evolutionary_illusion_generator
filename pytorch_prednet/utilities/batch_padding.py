import argparse
import os

parser = argparse.ArgumentParser(description='image_analysis')
parser.add_argument('image_dir', help='Path to prednet output images')
parser.add_argument('--padding', "--p",  default=4, type=int, help='Number of padding 0s')
parser.add_argument('--prefix',  default="", help='prefix before the file number')

args = parser.parse_args()

path = args.image_dir
for filename in sorted(os.listdir(path)):
	temp = filename.split(".")
	num = temp[0]
	if(len(args.prefix)>0):
		num = temp[0].split(args.prefix)[1]
	ext = temp[len(temp)-1]
	num = num.zfill(args.padding)
	new_filename = num + "." + ext
	print(new_filename)
	os.rename(os.path.join(path, filename), os.path.join(path, new_filename))