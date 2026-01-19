import argparse
import math
import numpy as np
import os
from PIL import Image
from random import random, shuffle


# Create an Image object from an Image
output_path = "scrambled.png"

def scramble(cell_size, input_path):
	input_image  = Image.open(input_path)

	#make image divisible
	x_cells = math.ceil(input_image.size[0]/cell_size)
	y_cells = math.ceil(input_image.size[1]/cell_size)


	new_image = Image.new('RGB', (x_cells*cell_size, y_cells*cell_size), (255, 255, 255))
	new_image.paste(input_image)
	# this inverts x and y for some reason
	new_image = np.array(new_image)
	scrambled = Image.new('RGB', (x_cells*cell_size, y_cells*cell_size))
	scrambled = np.array(scrambled) #np.reshape(np.array(scrambled), (x_cells*cell_size,y_cells*cell_size,3))

	# np inverts PIL's x and y for some reason
	for y in range(x_cells):
		for x in range(y_cells):
			# scramble
			# x_range = np.linspace(x*cell_size, (x+1)*cell_size, num = cell_size, endpoint=False).astype(int)
			# y_range = np.linspace(y*cell_size, (y+1)*cell_size, num = cell_size, endpoint=False).astype(int)
			# shuffle(x_range)

			whole_range = np.arange(0,cell_size*cell_size) 
			shuffle(whole_range)
			flat_source = new_image[x*cell_size:(x+1)*cell_size, y*cell_size:(y+1)*cell_size,:]#.flatten()
			flat_source = flat_source.reshape((cell_size*cell_size,3))

			for xx in range(cell_size):
				#shuffle(y_range)
				for yy in range(cell_size):
					# scrambled[x_range[xx],y_range[yy],:] = new_image[x*cell_size+xx, y*cell_size+yy,:]
					scrambled[x*cell_size+xx, y*cell_size+yy,:] = flat_source[whole_range[xx*cell_size+yy]] 
		
	Image.fromarray(scrambled).save(output_path, "PNG")



parser = argparse.ArgumentParser(description='scrambler')
parser.add_argument('--cell', '-c', default=3, type=int, help='size (pixels) of each cell being scrambled')
parser.add_argument('--input', '-i', default='', help='image file')
args = parser.parse_args()

scramble(args.cell, args.input)