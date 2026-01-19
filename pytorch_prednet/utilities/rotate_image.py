import os
from PIL import Image

# Rotate it by x degrees
rotation = 45
#1 for clockwise, -1 for counter-clockwise
rotation_direction = -1

# Create an Image object from an Image
input_image  = Image.open("./spiral.png")
print(input_image.size)
#to_crop = input_image.size[0] - input_image.size[1]
#to_crop = to_crop/2 
#print(to_crop)
#input_image = input_image.crop((to_crop, 0, input_image.size[0] - to_crop, input_image.size[1]))

output_directory = "./spiral_"+ str(-rotation_direction*rotation) +"/input_images/"
output_size = (160,128)
center = (output_size[0] - input_image.size[0], output_size[1]-input_image.size[1])
center = (center[0]/2, center[1]/2)

if not os.path.exists(output_directory):
    os.makedirs(output_directory)



n = 50 #360//rotation
#rotated = input_image #.convert('RGB')
#non_transparent = Image.new('RGB', output_size, (255,255,255))
#non_transparent.paste(input_image, center, mask = input_image)
#non_transparent.save(output_directory + "0.jpg")

# converted to have an alpha layer
input_image = input_image.convert('RGBA')

for i in range(0,n):
	rotated = input_image.rotate(rotation_direction*-i*rotation)
	non_transparent = Image.new('RGB', output_size, (255,255,255))
	non_transparent.paste(rotated, center, mask = rotated)
	non_transparent.save(output_directory +  str(i).zfill(3) + ".jpg")