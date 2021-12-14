#import bpy         #bpy = blender python
import math
#import pandas as pd
#import xlrd
#from time import process_time
#import csv
import pickle

import time
#import multiprocessing 
from multiprocessing import Pool
import pandas as pd
import numpy as np
import os
import time


# # Add prennial ryegrass plant heights
# rygrass_plant_heights = {1: 0.170227,
# 2:0.118439,
# 3: 0.179798,
# 4: 0.141504, 
# 5: 0.068637}


pasture_x_width = 10
pasture_y_length = 10
pasture_area_m2 = pasture_x_width*pasture_y_length

patch_x_width = 2   #for x
patch_y_length = 2
patch_area_m2 = patch_x_width*patch_y_length

species_1_number_of_patches = int((pasture_area_m2/patch_area_m2))
print("Species_1_number_of_patches: ", species_1_number_of_patches)
number_of_x_patches = int(pasture_x_width/patch_x_width)
number_of_y_patches = int(pasture_y_length/patch_y_length)
print("number_of_x_patches: ", number_of_x_patches)
print("number_of_y_patches: ", number_of_y_patches)


num_of_vertices = 0
height_variation = 0 
# days = [133,140,147,154]
# days = [95,99,103,107,111,115,119,123,127,131]

days = [135,139,143,147,151,155,159,163,167,171,175,179,183,187,191,195,199,203,207,211]
species_1_submodels = 5
species_1_plants_per_square_meter = 250
species1_plants_in_pasture = species_1_plants_per_square_meter * pasture_x_width * pasture_y_length
print("species1_plants_in_pasture: ", species1_plants_in_pasture)

x_step_size_patch = patch_x_width
y_step_size_patch = patch_y_length
print("x_step_size_patch is: ", x_step_size_patch)
print("y_step_size_patch is: ", y_step_size_patch)
x_patch_points = [x for x in range(0, pasture_x_width,x_step_size_patch)]
print("x_patch_points: ", x_patch_points)
y_patch_points = [y for y in range(0, pasture_y_length,y_step_size_patch)]
print("y_patch_points: ", y_patch_points)

for day_number in days:

	############################################################################################
	# Save numpy array
	# /home/ksa/Desktop/Pasture_Monitoring/pose_extraction_creation_scripts/30mby30mheights.xlsx
	day_index = day_number + 1
	# df = pd.read_excel('2009_ground_truth.xlsx', sheet_name='Sheet1', usecols = [0,1,day_index])

	df = pd.read_excel('2009_ground_truth.xlsx', sheet_name='2009_ground_truth', usecols = [0,1,day_index])

	# select all rows and select first two columns only which have the x,y coords
	x_y_day_heights_coords = (df.iloc[:,[0,1,2]]).to_numpy()
	# y_coords = (df.iloc[:,[1]]).to_numpy()
	# day_heights = (df.iloc[:,[2]]).to_numpy()
	print("X coords, Y coords and plant heights at (X,Y) are: \n", x_y_day_heights_coords)
	# print(y_coords)
	# print(day_heights)


	with open('x_y_heights_coords_numpy_day_' + str(day_number) + '.npy', 'wb') as f:
	    np.save(f, x_y_day_heights_coords)



	# with open('30percent_variance_max_height_day168' + '.npy', 'wb') as f:
	#     np.save(f, x_y_day_heights_coords)


	# with open('30percent_variance_least_height_day365' + '.npy', 'wb') as f:
	#     np.save(f, x_y_day_heights_coords)

	# with open('y_coords_numpy_day_' + str(day_number) + '.npy', 'wb') as f:
	#     np.save(f, y_coords)

	# with open('day_heights_numpy_day_' + str(day_number) + '.npy', 'wb') as f:
	#     np.save(f, day_heights)




