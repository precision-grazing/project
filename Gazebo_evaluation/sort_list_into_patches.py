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

start_time = time.time()


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
# days = [98,105,112,119,126,133,140,147,154]

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






# with open('30percent_variance_max_height_day168' + '.npy', 'rb') as f:
#     x_y_day_heights_coords = np.load(f)

# with open('30percent_variance_least_height_day365' + '.npy', 'rb') as f:
#     x_y_day_heights_coords = np.load(f)

# with open('y_coords_numpy.npy', 'rb') as f:
#     y_coords = np.load(f)

# with open('day_heights_numpy.npy', 'rb') as f:
#     day_heights = np.load(f)

#initialise a 2D numpy array

# print("patch_1_10by10_arr numpy aray is: ", patch_1_10by10_arr, "Numpy shape: ", patch_1_10by10_arr.shape)


for day_number in days:

    with open('x_y_heights_coords_numpy_day_' + str(day_number) + '.npy', 'rb') as f:
        x_y_day_heights_coords = np.load(f)

    print("Number of loops: ", len(x_patch_points)* len(y_patch_points))
    loop_counter = 1

    # pickle_poses = open("pasture_day30_450plants_per_square_meter.pickle", "rb")
    pickle_poses = open("pasture_day30_250plants_per_square_meter.pickle", "rb")


    species_1_poses_list = pickle.load(pickle_poses)

           
    for x_loop in x_patch_points:
        print("Currently in Loop: ", loop_counter)

        for y_loop in y_patch_points:
            

            patch_index_x = int((x_loop + x_step_size_patch)/x_step_size_patch)
            patch_index_y = int((y_loop + y_step_size_patch)/y_step_size_patch)

            print("Patch no.: " + str(patch_index_x) + "_" + str(patch_index_y))

            patch_name = "patch_" + str(patch_index_x) + "_" + str(patch_index_y) + "_" + str(x_step_size_patch) + "by" + str(y_step_size_patch) + "_pasturesize" + str(pasture_x_width) +  "by" + str(pasture_y_length)+  "m"

            patch_arr = []
            patch_array_name_string = patch_name + "npy_array"


            for plant_number in range (species1_plants_in_pasture):   #405000
                # print("Plant number is:  ", plant_number)
                x_list_value = x_y_day_heights_coords[plant_number][0]
                y_list_value = x_y_day_heights_coords[plant_number][1]
                plant_height = x_y_day_heights_coords[plant_number][2]

                # print("x_y_day_heights_coords numpy array: ",x_y_day_heights_coords[plant_number], "Numpy shape: ", x_y_day_heights_coords[plant_number].shape)

                # print("X list value is", x_list_value)
                # print("X list value is", y_list_value)
                # print("Plant_height value is", plant_height)


                add_or_not_boolean_x =  (x_list_value >= x_loop) and (x_list_value <= x_loop + x_step_size_patch)
                add_or_not_boolean_y =  (y_list_value >= y_loop) and (y_list_value <= y_loop + y_step_size_patch)

                # print(add_or_not_boolean_x)
                # print(add_or_not_boolean_y)


                if (add_or_not_boolean_x and add_or_not_boolean_y):
                    # print(list(x_y_day_heights_coords[plant_number]))


                    height_list = list(x_y_day_heights_coords[plant_number])

                    #height list has rounded off x,y and jun's algo's height. 
                    #we have the rounded off height to confirm that we have the right height for that x,y
                    pose_and_height  = species_1_poses_list[plant_number] + height_list 
                    # print(pose_and_height)    

                    patch_arr.append(pose_and_height)

                    # patch_array_final = np.vstack((patch_arr, x_y_day_heights_coords[plant_number]))
                    # print(patch_arr)

                    # delete first row
                    #store patch array

            # patch_arr = np.delete(patch_arr, 0, 0)
            # print(patch_arr)
            # print(patch_arr.shape)

            patch_arr_numpy = np.array(patch_arr)
            # patch_numpy_save_location_path = '/home/ksa/Desktop/Pasture_Monitoring/FINAL_SCRIPTS/npy_patches/day365_min_height/' + patch_array_name_string

            patch_numpy_save_location_path = '/home/ksa/Desktop/Pasture_Monitoring/Patch_generation_pipeline1April/Day' +str(day_number) +'/npy_arrays/' + patch_array_name_string

            with open(patch_numpy_save_location_path + '.npy', 'wb') as f:
                print("Saving numpy array ")
                np.save(f, patch_arr_numpy)

        loop_counter = loop_counter + 1



    print("Generating ",species_1_number_of_patches, " patches took", (time.time() - start_time)/60, "minutes")




    #











# pickle_heights = open("/home/ksa/Desktop/Pasture_Monitoring/scripts/patch_x_y_locations/36patches_Pickle_files/pasture_heights_405000_plants.pickle", "rb")
# species_1_heights = pickle.load(pickle_heights)
# print(species_1_heights)
# #print(species_1_heights)
# scaling_factor = 1

# pickle_poses = open("/home/ksa/Desktop/Pasture_Monitoring/scripts/patch_x_y_locations/36patches_Pickle_files/patch_36_450plants_per_square_meter.pickle", "rb")
# species_1_poses_list = pickle.load(pickle_poses)
















































