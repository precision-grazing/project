import numpy as np
import pickle
import xml.etree.ElementTree as ET
# tree = ET.parse('pasture_ground_parameters.xml')
# root = tree.getroot()   #returns root element of the XML file
import math
from random import *
from collada import *
from time import process_time
import time



# Add prennial ryegrass plant heights
rygrass_plant_heights = {1: 0.170227,
2: 0.118349,
3: 0.179795,
4: 0.141542, 
5: 0.068646}

pasture_x_width = 10.0
pasture_y_length = 10.0
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
# day_number = 168
day_number = 211
# day_number = 154
species_1_submodels = 5
species_1_plants_per_square_meter = 250
species1_plants_in_pasture = species_1_plants_per_square_meter * pasture_x_width * pasture_y_length
print("species1_plants_in_pasture: ", species1_plants_in_pasture)

x_step_size_patch = patch_x_width
y_step_size_patch = patch_y_length
print("x_step_size_patch is: ", x_step_size_patch)
print("y_step_size_patch is: ", y_step_size_patch)

x_patch_points = [x for x in np.arange(0, pasture_x_width,x_step_size_patch)]

print("x_patch_points: ", x_patch_points)


y_patch_points = [y for y in np.arange(0, pasture_y_length,y_step_size_patch)]
print("y_patch_points: ", y_patch_points)

scaling_factor = 1
x_y_patch_points = []



#    x_y_patch_points = []
#    for x in np.arange(1.0, (pasture_x_width/patch_x_width)+1,1.0):
#        for y in np.arange(1.0, (pasture_y_length/patch_y_length)+1,1.0):
#            x_y_temp = [x,y]
#            x_y_patch_points.append(x_y_temp)
#    print(x_y_patch_points)


x_y_patch_points = []
for x in np.arange(1.0,11.0,1.0):
    for y in np.arange(1.0, 11.0,1.0):
        x_y_temp = [x,y]
        x_y_patch_points.append(x_y_temp)
print(x_y_patch_points)

#    multiprocessing_func(x_y_patch_points)


print("is this loop")





#start timer to measure time taken by python code to generate the world
start = process_time()

#we can add different patches, low density and high density patches
species_1_patches = {1: '/home/ksa/catkin_ws/src/hector_quadrotor_tutorial/hector_gazebo/hector_gazebo_worlds/worlds/8by6_ryegrass_patch_21600plants.dae.dae', 2: '/home/ksa/Desktop/Pasture_Monitoring/blender_dae_script/perennialRyegrass.dae'}


#we can have a fixed mean height for the patch
#scale the entire patch, instead of each plant in the patch

mean_height_patch = 0.060   #60mm = 0.06 meters

scaling_factor = 1

species_name = 'ryegrass'
count = 1

tree = ET.parse('empty.world')
root = tree.getroot()
# model = ET.parse('../worlds/grass_model_format_no_collision.model')
model = ET.parse('grass_model_format.model')

model_root = model.getroot()

for patch_index in range(0,25,1):

    patch_start_time = time.time()
    # x_loop = patch_index[0]
    # print("x_loop: ", x_loop)
    # y_loop = patch_index[1]
    # print("Patch no.: " + str(x_loop) + "_" + str(y_loop))
    #     #/home/ksa/Desktop/Pasture_Monitoring/patch_generation/patches/npy_patches/patch_1.0_1.0_0.5by0.5_pasturesize30.0by30.0mnpy_array.npy

    # patch_name = "patch_" + str(x_loop) + "_" + str(y_loop) + "_" + str(x_step_size_patch) + "by" + str(y_step_size_patch) + "_pasturesize" + str(pasture_x_width) +  "by" + str(pasture_y_length)+  "m"
    

    #/home/ksa/Desktop/Pasture_Monitoring/patch_generation/patches/0.5_by_0.5m_dae_patches/0.5mby0.5m_patches_day7_30mby30m_pasture/patch_1.0_1.0_0.5by0.5_pasturesize30.0by30.0m.dae
    # file_path_string =  "/home/ksa/Desktop/Pasture_Monitoring/patches/day168_max_height_30percent_variance/2by2m_square_patch_index_" + str(patch_index)  + ".dae"
    file_path_string =  "/home/ksa/Desktop/Pasture_Monitoring/Patch_generation_pipeline1April/Day" + str(day_number) + "/patches/2by2m_square_patch_index_" + str(patch_index)  + ".dae"

    # /home/ksa/Desktop/Pasture_Monitoring/patches/day365_min_height_30percent_variance/2by2m_square_patch_index_0.dae

    x_val = 0.0
    y_val = 0.0
    z_val = 0.0

    roll = 0
    pitch = 0
    yaw = 0
    
    #scaling factor for easy viewing
    scaling_factor = 1

    x_scale = scaling_factor
    y_scale = scaling_factor
    z_scale = scaling_factor


    gazebo_model_name = species_name + str(count)
    model_file = file_path_string
    # print(model_file)

    pose_val = str(x_val) + " " + str(y_val) + " " + str(z_val) + " " +str(roll) + " " + str(pitch) + " " + str(yaw)

    scale_val = str(x_scale) + " " + str(y_scale) + " " + str(z_scale)

    grass_name = "grass_" + str(gazebo_model_name)
    link_name = "grass_" + str(gazebo_model_name) + "_link"
    col_name = "grass_" + str(gazebo_model_name) + "_collision"

    for model_name in model_root.iter('model'):
        model_name.set('name', grass_name)
    for model_name in model_root.iter('link'):
        model_name.set('name', link_name)
    for pose in model_root.iter('pose'):
        pose.text = pose_val
    for model_name in model_root.iter('collision'):
        print(model_name)
        model_name.set('name', col_name)

        
    for model_name in model_root.iter('visual'):
        model_name.set('name', grass_name)

    for scale_name in model_root.iter('scale'):
        scale_name.text = scale_val

    for model_location in model_root.iter('uri'):
        model_location.text = model_file
    for world in root.findall('world'):
        world.append(model_root)
    
    # tree.write('day'+str(day_number)+'MAX_HEIGHT_10by10m_species1_PASTURE.world')
    # tree = ET.parse('day'+str(day_number)+'MAX_HEIGHT_10by10m_species1_PASTURE.world')

    tree.write('day'+str(day_number)+'_2009.world')
    tree = ET.parse('day'+str(day_number)+'_2009.world')

    root = tree.getroot()
    count = count + 1

end = process_time()
print("Time to completion:", (end - start))


















































