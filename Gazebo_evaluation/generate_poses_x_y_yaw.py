############################################################################################
## This code generates the pose = (x,y,0,0,0,yaw) = (x,y,z,r,p,y) for pasture of given size
## Patch size can be varied, we are considering flat ground for now
############################################################################################
import numpy as np
import matplotlib.pyplot as plt
import math
# from math import pi   #can also do math.pi
import random
import xlrd
import time
import pickle
import pandas as pd 
    
#start timer to measure time taken by python code to generate the world
start = time.time()

# Add prennial ryegrass plant heights
rygrass_plant_heights = {1: 0.170227,
2: 0.118349,
3: 0.179795,
4: 0.141542, 
5: 0.068646}

pasture_x_width = 10.0
pasture_y_length = 10.0
pasture_area_m2 = pasture_x_width*pasture_y_length

patch_x_width = 2.0   #for x
patch_y_length = 2.0
patch_area_m2 = patch_x_width*patch_y_length

species_1_number_of_patches = int((pasture_area_m2/patch_area_m2))
print("Species_1_number_of_patches: ", species_1_number_of_patches)
number_of_x_patches = int(pasture_x_width/patch_x_width)
number_of_y_patches = int(pasture_y_length/patch_y_length)
print("number_of_x_patches: ", number_of_x_patches)
print("number_of_y_patches: ", number_of_y_patches)


day_number = 30
species_1_submodels = 5
species_1_plants_per_square_meter = 250
species1_plants_in_pasture = species_1_plants_per_square_meter * pasture_x_width * pasture_y_length
print("species1_plants_in_pasture: ", species1_plants_in_pasture)

x_step_size_patch = patch_x_width
y_step_size_patch = patch_y_length
print("x_step_size_patch is: ", x_step_size_patch)
print("y_step_size_patch is: ", y_step_size_patch)

# to generate the same random values everytime
random.seed(7)
x_origin = 0
y_origin = 0



#############################################################################################
# Add models to the world in gazebo for ryegrass

species_1_poses = []
species_name = 'ryegrass'

for j in range(0, int(species1_plants_in_pasture)):
    x_val = (random.random() * pasture_x_width) + x_origin
    y_val = (random.random() * pasture_y_length) + y_origin
    z_val = 0
    roll = 0
    pitch = 0
    yaw = random.random() * (2 * math.pi)

    model_num = random.randint(0, 4)
    #scaling factor for easy viewing
    scaling_factor = 1
    pose_val = str(x_val) + " " + str(y_val) + " " + str(z_val) + " " +str(roll) + " " + str(pitch) + " " + str(yaw)
    pose_list = [x_val, y_val, z_val, roll, pitch, yaw, model_num]
    print(pose_list)
    species_1_poses.append(pose_list)

pickle_out = open("pasture_day" + str(day_number) + "_250plants_per_square_meter.pickle","wb")
pickle.dump(species_1_poses, pickle_out)
pickle_out.close()

print(species_1_poses)

pickle_in = open("pasture_day" + str(day_number) + "_250plants_per_square_meter.pickle", "rb")
species_1_poses_list = pickle.load(pickle_in)

species_1_x_y_locations = []


for grass_plant_index in range(0, int(species1_plants_in_pasture)):

            x_val = round(species_1_poses_list[grass_plant_index][0],3)
            y_val = round(species_1_poses_list[grass_plant_index][1],3)
            # ground_truth_z_val = species_1_heights[0][grass_plant_index]
            # yaw = species_1_poses_list[grass_plant_index][5]
            # model_num = species_1_poses_list[grass_plant_index][6]
            # plt.scatter(x_val, y_val)
            # plt.show()
            # pose_list = [x_val, y_val, z_val, 0, 0, yaw, model_num]
            # x_y_z_ground_list = [x_val, y_val, ground_truth_z_val]
            x_y_z_ground_list = [x_val, y_val]

            # df['X-coordinate'] = x_y_z_ground_list[0::1]
            # df['Y-coordinate'] = x_y_z_ground_list[0::1] 

            species_1_x_y_locations.append(x_y_z_ground_list)

  
# Converting to excel 

df = pd.DataFrame() 

d1 = {'X_Y_coordinates':species_1_x_y_locations}
df2 = pd.DataFrame(d1)
print (df2)
# df3 = df2.teams.apply(pd.Series)
# df3.columns = ['team1', 'team2']

df3 = pd.DataFrame(df2['X_Y_coordinates'].to_list(), columns=['X-coordinate','Y-coordinate'])
print (df3)

# patch_name = 

df3.to_excel("pasture_day" + str(day_number) + "_xy_coords.xlsx", index = False)



end = time.time()

print("Time to completion:", (end - start))



































