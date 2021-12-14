import bpy         #bpy = blender python
import math
#import pandas as pd
#import xlrd
#from time import process_time
#import csv
import pickle
import numpy as np
import time
#import multiprocessing 
from multiprocessing import Pool

day_number = 211

def add_plant_at_given_pose(x_val, y_val, z_val, yaw, model_num, plant_height):
   
    model_num_string = str(int(model_num))
    print(model_num_string)
    #file_path_string = "/home/ksa/Desktop/Pasture_Monitoring/grass_meshes/decimated_ryegrass_extruded_normals_out/perennialRyegrass_0" + model_num_string + ".dae"
    file_path_string = "/home/ksa/Desktop/Pasture_Monitoring/grass_meshes/Manual_export/low_poly_grass_13Nov/low_poly_grass/lowpoly_" + model_num_string + ".dae"
    #    print(file_path_string)
    
    ryegrass_predicted_height = plant_height    
    #    print("ryegrass_predicted_height", ryegrass_predicted_height)
    #    print("actual_blender_height", rygrass_plant_heights[model_num])
    
#    x_scale = scaling_factor*(ryegrass_predicted_height/rygrass_plant_heights[model_num])
#    y_scale = scaling_factor*(ryegrass_predicted_height/rygrass_plant_heights[model_num])
#    z_scale = scaling_factor*(ryegrass_predicted_height/rygrass_plant_heights[model_num])

    x_scale = scaling_factor*0.4
    y_scale = scaling_factor*0.4
    z_scale = scaling_factor*(ryegrass_predicted_height/lowpoly_plant_heights[model_num])

    bpy.ops.wm.collada_import(filepath=file_path_string)
    
    x_new = x_val - (x_offset[model_num]*x_scale)
    y_new = y_val - (y_offset[model_num]*y_scale)
    
    
    bpy.context.active_object.location = (x_new, y_new, z_val)
    bpy.context.active_object.rotation_euler = (0, 0, yaw)
    bpy.context.active_object.scale = (x_scale, y_scale, z_scale) 

    
def spawn_patch(x_y_loop_list):
    
    
    patch_start_time = time.time()
    x_loop = x_y_loop_list[0]
    print("x_loop: ", x_loop)
    y_loop = x_y_loop_list[1]
    print("Patch no.: " + str(x_loop) + "_" + str(y_loop))
        #/home/ksa/Desktop/Pasture_Monitoring/patch_generation/patches/npy_patches/patch_1.0_1.0_0.5by0.5_pasturesize30.0by30.0mnpy_array.npy
    patch_name = "patch_" + str(x_loop) + "_" + str(y_loop) + "_" + str(x_step_size_patch) + "by" + str(y_step_size_patch) + "_pasturesize" + str(pasture_x_width) +  "by" + str(pasture_y_length)+  "m"

    # /home/ksa/Desktop/Pasture_Monitoring/FINAL_SCRIPTS/npy_patches/day365_min_height/patch_1_1_2by2_pasturesize10by10mnpy_array.npy
    
    #/home/ksa/Desktop/Pasture_Monitoring/Patch_generation_pipeline1April/Day91/npy_arrays/patch_1_1_2by2_pasturesize10by10mnpy_array.npy
    
    with open('/home/ksa/Desktop/Pasture_Monitoring/Patch_generation_pipeline1April/Day' +str(day_number) +'/npy_arrays/' + patch_name +'npy_array.npy', 'rb') as f:
        x_y_day_heights_coords = np.load(f)
    
    x_y_day_heights_coords_list = list(x_y_day_heights_coords)
    print(x_y_day_heights_coords[0])
    
#    number_of_plants = 1
    
    for grass_plant_index in range(len(x_y_day_heights_coords_list)):
#    for grass_plant_index in range(number_of_plants):
        
        start_time = time.time()
        
        x_val = x_y_day_heights_coords[grass_plant_index][0]
        y_val = x_y_day_heights_coords[grass_plant_index][1]
        z_val = 0.0
        yaw = x_y_day_heights_coords[grass_plant_index][5]
        model_num = (x_y_day_heights_coords[grass_plant_index][6]) +1
        plant_height = (x_y_day_heights_coords[grass_plant_index][9])/1000   #convert from mm to meters
        
        print("x_val:", x_val, "  y_val:", y_val, "  grass_height:", plant_height) 
        add_plant_at_given_pose(x_val, y_val, z_val, yaw, model_num, plant_height)
#        add_plant_at_given_pose(1, 2, 0, 0.1, 3, 0.3)
        
        print("Total number of grass plants: ", len(x_y_day_heights_coords_list))
        print("Currently at index: ", grass_plant_index)   
       
        end_time = time.time() - start_time
        print(f"Processing plant {grass_plant_index} took {end_time} time using serial processing")
#        
    objects = bpy.context.scene.objects
    print(len(objects))
     
    
    
   
    patch_end_time = time.time() - patch_start_time  
    print(f"Processing {patch_name} took {patch_end_time} time using serial processing")

def clear_out_memory():
    for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)
         
def count_number_of_objects_in_scene():
    grass_objects = bpy.context.scene.objects
    return grass_objects

def booltool_boolean_union_two_objects():
    bool_start_time = time.time()
#    
        
        
    grass_objects = count_number_of_objects_in_scene()
    print("Number of grass plants in the beginning: ",  len(bpy.context.scene.objects)) 
    
    object_names = []
    for i in range (0,(len(bpy.context.scene.objects)),1):
        object_names.append(bpy.context.scene.objects[i].name)
        
    for i in range (0,(len(bpy.context.scene.objects)-1),2):
#        grass_objects = bpy.context.scene.objects
        print("Start of loop, i is: ",i)
        print("number of objects in scene: ",len(bpy.context.scene.objects))
        
        
        bpy.ops.object.mode_set(mode = 'OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        
        
        
#        if len(grass_objects)<=5:
#            bpy.ops.object.select_all(action='SELECT')
#            bpy.ops.object.modifier_apply(modifier="Auto Boolean")
#            bpy.ops.object.booltool_auto_union()
#            bpy.ops.object.select_all(action='DESELECT')
            
#        grass_objects = bpy.context.scene.objects
#        grass_objects = count_number_of_objects_in_scene()
            
#        if i >= (len(bpy.context.scene.objects)-1):
#            break
        
        if(bpy.data.objects[object_names[i]]):
            grass1 = bpy.data.objects[object_names[i]]
            grass1.select_set(True)
            bpy.context.view_layer.objects.active = grass1
        
#        grass_objects = count_number_of_objects_in_scene()
        
#        if i >= (len(grass_objects)-1):
#            break
        
        if (bpy.data.objects[object_names[i+1]]):
            bpy.data.objects[object_names[i+1]].select_set(True)
            
        bpy.ops.object.make_single_user(object=True, obdata=True)
        bpy.ops.object.convert(target='MESH')
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.reveal()
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.editmode_toggle()
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.reveal()
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.object.editmode_toggle()
        bpy.ops.object.modifier_apply(modifier="Auto Boolean")
        bpy.ops.object.booltool_auto_union()
        
        
    bool_end_time = (time.time() - bool_start_time)/3600
    print(f"Boolean union of {len(grass_objects)/2} grass pairs took {bool_end_time} hours using serial processing")
    grass_objects = count_number_of_objects_in_scene()
#        clear_out_memory()

 


def boolean_union_two_objects():
    
    clear_out_memory()
    
    bool_start_time = time.time()
    grass_objects = count_number_of_objects_in_scene()
    print("Number of grass plants in the beginning: ",  len(bpy.context.scene.objects)) 
    
    object_names = []
    for i in range (0,(len(bpy.context.scene.objects)),1):
        object_names.append(bpy.context.scene.objects[i].name)
        
    for i in range (0,(len(object_names)),2):
#        clear_out_memory()
        
        print("Start of loop, i is: ",i)
        print("number of objects in scene: ",len(bpy.context.scene.objects))
        
        
        
        bpy.ops.object.select_all(action='DESELECT')
        grass_objects = bpy.context.scene.objects
            
#        if i >= (len(grass_objects)-1):
#            break
        
        grass_1_name = object_names[i]
        grass_2_name = object_names[i+1]
        
        print("grass_1_name is: ", grass_1_name)
        print("grass_2_name is: ", grass_2_name)
#        context = bpy.context
#        scene = context.scene
#        
        grass_1 = grass_objects.get(grass_1_name)
        grass_1.select_set(True)
        bpy.context.view_layer.objects.active = grass_1
        grass_2 = grass_objects.get(grass_2_name)
        grass_2.select_set(True)
        

        if grass_1 and grass_2:
            bool = grass_1.modifiers.new(name='booly', type='BOOLEAN')
            bool.object = grass_2
            bool.double_threshold = 0
            bool.operation = 'UNION'
            bpy.ops.object.modifier_apply(modifier="Boolean")
#            bpy.ops.outliner.item_activate(extend=False, deselect_all=True)

            bpy.ops.object.modifier_apply(modifier="booly")
#            bpy.ops.object.modifier_apply(modifier="booly")

            bpy.ops.object.select_all(action='DESELECT')
            if (len(grass_objects)==1):
                break
            leftover_second_object = grass_2
            leftover_second_object.select_set(True)
            bpy.context.view_layer.objects.active = leftover_second_object
            bpy.ops.object.delete(use_global=False, confirm=False)
            
#            clear_out_memory()

    
        #    bpy.ops.object.modifier_apply({"object": cube},apply_as='DATA', modifier=bool.name)

#            
            
        bool_end_time = (time.time() - bool_start_time)/3600
        print(f"Boolean union of {len(grass_objects)/2} grass pairs took {bool_end_time} hours using serial processing")
        grass_objects = count_number_of_objects_in_scene()
#        clear_out_memory()

        
#        leftover_second_object = grass_2
#        leftover_second_object.select_set(True)
#        bpy.context.view_layer.objects.active = leftover_second_object
#        bpy.ops.object.delete(use_global=False, confirm=False)


        
    
def multiprocessing_func(x_y_patch_points):
    start_time = time.time()
    p = Pool(9)
    print(p.map(spawn_patch, x_y_patch_points)) 
    p.close()
    p.join()
    end_time = time.time() - start_time

    print(f"Processing {len(species_1_number_of_plants)} plants took {end_time} time using multi processing")
    
    
    
if __name__ == '__main__':
    
#    # select evrything
    bpy.ops.object.select_all(action='SELECT')

#    # delete evrything to prevent repetition everytime you run the script
    bpy.ops.object.delete(use_global=False, confirm=False)

    clear_out_memory()


#    # Add prennial ryegrass plant heights
#    rygrass_plant_heights = {1: 0.170227,
#    2: 0.118349,
#    3: 0.179795,
#    4: 0.141542, 
#    5: 0.068646}
    
    #old heights 
#    lowpoly_plant_heights = {1: 0.291262,
#    2: 0.402166,
#    3: 0.37446,
#    4: 0.291314, 
#    5: 0.323194}

    lowpoly_plant_heights = {1: 0.275258,
    2: 0.392174,
    3: 0.372261,
    4: 0.278919, 
    5: 0.301028}
    
    x_offset = {1:-0.04819,
    2: -0.043713,
    3: -0.007913,
    4: -0.004449,
    5: -0.03991}
    
    y_offset = {1: -0.111623,
    2: 0.027988,
    3: 0.046346,
    4: 0.039322,
    5: 0.040621}
    
    
    
    
    
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
#    day_number = 147
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
  
    x_y_patch_points = []
    for x in np.arange(1.0,6.0,1.0):
        for y in np.arange(1.0, 6.0,1.0):
#            x_y_temp = [x,y]
            x_y_temp = [int(x),int(y)]
            x_y_patch_points.append(x_y_temp)
    print(x_y_patch_points)

#    multiprocessing_func(x_y_patch_points)

    x_y_patch_points = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5]]


#    square_patch_line_number = 1
#    patch_numbers = [3, 4 ,5]
#    
###    start_index = (line_number-1)*16
##    combined_patches_5_each_list = [0,20,40,60]
###    for patch_index in range(0,len(x_y_patch_points_list),5):
##    for patch_index in combined_patches_5_each_list:
#        
    patch_combine_start_time = time.time()
#    for patch_index_number in range(patch_index, patch_index +  4, 1):
#        print(patch_index_number)
    
#    patch_indexes_list = [5,6,7,8,9]
#    patch_indexes_list = [20,21,22,23,24]
#    patch_indexes_list = [15,16,17,18,19]
#    patch_indexes_list = [10,11,12,13,14]
#    patch_indexes_list = [0,1,2,3,4]
    patch_indexes_list = [x for x in range (0,25,1)]
    
    for index in patch_indexes_list:
        spawn_patch(x_y_patch_points[index])
#      
            
            
            
        grass_objects = bpy.context.scene.objects
        
        while len(bpy.context.scene.objects) > 1:
            booltool_boolean_union_two_objects()
    #            boolean_union_two_objects()
        
        print("Boolean done")
        
       
    #    
        bpy.ops.object.select_all(action='SELECT')
        
        export_objects = bpy.context.scene.objects
        export_patch_dae = bpy.data.objects[export_objects[0].name]
        export_patch_dae.select_set(True)
        
        patch_name = "2by2m_square_patch_index_" + str(index)
        
        patch_path_for_dae_export = "/home/ksa/Desktop/Pasture_Monitoring/Patch_generation_pipeline1April/Day" +str(day_number) +"/patches/" + patch_name + ".dae" 
        ###bpy.ops.wm.collada_export(filepath=patch_path_for_dae_export,check_existing=True, filter_blender=False, filter_backup=False,filter_image=False, filter_movie=False, filter_python=False,filter_font=False, filter_sound=False, filter_text=False,filter_archive=False, filter_btx=False, filter_collada=True,filter_alembic=False, filter_usd=False, filter_volume=False,filter_folder=True, filter_blenlib=False, filemode=8,display_type='DEFAULT', sort_method='FILE_SORT_ALPHA',prop_bc_export_ui_section='main', apply_modifiers=True,export_mesh_type=0, export_mesh_type_selection='view',export_global_forward_selection='Y', export_global_up_selection='Z',apply_global_orientation=True, selected=True, include_children=True,include_armatures=True, include_shapekeys=True,deform_bones_only=False, include_animations=True,include_all_actions=True, export_animation_type_selection='sample',sampling_rate=1, keep_smooth_curves=False, keep_keyframes=True,keep_flat_curves=False, active_uv_only=False, use_texture_copies=True,triangulate=True, use_object_instantiation=True,use_blender_profile=True, sort_by_name=False,export_object_transformation_type=0,export_object_transformation_type_selection='matrix',export_animation_transformation_type=0,export_animation_transformation_type_selection='matrix',open_sim=False, limit_precision=False, keep_bind_info=True)  
        
        bpy.ops.wm.collada_export(filepath=patch_path_for_dae_export, check_existing=True, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=True, filter_alembic=False, filter_usd=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=8, display_type='DEFAULT', sort_method='FILE_SORT_ALPHA', prop_bc_export_ui_section='main', apply_modifiers=True, export_mesh_type=0, export_mesh_type_selection='view', export_global_forward_selection='Y', export_global_up_selection='Z', apply_global_orientation=False, selected=True, include_children=True, include_armatures=True, include_shapekeys=True, deform_bones_only=False, include_animations=False, include_all_actions=False, export_animation_type_selection='sample', sampling_rate=1, keep_smooth_curves=False, keep_keyframes=True, keep_flat_curves=True, active_uv_only=False, use_texture_copies=True, triangulate=False, use_object_instantiation=True, use_blender_profile=True, sort_by_name=False, export_object_transformation_type=0, export_object_transformation_type_selection='matrix', export_animation_transformation_type=0, export_animation_transformation_type_selection='matrix', open_sim=False, limit_precision=False, keep_bind_info=True)

        print("PATCH EXPORTED")
        #select evrything
        bpy.ops.object.select_all(action='SELECT')

    #    # delete evrything to prevent repetition everytime you run the script
        bpy.ops.object.delete(use_global=False)
                    
        clear_out_memory()
#                
        patch_combine_end_time = (time.time() - patch_combine_start_time)/3600
        print(f"Boolean union of {len(grass_objects)} patches took {patch_combine_end_time} hours using serial processing")

#    #            
 
        
    print("Loop finished")
    
    
#    grass_objects = bpy.context.scene.objects
#    while len(grass_objects) > 1:
#                boolean_union_two_objects()
#    
#    
#    
       
    
    











