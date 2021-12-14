# Pasture Generation and Point Cloud Processing

<!-- [I'm an inline-style link](https://www.google.com) -->

First, downlaod and catkin_make with the ROS packages at:  
1. [Hector Quadrotor ROS package](https://github.com/hsd1121/hector_quadrotor_tutorial/tree/melodic)     
2. [Point Cloud Processing ROS package](https://github.com/hsd1121/PointCloudProcessing/tree/melodic)  

Please use the melodic branch for use with Ubuntu 18. The following instructions have been tested with Ubuntu 18 + ROS Melodic:
# Pasture Generation

1. Run the make_heights_and_xy_coords_npy_array.py file to generate the npy array for the entire pasture.  

2. Run the sort_lists_into_patches.py to sort that npy array into patches of specified dimensions. (2 by 2meter by default in script)

3. Use the npy arrays for patches in the blender script.

4. After generating the collada files for each patch, use spawn_patches_in_world_dae.py to combine all patches in a single .world file. This .world file is copied in the ~/catkin_ws/src/hector_quadrotor_tutorial/hector_gazebo/hector_gazebo_worlds/worlds folder. 


---

Use the following commands to run the gazebo simulation and save the point clouds using the hector quadcopter in Ubuntu 18:

# Point Cloud Generation
```zsh

# in every new terminal do
source ~/catkin_ws/devel/setup.zsh #or setup.bash if you're using bash
#recommended to add this line in your .zshrc file

# in first terminal (relaunch and close everytime when restarting the scripts):
roscore
# in second terminal
## it takes about 28GB RAM (gzserver) to load the gazebo simulation, turn the GUI to false so that it loads up quicker in about 35 minutes.
## RVIZ starts as soon as the simulation is loaded. The RAM usage of RViz will keep increasing almost linearly as the number of points collected by the LiDAR goes up. It goes up till the Decay time set in the RViz simulation.
roslaunch hector_quadrotor_demo pasture_and_quadcopter.launch
# in third terminal
rosrun point_cloud_processing transform_sim_and_save 

# to concatenate clouds, create ~/Data/Pointcloud directory, then:
rosrun point_cloud_processing concatenate_cloud <subdirectory name> <number of files>
rosrun point_cloud_processing concatenate_cloud /home/ksa/Data/PointCloud/9-18-2020/unfiltered/ 236

#controlling the quadcopter using teleop
rosservice call /enable_motors "enable: true"
rosrun teleop_twist_keyboard teleop_twist_keyboard.py 

# using autonomous navigation script
rosrun hector_quadrotor_navigation quadrotor_navigation_spawn_directly.py


#viewing the cloud
rosrun point_cloud_processing view_single_cloud /home/ksa/Data/PointCloud/concatenated_cloud.pcd

```

# Processing point clouds
Please replace ```/home/ksa``` by your own ```/home/username``` in all of the following:
```
#to use crop box filter on the point cloud for plot + 2m on each side to get plot_all
rosrun point_cloud_processing single_crop_box_filter /home/ksa/Desktop/Pasture_Monitoring/Patch_generation_pipeline1April/Python_scripts/plot_all_params.txt <raw_concatenated_pcd>

# for generating max_heights_point_cloud and csv files of points with height greater than mean+4*std_dev
rosrun point_cloud_processing plot_heights_std_dev_filter_with_max_heights /home/ksa/Desktop/Pasture_Monitoring/Patch_generation_pipeline1April/Python_scripts/plot_only_params.txt  <plot_till_2m_padding.pcd>

- All files are generated at the path "/home/ksa/Desktop/Pasture_Monitoring/pcd_to_heights/test_pcd/"

#plot_all path
/home/ksa/Desktop/Pasture_Monitoring/Patch_generation_pipeline1April/Python_scripts/plot_all_params.txt

#plot_only path
/home/ksa/Desktop/Pasture_Monitoring/Patch_generation_pipeline1April/Python_scripts/plot_only_params.txt


#for plot only so +0.2m on each side
rosrun point_cloud_processing single_crop_box_filter /home/ksa/Desktop/Pasture_Monitoring/pcd_to_heights/plot_only_params.txt /home/ksa/Desktop/Pasture_Monitoring/pcd_to_heights/pasture.pcd


/home/ksa/Desktop/Pasture_Monitoring/pcd_to_heights/plot_only_params.txt /home/ksa/Desktop/Pasture_Monitoring/pcd_to_heights/day91_april1_2009/april_1_day91_raw_concatenated_cloud_crop_box_filtered.pcd

rosrun point_cloud_processing plot_heights_std_dev_filter_with_max_heights <plot_only_params> <plot_all_crop_box_filtered_point_cloud>


rosrun point_cloud_processing turfgrass_heights /home/ksa/Desktop/Pasture_Monitoring/pcd_to_heights/pasture_10.2m_crop_box_filtered.pcd /home/ksa/Desktop/Pasture_Monitoring/pcd_to_heights/pasture_12m_crop_box_filtered.pcd

rosrun point_cloud_processing plot_heights plot_only_params.txt same_height_concatenated_cloud_crop_box_filtered_all.pcd


```

# Autonomous Navigation script
Install required packages:
```
pip install rospkg
pip3 install rospkg
sudo apt-get install ros-melodic-teleop-twist-keyboard
sudo apt-get install ros-melodic-teleop-twist-joy

```
Launching the script:
```
roscore
roslaunch hector_quadrotor_demo pasture_and_quadcopter.launch
rosrun hector_quadrotor_navigation quadrotor_navigation.py
```

# Recording and playing rosbag files
You can record the messages being published on all topics and view them later in Rviz(ROS Visualization) using:
```
#to record a rosbag file
rosbag record -a

#to play the rosbag file and see it in rviz
#in first terminal
rosparam set /use_sim_time "true"
#toggle spacebar to play/pause the rosbag file
rosbag play --pause -l --clock 2020-09-23-21-46-48.bag
#in another terminal
rviz rviz
```
