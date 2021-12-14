#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist, Point, Quaternion
import tf
from math import radians, copysign, sqrt, pow, pi, atan2
from tf.transformations import euler_from_quaternion
import numpy as np
import subprocess
import os
import time
import math 
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from tf.transformations import quaternion_from_euler


# for i in range(100000):
#     #to keep it stationary
#     yaw_radian_angle = math.radians(0)

#     # input to q i in the form of roll, pitch, yaw
#     q = quaternion_from_euler(0, 0, 0)
#     # state_msg is an object
#     state_msg = ModelState()
#     state_msg.model_name = 'quadrotor'
#     state_msg.pose.position.x = 0.5
#     state_msg.pose.position.y = 0
#     state_msg.pose.position.z = 0.5


#     state_msg.pose.orientation.x = q[0]
#     state_msg.pose.orientation.y = q[1]
#     state_msg.pose.orientation.z = q[2]
#     state_msg.pose.orientation.w = q[3]

#     set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
#     resp = set_state(state_msg)
#     print(resp)
#     time.sleep(0.05)



for x in np.arange(0,10,0.3):
        # print(x)
        for y in np.arange(0,10,0.3):

            # for yaw_angle_degree in np.arange(0, 180, 30):
            for yaw_angle_degree in np.arange(0, 1, 1):

                yaw_radian_angle = math.radians(yaw_angle_degree)

                # input to q i in the form of roll, pitch, yaw
                q = quaternion_from_euler(0, 0, yaw_radian_angle)
                # state_msg is an object
                state_msg = ModelState()
                state_msg.model_name = 'quadrotor'
                state_msg.pose.position.x = x
                state_msg.pose.position.y = y
                state_msg.pose.position.z = 2
                

                state_msg.pose.orientation.x = q[0]
                state_msg.pose.orientation.y = q[1]
                state_msg.pose.orientation.z = q[2]
                state_msg.pose.orientation.w = q[3]

                set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                resp = set_state(state_msg)
                print(resp)

                time.sleep(0.2)



