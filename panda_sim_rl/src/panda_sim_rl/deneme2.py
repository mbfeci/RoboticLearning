#!/usr/bin/env python
from panda_manipulation.manipulator import Manipulator
import rospy
from sensor_msgs.msg import JointState
import time
import operator
from panda_manipulation.MyObject import MyObject, Sphere, Box, Cylinder, Duck, Bunny 
from tf import TransformListener
from geometry_msgs.msg import Pose, PoseStamped
from math import pi

rospy.init_node("deneme2_node")
manipulator = Manipulator()

one_joint_execution_duration = 0.01
execution_time = [(x+1)*one_joint_execution_duration for x in range(2)]

manipulator.hand.execute_trajectory([2,2],[0.01])