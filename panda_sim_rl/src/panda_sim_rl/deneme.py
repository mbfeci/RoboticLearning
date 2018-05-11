#!/usr/bin/env python
from panda_manipulation.manipulator import Manipulator
import rospy
from sensor_msgs.msg import JointState
import time
import operator
from panda_manipulation.MyObject import MyObject, Sphere, Box, Cylinder, Duck, Bunny 
from tf import TransformListener
from geometry_msgs.msg import Pose, PoseStamped
from random import randint, uniform

def getGripperPose2():
    pos = manipulator.arm.get_current_pose()
    return pos #np.array([pos.position.x, pos.position.y, pos.position.z])

def isInMotion():
    for k in range(7):
        val1 = my_joint_values[k]
        val2 = his_joint_values[k]
        if val1-val2>1.0e-1 or val1-val2<-1.0e-1:
            return True
    return False

def waitTillReady2():
    for k in range(300):
        if not isInMotion(): return
        rospy.sleep(0.01)
    print "SOMETHING IS WRONG, STOPPED WAITING AFTER 3 SECS!"

def jsCB(msg):
    temp_dict = dict(zip(msg.name, msg.position))
    his_joint_values = [temp_dict[x] for x in joint_names]


rospy.init_node("deneme_node")
manipulator = Manipulator()

joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3',
  'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
one_joint_execution_duration = 0.01
execution_time = [(x+1)*one_joint_execution_duration for x in range(7)]
sub = rospy.Subscriber('/joint_states', JointState, jsCB)
his_joint_values = [-2.5450077537365753e-08, 2.544688075917041e-08, 2.3532384855862176e-05, 7.785478886557229e-05, -2.0168301624323703e-06, -8.301148533007563e-07, -2.2624831839124226e-06]
tf = TransformListener()

box = Box()
pos = Pose()
pos.position.x = uniform(0.15,0.4)
pos.position.y = uniform(-0.3, 0)
pos.position.z = 0.8
box.set_position(pos) 
box.place_on_table()
rospy.sleep(1)
pos = box.get_position()

stamped_pos = PoseStamped()
pos.position.z += 0.2
pos.orientation = getGripperPose2().orientation
stamped_pos.pose = pos
stamped_pos.header.frame_id = '/world'
pos = tf.transformPose("/panda_link0", stamped_pos)
my_joint_values = manipulator.arm.get_IK(pos.pose)
manipulator.arm.execute_trajectory([my_joint_values], execution_time)
waitTillReady2()


pos.pose.position.z -= 0.08
my_joint_values = manipulator.arm.get_IK(pos.pose)
manipulator.arm.execute_trajectory([my_joint_values], execution_time)
waitTillReady2()



