#!/usr/bin/env python
import sys
import rospy
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Header
from panda_manipulation.MyObject import MyObject, Sphere, Box, Cylinder, Duck, Bunny 
from tf import TransformListener
from random import randint, uniform
import math
from actionlib_msgs.msg import GoalStatusArray
import operator
from panda_manipulation.manipulator import Manipulator
from actionlib_msgs.msg import GoalStatus
from gym import spaces


class RobotGame():
    def __init__(self):
        rospy.init_node("robotGame")

        self.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3',
  'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        self.his_joint_values = [-2.5450077537365753e-08, 2.544688075917041e-08, 2.3532384855862176e-05, 7.785478886557229e-05, -2.0168301624323703e-06, -8.301148533007563e-07, -2.2624831839124226e-06]
        self.my_joint_values = self.his_joint_values

        self.arm_joint_upper_limits = np.array([2.8973, 1.7628, 2.8973, 0.0175, 2.8973, 3.7525, 2.8973])
        self.arm_joint_bottom_limits = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.arm_joint_diff = self.arm_joint_upper_limits - self.arm_joint_bottom_limits

        #OPTION 1:

        self.observation_bound = np.array([100]*13)
        self.action_bound2 = np.array([0.1]*7)        
        self.action_space = spaces.Box(low=-self.action_bound2, high=self.action_bound2)
        self.observation_space = spaces.Box(low=-self.observation_bound, high=self.observation_bound )

        #OPTION 2:

        # self.observation_bound = np.array([5]*6)
        # self.action_bound = np.array([0.05]*3)
        # self.action_space = spaces.Box(low=-self.action_bound, high=self.action_bound)
        # self.observation_space = spaces.Box(low=-self.observation_bound, high=self.observation_bound )

        self.action_bound = [(-self.action_bound2).tolist(), self.action_bound2.tolist()]
        self.action_dim = self.action_space.shape[0]
        self.state_dim = self.observation_space.shape[0]
        self.gamma = 1
        self.one_joint_execution_duration = 0.01
        self.execution_time = [(x+1)*self.one_joint_execution_duration for x in range(7)]

        self.dist_threshold = 0.1
        self.vector_from_center = np.array([0, 0, 0.05])
        self.object_moved_threshold = 0.05

        self.tf = TransformListener()
        self.sub = rospy.Subscriber('/joint_states', JointState, self.jsCB)
        self.box = Box()
        self.manipulator = Manipulator()
        #self.reset()
        
    def jsCB(self,msg):
        temp_dict = dict(zip(msg.name, msg.position))
        self.his_joint_values = [temp_dict[x] for x in self.joint_names]

    def isInMotion(self):
        for k in range(7):
            val1 = self.my_joint_values[k]
            val2 = self.his_joint_values[k]
            if val1-val2>1.0e-1 or val1-val2<-1.0e-1:
                return True
        return False

    def getGripperPose3(self, reference):
        self.tf.waitForTransform(reference,"/panda_leftfinger",rospy.Time(),rospy.Duration(10))
        t = self.tf.getLatestCommonTime(reference, "/panda_leftfinger") 
        position, quaternion = self.tf.lookupTransform(reference,"/panda_leftfinger",t)
        return np.array([position[0],position[1],position[2]])

    def getGripperPose2(self):
        pos = self.manipulator.arm.get_current_pose()
        return pos #np.array([pos.position.x, pos.position.y, pos.position.z])

    def getGripperPose(self, reference):
        self.tf.waitForTransform(reference,"/panda_hand",rospy.Time(),rospy.Duration(10))
        t = self.tf.getLatestCommonTime(reference, "/panda_hand") 
        position, quaternion = self.tf.lookupTransform(reference,"/panda_hand",t)
        return np.array([position[0],position[1],position[2]])

    def getRequiredJoints(self, delta):
        pos = self.manipulator.arm.get_current_pose()
        #gripperPos = self.getGripperPose('/panda_link0')
        pos.position.x += delta[0]
        pos.position.y += delta[1]
        pos.position.z += delta[2]
        #pos.orientation = None
        print "IK SOLUTION OF: ", pos
        return self.manipulator.arm.get_IK(pos)

    def getDist(self):
        currentPos = self.getGripperPose('/world')
        pos = self.box.get_position()
        self.destPos = np.array([pos.position.x, pos.position.y, pos.position.z])#0.05??
        
        #TODO: CHECK!!!
        #print "box pos (GAZEBO INTERFACE): ", self.destPos
        # self.tf.waitForTransform('/world',"box",rospy.Time(),rospy.Duration(10))
        # t = self.tf.getLatestCommonTime('/world', "box") 
        # position, quaternion = self.tf.lookupTransform('/world',"box",t)
        # print "box pos (TF): ", position
        
        return np.linalg.norm(currentPos-self.destPos-self.vector_from_center)

    def reset(self):
        self.my_joint_values = self.random_joint_initialize()
        self.manipulator.arm.execute_trajectory([self.my_joint_values], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        self.waitTillReady2()
        pos = Pose()
        pos.position.x = uniform(0.15,0.4)
        pos.position.y = uniform(-0.3, 0)
        pos.position.z = 0.8
        self.box.set_position(pos) 
        self.box.place_on_table()
        rospy.sleep(1)
        pos = self.box.get_position()
        self.startDestPos = np.array([pos.position.x, pos.position.y, pos.position.z])
        self.destPos = np.array([pos.position.x, pos.position.y, pos.position.z])
        #OPTION 1:
        return np.concatenate((self.my_joint_values, self.getGripperPose('world'), self.destPos))

        #OPTION 2:
        #return np.concatenate((self.getGripperPose2(), self.destPos))

    def random_joint_initialize(self):
        return np.random.uniform(self.arm_joint_bottom_limits,self.arm_joint_upper_limits, 7).tolist()

    def waitTillReady1(self):
        state = self.manipulator.arm.client.get_state()
        print state
        while not state==GoalStatus.SUCCEEDED:
            rospy.sleep(0.01)
            state = self.manipulator.arm.client.get_state()
            if state==GoalStatus.ABORTED:
                self.manipulator.arm.execute_trajectory([self.my_joint_values], self.execution_time)
                rospy.sleep(0.01)
                state = self.manipulator.arm.client.get_state()
            print state

    def waitTillReady2(self):
        for k in range(300):
            if not self.isInMotion(): return
            rospy.sleep(0.01)
        print "SOMETHING IS WRONG, STOPPED WAITING AFTER 3 SECS!"
        # while self.isInMotion():
        #     rospy.sleep(0.01)

    def waitTillReady3(self):
        while not self.manipulator.arm.client.wait_for_result():
            rospy.sleep(0.01)
            print self.manipulator.arm.client.wait_for_result()

    def step(self,delta):
        done = False
        prevDist = self.getDist()

        #OPTION 1:
        joint_values = np.clip(self.my_joint_values+delta, self.arm_joint_bottom_limits, self.arm_joint_upper_limits)
        
        #OPTION 2:
        # joint_values = self.getRequiredJoints(delta)
        # print "IS: ", joint_values

        #OPTION 3:
        #Just go to the object XD
        # stamped_pos = PoseStamped()
        # pos = self.box.get_position()
        # pos.position.z += 0.2
        # pos.orientation = self.getGripperPose2().orientation
        # stamped_pos.pose = pos
        # stamped_pos.header.frame_id = '/world'
        # print stamped_pos
        # pos = self.tf.transformPose("/panda_link0", stamped_pos)
        # print pos

        #joint_values = self.manipulator.arm.get_IK(pos.pose)
        #print joint_values

        # if joint_values is None:
        #     print "Gripper position is out of bounds, penalty -10"
        #     return np.concatenate((self.getGripperPose('/world'), self.destPos)), -10, done, None

        self.my_joint_values = joint_values
        self.manipulator.arm.execute_trajectory([self.my_joint_values], self.execution_time)

        #OPTION 3 - Continued
        #Burayi sonra sil
        # self.waitTillReady2()
        # pos.pose.position.z -= 0.15
        # joint_values = self.manipulator.arm.get_IK(pos.pose)
        # self.my_joint_values = joint_values
        # self.manipulator.arm.execute_trajectory([self.my_joint_values], self.execution_time)
        
        self.waitTillReady2()
                
        curDist = self.getDist()
        reward = -(curDist*curDist + 0.5*(prevDist-curDist)*(prevDist-curDist))

        #reward = prevDist - curDist
        #print "CURRENT DISTANCE: ", curDist
        #print "Dest pos: ", self.destPos
        info = False
        if curDist <= self.dist_threshold:
            reward +=100
            done = True
            info = True
        elif self.destPos[2] < 0.3:
            print "OBJECT FELL OFF THE GROUND! PENALTY: -10"
            reward -= 10
            done = True
            info = False
        elif np.linalg.norm(self.destPos-self.startDestPos)>=self.object_moved_threshold:
            #print "OBJECT IS MOVED! PENALTY: -20"
            reward -= 10 #TODO: give penalty instead of reward ?
            done = True
            info = False
        

        #OPTION 1:
        return np.concatenate((self.my_joint_values, self.getGripperPose('world'), self.destPos)), reward, done, info
        
        #OOTION 2:
        #return np.concatenate((self.getGripperPose2(), self.destPos)), reward, done, None

    def done(self):
        self.sub.unregister()
        rospy.signal_shutdown("done")

if __name__ == "__main__":
            r = RobotGame()
            print r.getCurrentJointValues()
            print r.getCurrentPose()
            r.reset()
            print r.getCurrentJointValues()
            print r.getCurrentPose()

