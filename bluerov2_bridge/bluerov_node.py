#!/usr/bin/env python

from __future__ import division

import json
import math
import re
import rospy
import sys
import time
from matplotlib import pyplot

from bridge import Bridge

import threading

from Gridy_based import Plannification
from dynamic_window_approach import Evitement
from sensor_msgs.msg import LaserScan
import attitude_control

try:
    from pubs import Pubs
#     from subs import Subs
#     from video import Video
except:
    from bluerov.pubs import Pubs
#     from bluerov.subs import Subs
#     from bluerov.video import Video

# from TrajectoryGenerator import TrajectoryGenerator

# convert opencv image to ros image msg
from cv_bridge import CvBridge

# msgs type
from geometry_msgs.msg import TwistStamped, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool
from std_msgs.msg import String
from std_msgs.msg import UInt16
from sensor_msgs.msg import Joy
# from bluerov_ros_playground.msg import Bar30
# from bluerov_ros_playground.msg import Attitude 
# from bluerov_ros_playground.msg import State 

"""
Generates a quintic polynomial trajectory.
Author: Daniel Ingram (daniel-s-ingram)
"""

import numpy as np

class TrajectoryGenerator():
    def __init__(self, start_pos, des_pos, T, start_vel=[0,0,0], des_vel=[0,0,0], start_acc=[0,0,0], des_acc=[0,0,0]):
        self.start_x = start_pos[0]
        self.start_y = start_pos[1]
        self.start_z = start_pos[2]

        self.des_x = des_pos[0]
        self.des_y = des_pos[1]
        self.des_z = des_pos[2]

        self.start_x_vel = start_vel[0]
        self.start_y_vel = start_vel[1]
        self.start_z_vel = start_vel[2]

        self.des_x_vel = des_vel[0]
        self.des_y_vel = des_vel[1]
        self.des_z_vel = des_vel[2]

        self.start_x_acc = start_acc[0]
        self.start_y_acc = start_acc[1]
        self.start_z_acc = start_acc[2]

        self.des_x_acc = des_acc[0]
        self.des_y_acc = des_acc[1]
        self.des_z_acc = des_acc[2]

        self.T = T

    def solve(self):
        A = np.array(
            [[0, 0, 0, 0, 0, 1],
             [self.T**5, self.T**4, self.T**3, self.T**2, self.T, 1],
             [0, 0, 0, 0, 1, 0],
             [5*self.T**4, 4*self.T**3, 3*self.T**2, 2*self.T, 1, 0],
             [0, 0, 0, 2, 0, 0],
             [20*self.T**3, 12*self.T**2, 6*self.T, 2, 0, 0]
            ])

        b_x = np.array(
            [[self.start_x],
             [self.des_x],
             [self.start_x_vel],
             [self.des_x_vel],
             [self.start_x_acc],
             [self.des_x_acc]
            ])
        
        b_y = np.array(
            [[self.start_y],
             [self.des_y],
             [self.start_y_vel],
             [self.des_y_vel],
             [self.start_y_acc],
             [self.des_y_acc]
            ])

        b_z = np.array(
            [[self.start_z],
             [self.des_z],
             [self.start_z_vel],
             [self.des_z_vel],
             [self.start_z_acc],
             [self.des_z_acc]
            ])

        self.x_c = np.linalg.solve(A, b_x)
        self.y_c = np.linalg.solve(A, b_y)
        self.z_c = np.linalg.solve(A, b_z)


class BlueRov(Bridge):
    def __init__(self, device='udp:192.168.2.1:14550', baudrate=115200):
        """ BlueRov ROS Bridge

        Args:
            device (str, optional): mavproxy device description
            baudrate (int, optional): Serial baudrate
        """
        super(BlueRov, self).__init__(device, baudrate)

        self.evitement = None
        self.mission_point_sent = False
        self.init_evit = False
        self.ok_pose = False
        self.mission_ongoing = False
        self.mission_evit = False
        self.mission_scan = False

        try:
            rospy.init_node('user_node', log_level=rospy.DEBUG)
        except rospy.ROSInterruptException as error:
            print('pubs error with ROS: ', error)
            exit(1)

        self.pub = Pubs()
        # self.sub = Subs()
        self.ROV_name = 'BlueRov2'
        self.model_base_link = '/base_link'

        # # self.video = Video()
        # self.video_bridge = CvBridge()

        self.pub_topics = [
            [
                self._create_position_msg,
                '/local_position',
                PoseStamped,
                1
            ],

            [
                self._create_odometry_msg,
                '/odometry',
                Odometry,
                1
            ]

        ]

        self.mavlink_msg_available = {}

        for _, topic, msg, queue in self.pub_topics:
            self.mavlink_msg_available[topic] = 0
            self._pub_subscribe_topic(topic, msg, queue)



    @staticmethod

    def pub_pass(self):
        pass


    def _pub_subscribe_topic(self, topic, msg, queue_size=1):
        """ Subscribe to a topic using the publisher

        Args:
            topic (str): Topic name
            msg (TYPE): ROS message type
            queue_size (int, optional): Queue size
        """
        self.pub.subscribe_topic(self.ROV_name + topic, msg, queue_size)

    def _create_header(self, msg):
        """ Create ROS message header

        Args:
            msg (ROS message): ROS message with header
        """
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.model_base_link

    # TODO : tester l'utilisation des vélocités comme dans _create_odometry_msg
    def _create_position_msg(self):
        """ Create odometry message from ROV information

        Raises:
            Exception: No data to create the message
        """
        
        # Check if data is available
        if 'LOCAL_POSITION_NED' not in self.get_data():
            raise Exception('no LOCAL_POSITION_NED data')

        if 'ATTITUDE' not in self.get_data():
            raise Exception('no ATTITUDE data')

        #TODO: Create class to deal with BlueRov state
        msg = PoseStamped()

        self._create_header(msg)

        # http://mavlink.org/messages/common#LOCAL_POSITION_NED
        local_position_data = self.get_data()['LOCAL_POSITION_NED']
        xyz_data = [local_position_data[i]  for i in ['x', 'y', 'z']]
        vxyz_data = [local_position_data[i]  for i in ['vx', 'vy', 'z']]
        msg.pose.position.x = xyz_data[0]
        msg.pose.position.y = xyz_data[1]
        msg.pose.position.z = - xyz_data[2]
        # print(xyz_data)

        # https://mavlink.io/en/messages/common.html#ATTITUDE
        attitude_data = self.get_data()['ATTITUDE']
        orientation = [attitude_data[i] for i in ['roll', 'pitch', 'yaw']]

        #https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Euler_Angles_to_Quaternion_Conversion

        cr = math.cos(orientation[0] * 0.5)
        sr = math.sin(orientation[0] * 0.5)
        cp = math.cos(orientation[1] * 0.5)
        sp = math.sin(orientation[1] * 0.5)
        cy = math.cos(orientation[2] * 0.5)
        sy = math.sin(orientation[2] * 0.5)


        # msg.pose.orientation.w = cy * cr * cp + sy * sr * sp
        # msg.pose.orientation.x = cy * sr * cp - sy * cr * sp
        # msg.pose.orientation.y = cy * cr * sp + sy * sr * cp
        # msg.pose.orientation.z = sy * cr * cp - cy * sr * sp
        # on envoie Yaw Pitch Raw à la place du quaternion
        msg.pose.orientation.w = 1
        msg.pose.orientation.x = math.degrees(orientation[0])
        msg.pose.orientation.y = math.degrees(orientation[1])
        msg.pose.orientation.z = math.degrees(orientation[2])
        
        self.pub.set_data('/local_position', msg)


    def _create_odometry_msg(self):
        """ Create odometry message from ROV information

        Raises:
            Exception: No data to create the message
        """

        # Check if data is available
        if 'LOCAL_POSITION_NED' not in self.get_data():
            raise Exception('no LOCAL_POSITION_NED data')

        if 'ATTITUDE' not in self.get_data():
            raise Exception('no ATTITUDE data')

        #TODO: Create class to deal with BlueRov state
        msg = Odometry()

        self._create_header(msg)

        #http://mavlink.org/messages/common#LOCAL_POSITION_NED
        local_position_data = self.get_data()['LOCAL_POSITION_NED']
        xyz_data = [local_position_data[i]  for i in ['x', 'y', 'z']]
        vxyz_data = [local_position_data[i]  for i in ['vx', 'vy', 'z']]
        msg.pose.pose.position.x = xyz_data[0]
        msg.pose.pose.position.y = xyz_data[1]
        msg.pose.pose.position.z = xyz_data[2]
        msg.twist.twist.linear.x = vxyz_data[0]/100
        msg.twist.twist.linear.y = vxyz_data[1]/100
        msg.twist.twist.linear.z = vxyz_data[2]/100

        #http://mavlink.org/messages/common#ATTITUDE
        attitude_data = self.get_data()['ATTITUDE']
        orientation = [attitude_data[i] for i in ['roll', 'pitch', 'yaw']]
        orientation_speed = [attitude_data[i] for i in ['rollspeed', 'pitchspeed', 'yawspeed']]

        #https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Euler_Angles_to_Quaternion_Conversion
        cy = math.cos(orientation[2] * 0.5)
        sy = math.sin(orientation[2] * 0.5)
        cr = math.cos(orientation[0] * 0.5)
        sr = math.sin(orientation[0] * 0.5)
        cp = math.cos(orientation[1] * 0.5)
        sp = math.sin(orientation[1] * 0.5)

        msg.pose.pose.orientation.w = cy * cr * cp + sy * sr * sp
        msg.pose.pose.orientation.x = cy * sr * cp - sy * cr * sp
        msg.pose.pose.orientation.y = cy * cr * sp + sy * sr * cp
        msg.pose.pose.orientation.z = sy * cr * cp - cy * sr * sp
        msg.twist.twist.angular.x = orientation_speed[0]
        msg.twist.twist.angular.y = orientation_speed[1]
        msg.twist.twist.angular.z = orientation_speed[2]

        self.pub.set_data('/odometry', msg)


    def publish(self):
        """ Publish the data in ROS topics
        """
        self.update()
        for sender, topic, _, _ in self.pub_topics:
            try:
                if time.time() - self.mavlink_msg_available[topic] > 1:
                    sender()
            except Exception as e:
                self.mavlink_msg_available[topic] = time.time()
                print(e)





        # position = self.get_data()['LOCAL_POSITION_NED']
        # xyz_data = [position[i]  for i in ['x', 'y', 'z']]
        # x_courant = xyz_data[0]
        # y_courant = xyz_data[1]
        # z_courant = - xyz_data[2]

        ###################################### Fonctions de missions #######################################################

    def do_scan(self, area_center : list):

        ox = [area_center[0] - 5, area_center[0] - 5, area_center[0] + 5, area_center[0] + 5, area_center[0] - 5]
        oy = [area_center[1] + 5, area_center[1] - 5, area_center[1] - 5, area_center[1] + 5, area_center[1] + 5]
        oz = [0]
        resolution = 3
        plannification = Plannification()
        px, py = plannification.planning(ox, oy, resolution)

        self.mission_ongoing = True
        self.mission_scan = True
        desired_position = [0.0, 0.0, -oz[0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        current_pose = self.current_pose
        desired_position[0], desired_position[1] = px[self.mission_scan_point], py[self.mission_scan_point]

        rate = rospy.Rate(50.0)

        while self.mission_ongoing:
            self.get_bluerov_data()
            if abs(current_pose[0] - desired_position[0]) < 0.2 and abs(current_pose[1] - desired_position[1]) < 0.2:
                # print("Arrivé au point")
                if self.mission_scan_point == len(px) - 1:
                    self.ok_pose = True
                    print('Mission de scan terminée !')
                    self.mission_scan_point = 0
                    self.mission_scan = False
                    self.mission_ongoing = False
                    self.mission_point_sent = False
                else :
                    self.mission_scan_point += 1
                    desired_position[0], desired_position[1] = px[self.mission_scan_point], py[self.mission_scan_point]
                self.mission_point_sent = False

            if self.mission_point_sent == False:
                time.sleep(0.05)
                # self.ok_pose = False
                self.set_position_target_local_ned(desired_position)
                time.sleep(0.05)
                self.mission_point_sent = True
            
            self.publish()
            rate.sleep()


        
    def do_evit(self, x_init, goal):
        self.mission_ongoing = True
        self.mission_evit = True
        z_mission = x_init[2]
        rate = rospy.Rate(50.0)

        while self.mission_ongoing:
            self.get_bluerov_data()
            if self.init_evit == False:

                if self.mission_point_sent == False:
                    time.sleep(0.05)
                    self.ok_pose = False
                    initial_position = [x_init[0], x_init[1], -z_mission, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.88, 0.0]
                    self.set_position_target_local_ned(initial_position)
                    time.sleep(0.05)
                    self.mission_point_sent = True

                if abs(self.current_pose[0] - x_init[0]) < 0.2 and abs(self.current_pose[1] - x_init[1]) < 0.2:
                    self.init_evit = True
                    print("je suis au point initial")
                    self.x = np.array([x_init[0], x_init[1], self.current_attitude[2], 0.0, 0.0])
                    self.mission_point_sent = False
                    self.evitement = Evitement(goal)



            # on observe les obstacles environnants
            # self.ob = self.get_lidar_obstacle(90)
            # self.ob = self.get_velodyne_obstacle()

            if self.init_evit == True:
                # si on a commencé la mission d'évitement 
                # print("init_evit " + str(self.init_evit))
                if abs(self.current_pose[0] - self.x[0]) < 0.02 and abs(self.current_pose[1] - self.x[1]) < 0.02:
                    # print("avant do evit planning")
                    obstacles = self.get_collider_obstacles()
                    self.ob = np.array([[]])
                    compteur = 0
                    while obstacles.size == 0:
                        obstacles = self.get_collider_obstacles()
                        compteur+=1
                        if obstacles.size > 0 or compteur > 150000:
                            break
                    self.ob = obstacles
                    # print(compteur)
                    # print("Final ", self.ob)

                    current_pose = np.array([self.current_pose[0], self.current_pose[1], self.x[2], self.x[3], self.x[4]])
                    # print("entrée = " + str(self.x))
                    self.x = self.evitement.planning(self.ob, self.x)
                    # self.x[2] = -self.x[2]
                    self.mission_point_sent = False
                    # print(self.x[0:2])
                    # print("self.current_pose = " + str(self.current_pose))
                    # return self.x, u, trajectory

                if self.mission_point_sent == False:
                    time.sleep(0.05)
                    self.ok_pose = False
                    desired_position = [self.x[0], self.x[1], -z_mission, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.x[2], self.x[4]]
                    self.set_position_target_local_ned(desired_position)
                    time.sleep(0.05)
                    self.mission_point_sent = True

                if self.evitement.goal_reached:
                    self.mission_ongoing = False
                    self.mission_evit = False
                    self.init_evit = False
                    self.mission_point_sent = False
                    print("Mission d'évitement terminée !")
                    self.evitement.goal_reached = False
                
                self.publish()
                rate.sleep()
    ################################## Fin fonction de missions #####################################################



if __name__ == '__main__':
    try:
        rospy.init_node('user_node', log_level=rospy.DEBUG)
    except rospy.ROSInterruptException as error:
        print('pubs error with ROS: ', error)
        exit(1)
    bluerov = BlueRov(device='udp:localhost:14551')

    # ox = [0.0, 10.0, 10.0, 0.0, 0.0]
    # oy = [0.0, 0.0, 30.0, 30.0, 0.0]
    # oz = [5]
    # resolution = 2

    change_mode = 0

    ox = [-30, -30, -10, -10, -30]
    oy = [-10, 10, 10, -10, -10]
    oz = [-7]
    resolution = 3

    plannification = Plannification()
    goal_evit_1 =  np.array([-19.96 , -10.18])
    goal_evit_2 = np.array([10.24 , 0.02])
    # evitement = Evitement(goal_evit_1)

    goal_scan =  np.array([-9.77 , 0.02])

    x_init_evit_1 = [-19.96, -29.82, -7]
    x_init_evit_2 = [-9.77, 0.02, -7]

    px, py = plannification.planning(ox, oy, resolution)
    px.append(goal_scan[0])
    py.append(goal_scan[1])
    rate = rospy.Rate(50.0)

    yaw_path=150
    yaw_send = False

    # position_desired = [0, 0, 0, 0.5, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]
    counter_mission = 1
    position_desired = [-100.0, -100.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    while not rospy.is_shutdown():

        bluerov.get_bluerov_data()

        # time.sleep(0.05)
        # evitement.goal_reached = True

        # if evitement.goal_reached == False:
        #     bluerov.do_evit(x_init, goal)

        # if yaw_send == False:
        #     attitude_control.set_target_attitude(bluerov.conn,0, 0, yaw_path)

        # elif bluerov.ok_pose == False:
        #     if change_mode == 0:
        #         bluerov.mission_sent_point = False
                # change_mode += 1

        if counter_mission == 2:
            bluerov.do_evit(x_init_evit_1, goal_evit_1)
            if bluerov.mission_ongoing == False and bluerov.mission_evit == False:
                counter_mission +=1
                print(counter_mission)
        
        elif counter_mission == 1:
            bluerov.do_scan(px, py, oz)
            if bluerov.mission_ongoing == False and bluerov.mission_scan == False:
                counter_mission +=1
                print(counter_mission)

        elif counter_mission == 3:
            bluerov.do_evit(x_init_evit_2, goal_evit_2)
            if bluerov.mission_ongoing == False and bluerov.mission_evit == False:
                counter_mission +=1
                print(counter_mission)

        print("mission_ongoing ", bluerov.mission_ongoing,"| mission_evit ", bluerov.mission_evit, "| mission_scan ", bluerov.mission_scan, "| counter ", counter_mission)
        # yaw_path = (yaw_path+10)%360 

        # bluerov.set_position_target_local_ned(position_desired)
        # counter+=1

        # if counter%10 == 0:
        #     counter = 0
        #     position_desired[0] += 5
        #     position_desired[1] -= 5

#############################TEST MOUVEMENT AVEC VITESSE ET ORIENTATION###########################

        # bluerov.set_speed_and_attitude_target(position_desired)
        # counter+=1

        # if counter%10 == 0:
        #     counter = 0
        #     position_desired[9] += math.pi/4
        #     print(position_desired[9])

#################################################################################################


        # time.sleep(0.05)
        # print(bluerov.get_velodyne_obstacle())
        # print(bluerov.velodyne_data)
        # bluerov.get_collider_obstacles()

        bluerov.publish()

        # bluerov.do_scan([0, 0, 0], [10, 10, 0], 10)
        rate.sleep()