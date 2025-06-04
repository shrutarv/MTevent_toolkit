#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:27:36 2024

@author: Shrutarv Awasthi
"""

import rospy
from cv_bridge import CvBridge
import json
from std_msgs.msg import Float32MultiArray
import cv2
import os
import rosbag
from dataClass import DataClass



objects_list = ['pallet', 'small_klt', 'big_klt', 'blue_klt', 'shogun_box', 'kronen_bier_crate', 'brinkhoff_bier_crate',
                'zivid_cardboard_box', 'dell_carboard_box', 'ciatronic_carboard_box', 'human', ' hupfwagon', 'mobile_robot']
obj = ['human_LH','human_RH', 'human_LL', 'human_RL', 'human_head', 'human_waist']
'''
The human has markers on right hand (RH), left hand (LH), left leg (LL), right leg (RL), head and waist.
These markers are tracked under the topic /vicon_markers in the rosbag file. This code extract the data for these markers
and creates a 3D bounding box for the human in the scene and saves the 3D BBox data.
'''
 # list objects other than human. such as table, hupwagen(forklift), etc.
count = 0
vicon_data = {}
root = '/media/eventcamera/event_data/dataset_25_march_zft/' # CHANGE: Directory where the folders for all the scenes are present and also the scene_data.json file.

def get_rotation_hupwagen(timestamp):

    if str(timestamp) not in hupwagen_data.keys():
        timestamp = min(hupwagen_data.keys(), key=lambda x: abs(int(x) - int(str(timestamp))))
    return hupwagen_data[str(timestamp)]['rotation']


with open(root + "/scene_data.json", "r") as file:
    scenes_data = json.load(file)

# Loop through all the scenes in the scene_data.json file
for scene, o in scenes_data.items():
    path = root + scene + '/'
    with open(path + 'vicon_data/hupwagen.json', 'r') as json_file:
        hupwagen_data = json.load(json_file)
        # if hupwagen_data is empty then set hupwagen flag to False
        if not hupwagen_data:
            hupwagen = False
        else:
            hupwagen = True
    for object in obj:
        # Loops through al the human objects
        print('Extracting data for object: ', object, ' with scene: ', scene)
        object_name = object

        bag = rosbag.Bag(root + scene + '/all.bag')

        vicon_object = '/vicon/' + object + '/' + object
        vicon_human = '/vicon/markers'
        time_LH = []
        time_RH = []
        time_LL = []
        time_RL = []
        time_moroKopf = []
        size = 400000
        vicon_data = {}

        if not os.path.exists(path + '/vicon_data'):
            os.makedirs(path + '/vicon_data')
        vicon_data = {}
        print('Extracting data for object: ', object, ' with scene: ',scene)
        for top, msg, tim in bag.read_messages(vicon_object):
            t = msg.header.stamp
            translation = [
                msg.transform.translation.x,
                msg.transform.translation.y,
                msg.transform.translation.z]
            rotation = [
                msg.transform.rotation.x,
                msg.transform.rotation.y,
                msg.transform.rotation.z,
                msg.transform.rotation.w]
            # save t, translation and rotation to a json file
            vicon_data[str(t)] = {'translation': translation, 'rotation': rotation, 'timestamp': str(t)}
            # vicon_data[str(t)] = {'translation': translation, 'rotation': rotation}

        with open(path + 'vicon_data/' + object + '.json', 'w') as json_file:
            json.dump(vicon_data, json_file, indent=2)
        print('saved object data')
        if object == 'human_LH':
            human_LH = DataClass(vicon_data)
        elif object == 'human_RH':
            human_RH = DataClass(vicon_data)
        elif object == 'human_LL':
            human_LL = DataClass(vicon_data)
        elif object == 'human_RL':
            human_RL = DataClass(vicon_data)
        elif object == 'human_head':
            human_head = DataClass(vicon_data)
        elif object == 'human_waist':
            human_waist = DataClass(vicon_data)

    def get_rot(obj_name,t):
        if obj_name == 'human_LH':
            rot = human_LH.get_rotation(t)
            return rot
        elif obj_name == 'human_RH':
            rot = human_RH.get_rotation(t)
            return rot
        elif obj_name == 'human_LL':
            rot = human_LL.get_rotation(t)
            return rot
        elif obj_name == 'human_RL':
            rot = human_RL.get_rotation(t)
            return rot
        elif obj_name == 'human_head':
            rot = human_head.get_rotation(t)
            return rot
        elif obj_name == 'human_waist':
            rot = human_waist.get_rotation(t)
            return rot


    def check_value(data, prev_t, present_t):
        # compare all the x,y and z values at current time with the previous time. If the mod of difference is greater than 2 then keep the previous value
        # else keep the current value
        data[present_t]['min_x'] = data[prev_t]['min_x'] if abs(
            data[present_t]['min_x'] - data[prev_t]['min_x']) > 2 else data[present_t]['min_x']
        data[present_t]['min_y'] = data[prev_t]['min_y'] if abs(
            data[present_t]['min_y'] - data[prev_t]['min_y']) > 2 else data[present_t]['min_y']
        data[present_t]['min_z'] = data[prev_t]['min_z'] if abs(
            data[present_t]['min_z'] - data[prev_t]['min_z']) > 2 else data[present_t]['min_z']
        data[present_t]['max_x'] = data[prev_t]['max_x'] if abs(
            data[present_t]['max_x'] - data[prev_t]['max_x']) > 2 else data[present_t]['max_x']
        data[present_t]['max_y'] = data[prev_t]['max_y'] if abs(
            data[present_t]['max_y'] - data[prev_t]['max_y']) > 2 else data[present_t]['max_y']
        data[present_t]['max_z'] = data[prev_t]['max_z'] if abs(
            data[present_t]['max_z'] - data[prev_t]['max_z']) > 2 else data[present_t]['max_z']

        return data


    # Initialise variables. The below variables are useful for BBox calculation.
    max_x = -100000
    max_y = -100000
    max_z = -100000
    min_x = 100000
    min_y = 100000
    min_z = 100000
    save_flag = True
    vicon_object = '/vicon/markers'

    vicon_data = {}
    first_flag = 0
    save_index = []
    count = 0
    entry_flag = True
    bag = rosbag.Bag(root + scene + '/all.bag') # all-bag contains all the vicon data for human
    # define an empty string
    previous_t = ''

    for top, msg, tim in bag.read_messages(vicon_object):

        entry_flag = True
        count = count + 1
        t = msg.header.stamp
        if first_flag == 0:
            # msg consists of all the markers in the scene. We need to find the markers which are not occluded and have the word human in them
            for j in range(int(len(msg.markers))):
                # if the marker_name contains human in the name string then save that value in a list
                if msg.markers[j].marker_name.find('human') != -1:
                    save_index.append(j)
            previous_t = str(t)
            first_flag = 1

        for i in save_index:
            if  msg.markers[i].occluded == False:
                if entry_flag:
                    max_x = -100000
                    max_y = -100000
                    max_z = -100000
                    min_x = 100000
                    min_y = 100000
                    min_z = 100000
                    entry_flag = False
                save_flag = True
                # find the max and min x, y, z values of the markers among all save_index
                if msg.markers[i].translation.x > max_x:
                    max_x = msg.markers[i].translation.x
                    # truncate the last character of the marker_name to get the human part
                    rot = get_rot('human_head', t)          # We assume the rotation of the human head is the same as the rotation of the human body
                if msg.markers[i].translation.y > max_y:
                    max_y = msg.markers[i].translation.y
                    rot = get_rot('human_head', t)
                if msg.markers[i].translation.z > max_z:
                    max_z = msg.markers[i].translation.z + 100
                    rot = get_rot('human_head', t)
                if msg.markers[i].translation.x < min_x:
                    min_x = msg.markers[i].translation.x
                    rot = get_rot('human_head', t)
                if msg.markers[i].translation.y < min_y:
                    min_y = msg.markers[i].translation.y
                    rot = get_rot('human_head', t)
                if msg.markers[i].translation.z < min_z:
                    min_z = msg.markers[i].translation.z
                    rot = get_rot('human_head', t)
            else:
                print('msg' + str(count) + ' _ ' + str(i), ' is occluded')
                rot = get_rot('human_head', t)


        if save_flag:
            # save the min and max values of x, y, z and rotation to a json file
            vicon_data[str(t)] = {'min_x': min_x, 'min_y': min_y, 'min_z': 0.1, 'max_x': max_x, 'max_y': max_y,
                                  'max_z': max_z, 'rotation': rot, 'timestamp': str(t)}
            #vicon_data = check_value(vicon_data, previous_t, str(t))
            #previous_t = str(t)
            with open(path + '/vicon_data/human_bbox.json', 'w') as json_file:
                json.dump(vicon_data, json_file, indent=2)
    print('saved human bbox data')

# Do the same for hupwagen or forklift if it exists in the scene. Hupwagen has 2 topics hupwagen_body and hupwagen_handle.
    vicon_object = '/vicon/markers'
    vicon_data = {}
    save_index = []
    first_flag = 0
    count = 0
    save_flag = True
    max_x = -100000
    max_y = -100000
    max_z = -100000
    min_x = 100000
    min_y = 100000
    min_z = 100000
    if hupwagen:
        obj = 'hupwagen'
        for top, msg, tim in bag.read_messages(vicon_object):

            entry_flag = True
            count = count + 1
            t = msg.header.stamp
            print(obj, 'timestamp: ', t)
            if first_flag == 0:

                for j in range(int(len(msg.markers))):
                    # if the marker_name contains hupwagen in the name string then save that value in a list
                    if msg.markers[j].marker_name.find(obj) != -1:
                        save_index.append(j)
                first_flag = 1

            for i in save_index:
                if  msg.markers[i].occluded == False:
                    if entry_flag:
                        max_x = -100000
                        max_y = -100000
                        max_z = -100000
                        min_x = 100000
                        min_y = 100000
                        min_z = 100000
                        entry_flag = False
                    save_flag = True
                    # find the max and min x, y, z values of the markers among all save_index
                    if msg.markers[i].translation.x > max_x:
                        max_x = msg.markers[i].translation.x
                        # truncate the last character of the marker_name to get the human part

                    if msg.markers[i].translation.y > max_y:
                        max_y = msg.markers[i].translation.y
                    if msg.markers[i].translation.z > max_z:
                        max_z = msg.markers[i].translation.z
                    if msg.markers[i].translation.x < min_x:
                        min_x = msg.markers[i].translation.x
                    if msg.markers[i].translation.y < min_y:
                        min_y = msg.markers[i].translation.y

                    if msg.markers[i].translation.z < min_z:
                        min_z = msg.markers[i].translation.z

                else:
                    print('msg' + str(count) + ' _ ' + str(i), ' is occluded')
                '''
                if obj == 'hupwagen':
                    rot = get_rotation_hupwagen(t)
                elif obj == 'table':
                    rot = get_rotation_table(t)
                '''
            if save_flag:
                vicon_data[str(t)] = {'min_x': min_x, 'min_y': min_y, 'min_z': 0.1, 'max_x': max_x, 'max_y': max_y,
                                      'max_z': max_z, 'timestamp': str(t)}

                with open(path + '/vicon_data/' + obj + '_bbox.json', 'w') as json_file:
                    json.dump(vicon_data, json_file, indent=2)
        print('saved', obj, 'hupwagen bbox data')
        bag.close()




