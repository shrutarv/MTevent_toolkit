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
import time
import rosbag
from dvs_msgs.msg import EventArray
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image
import numpy as np
import tensorflow as tf
from datetime import datetime
sRGB = True
objects_list = ['pallet', 'small_klt', 'big_klt', 'blue_klt', 'shogun_box', 'kronen_bier_crate', 'brinkhoff_bier_crate',
                'zivid_cardboard_box', 'dell_carboard_box', 'ciatronic_carboard_box', 'human', ' hupfwagon', 'mobile_robot']
#obj = ['human', 'zivid','hupwagen']
#objects = ['MR6D2']
#object_name = 'scene33'
human = True
hup = True
table = False
num = [1]
flag = 1
rgb_topic = '/camera/image_raw'
root = '/mnt/smbshare/'
with open(root + "scene_data.json", "r") as file:
    scenes_data = json.load(file)

def linear_to_srgb(image):
    return tf.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * tf.pow(image, 1.0 / 2.4) - 0.055,
    )

def process_image(rgb_image):
    # Normalize the image to [0, 1]
    linear_rgb_image = rgb_image / 255.0

    # Convert to TensorFlow tensor
    linear_rgb_tensor = tf.convert_to_tensor(linear_rgb_image, dtype=tf.float32)

    # Convert Linear RGB to sRGB
    srgb_tensor = linear_to_srgb(linear_rgb_tensor)
    srgb_image = srgb_tensor.numpy()  # Convert back to NumPy array

    # Clip and scale back to [0, 255]
    srgb_image = np.clip(srgb_image * 255.0, 0, 255).astype(np.uint8)
    return srgb_image


for scene, objects in scenes_data.items():
    flag = 1
    for object in objects:
        object_name = scene

        print('Extracting data for object: ', object, ' for scene: ', scene)
        #object_name = object + '_' + str(k)

        # This scripts extracts the topics /dvxplorer_left/events, /vicon/event_cam_sys/event_cam_sys, /rgb/image_raw,
        # /dvxplorer_right/events from the bag file.
        # To extract RGB images, execute extract_rgb_img_from_bag.py Read the bag file

        path = root + object_name + '/'
        bag_all = rosbag.Bag(root + object_name + '/all.bag')
        bag_event_left = rosbag.Bag(root + object_name + '/left.bag')
        bag_event_right = rosbag.Bag(root + object_name + '/right.bag')
        # Extract the topics /dvxplorer_left/events, /vicon/event_cam_sys/event_cam_sys, /rgb/image_raw, /dvxplorer_right/events
        events_topic_left = '/dvxplorer_left/events'
        events_topic_right = '/dvxplorer_right/events'
        vicon_topic_cam_sys = '/vicon/event_cam_sys/event_cam_sys'
        vicon_object = '/vicon/' + object + '/' + object
        vicon_human = '/vicon/markers'
        vicon_human_object = '/vicon/human_head/human_head'
        vicon_hupwagen_object = '/vicon/hupwagen_handle/hupwagen_handle'
        vicon_table_object = '/vicon/table/table'

        events_left = []
        events_right =[]
        vicon = []
        rgb = []
        size = 400000
        vicon_data = {}

        # Iterate over the bag file and extract the messages
        #for topic, msg, t in bag.read_messages(topics=[events_topic_left, events_topic_right, vicon_topic_cam_sys, rgb_topic]):

            # events_left = bag.read_messages(events_topic_left)
        count = 0

        if flag == 1:
            '''
            # mkdir if path does not exist
            if not os.path.exists(path + 'event_cam_left_npy'):
                os.mkdir(path + 'event_cam_left_npy')
            # Iterate over the bag file and extract the messages
            for top, msg, tim in bag_event_left.read_messages(events_topic_left):
                t = msg.header.stamp
                count = len(msg.events)
                event_data = np.zeros(count, dtype=[('t', 'float64'), ('x', 'int32'), ('y', 'int32'), ('p', 'int8')])

                x = []
                y = []
                polarity = []
                for i, event in enumerate(msg.events):
                    event_data[i] = (str(t), event.x, event.y, event.polarity)
                # save x,y polarity and timestamp in a .npy file
                #event_left_data = np.array([t, x, y, polarity], dtype=float)

                np.save(path + 'event_cam_left_npy/' + str(t) + '.npy', event_data)
                #print(count)
                #loaded_array = np.load('array_of_lists.npy', allow_pickle=True)
                #loaded_list1 = loaded_array[0]
                #loaded_list2 = loaded_array[1]
                #loaded_list3 = loaded_array[2]
            print('saved event cam left npy files')
            if not os.path.exists(path + 'event_cam_right_npy'):
                os.mkdir(path + 'event_cam_right_npy')
            for top, msg, tim in bag_event_right.read_messages(events_topic_right):
                t = msg.header.stamp
                count = len(msg.events)
                event_data = np.zeros(count, dtype=[('t', 'float64'), ('x', 'int32'), ('y', 'int32'), ('p', 'int8')])

                x = []
                y = []
                polarity = []
                for i, event in enumerate(msg.events):
                    event_data[i] = (str(t), event.x, event.y, event.polarity)
                # save x,y polarity and timestamp in a .npy file
                # event_left_data = np.array([t, x, y, polarity], dtype=float)

                np.save(path + 'event_cam_right_npy/' + str(t) + '.npy', event_data)
            print('saved event cam right npy files', object, 'scene', scene)
            '''
            count = 0
            if not os.path.exists(path + '/vicon_data'):
                os.makedirs(path + '/vicon_data')
            for top, msg, tim in bag_all.read_messages(vicon_topic_cam_sys):
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
                #vicon_data[str(t)] = {'translation': translation, 'rotation': rotation}
                count += 1
            # if the file exists, delete it
            if os.path.exists(path + '/vicon_data/event_cam_sys.json'):
                os.remove(path + '/vicon_data/event_cam_sys.json')
            with open(path + '/vicon_data/event_cam_sys.json', 'w') as json_file:
                json.dump(vicon_data, json_file, indent=2)
            print('saved event cam data', object, 'scene', scene)

            
            image_topic = bag_all.read_messages(rgb_topic)
            if not os.path.exists(path + '/rgb'):
                os.makedirs(path + '/rgb')

            for l, b in enumerate(image_topic):
                bridge = CvBridge()
                cv_image = bridge.imgmsg_to_cv2(b.message, "bgr8")
                # cv_image.astype(np.uint8)
                if sRGB:
                    cv_image = process_image(cv_image)
                # cv_image = cv_image[45:480,0:595]
                # cv_image = cv2.resize(cv_image, (640,480))
                cv2.imwrite(path + '/rgb/' + str(b.timestamp) + '.jpg', cv_image)
                # jpg images are required for SAM2
                #print('saved: images',)
            print('Done Extracting RGB images', object, 'scene', scene)

            # Close the bag file
            if human:
                if not os.path.exists(path + '/vicon_data'):
                    os.makedirs(path + '/vicon_data')
                vicon_data = {}
                print('Extracting data for human ')
                for top, msg, tim in bag_all.read_messages(vicon_human_object):
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
                # if the file exists, delete it
                if os.path.exists(path + '/vicon_data/human.json'):
                    os.remove(path + '/vicon_data/human.json')
                with open(path + 'vicon_data/' + 'human.json', 'w') as json_file:
                    json.dump(vicon_data, json_file, indent=2)
                print('saved human data', object, 'scene', scene)

            if hup:
                if not os.path.exists(path + '/vicon_data'):
                    os.makedirs(path + '/vicon_data')
                vicon_data = {}
                print('Extracting data for hupwagen')
                for top, msg, tim in bag_all.read_messages(vicon_hupwagen_object):
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
                # if the file exists, delete it
                if os.path.exists(path + '/vicon_data/hupwagen.json'):
                    os.remove(path + '/vicon_data/hupwagen.json')

                with open(path + 'vicon_data/' + 'hupwagen.json', 'w') as json_file:
                    json.dump(vicon_data, json_file, indent=2)
                print('saved hupwagen data', object, 'scene', scene)

            flag = 0

        if not os.path.exists(path + '/vicon_data'):
            os.makedirs(path + '/vicon_data')
        vicon_data = {}
        print('Extracting data for object: ', object, ' with scene: ', scene)
        for top, msg, tim in bag_all.read_messages(vicon_object):
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
        # if the file exists, delete it
        if os.path.exists(path + '/vicon_data/' + object + '.json'):
            os.remove(path + '/vicon_data/' + object + '.json')

        with open(path + 'vicon_data/' + object + '.json', 'w') as json_file:
            json.dump(vicon_data, json_file, indent=2)
        print('saved object data')




    bag_all.close()
    bag_event_left.close()
    bag_event_right.close()
    print('Done extracting data')


