# Transform the coordinates in RGB camera frame to event camera frame
# RGB camera is cam0, Event camera 1 is cam1 and event camera 2 is cam2
import numpy as np
import cv2
import os
import json
import trimesh

from utilities import *
import torch

# 1: "wooden_pallet", 2: "small_klt", 3: "big_klt", 4: "blue_klt", 5: "shogun_box",
# 6: "kronen_bier_crate", 7: "brinkhoff_bier_crate", 8: "zivid_cardboard_box", 9: "dell_carboard_box", 10: "ciatronic_carboard_box", 11: "human"

#object_name = 'scene12'

threshold = 10000000
last = False
iter = 1
# list objects such as human, hupwagen, etc
objects = ['human','hupwagen']
obj_iter = 0
human = True
base_dir = "/media/eventcamera/event_data/dataset_8_apr_zft/"
with open(base_dir + "scene_data.json", "r") as file:
    scenes_data = json.load(file)

for scene, o in scenes_data.items():
    for obj in objects:
        object_name = scene
        print('scene ', object_name)
        # import object data from json file
        with open('/home/eventcamera/RGB_Event_cam_system/Annotation/Annotation_rgb_ec/obj_model/models_info.json', 'r') as file:
            obj_model_data = json.load(file)
        root_dir = base_dir + object_name
        human_bbox_path = root_dir + '/vicon_data/' + obj + '_bbox.json'
        path = root_dir + '/' + object_name + '_' + obj
        json_path_camera_sys = root_dir + '/vicon_data/event_cam_sys.json'
        json_path_object = root_dir + '/vicon_data/' + obj + '.json'
        path_event_cam_left_img = root_dir  + '/event_images/left/'
        path_event_cam_right_img = root_dir + '/event_images/right/'
        output_dir = root_dir + '/annotation_' + obj + '/'
        output_dir_rgb = output_dir + obj + '_rgb_'
        output_dir_event_cam_left = output_dir + obj + '_ec_left_'
        output_dir_event_cam_right = output_dir + obj + '_ec_right_'
        rgb_image_path = root_dir + '/rgb/'
        iter += 1
        # if any of the above paths does not exist, create the path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                os.unlink(file_path)

        # extract the time stamps from the images from all 3 cameras
        rgb_timestamp = os.listdir(rgb_image_path)
        rgb_timestamp.sort()

        event_cam_left_timestamp = os.listdir(path_event_cam_left_img)
        event_cam_left_timestamp.sort()

        event_cam_right_timestamp = os.listdir(path_event_cam_right_img)
        event_cam_right_timestamp.sort()

        rgb_timestamp = check_max_timestamp_rgb_event(rgb_timestamp, event_cam_left_timestamp, event_cam_right_timestamp)

        # Load the vicon data of the human head object that is recorded using vicon when the dataset was created.
        with open(json_path_object, 'r') as file:
            vicon_object_data = json.load(file)
            # if vicon_object_data is empty, then skip the object
            if not vicon_object_data:
                print('No data for object ', obj, 'in scene', object_name)
                continue

        # extract only timestamp in a numpy array from dictionary loaded_array
        timestamp_vicon_object = []

        for k, v in vicon_object_data.items():
            timestamp_vicon_object.append(v['timestamp'])
        timestamp_vicon_object = np.array(timestamp_vicon_object)

        #rgb_timestamp = remove_extension_and_convert_to_int(rgb_timestamp)
        event_cam_left_timestamp = remove_extension_and_convert_to_int(event_cam_left_timestamp)
        event_cam_right_timestamp = remove_extension_and_convert_to_int(event_cam_right_timestamp)

        # Associate timestamps in both event cameras to rgb camera timestamps.
        timestamp_vicon_object = list(map(int, timestamp_vicon_object))
        dict_rgb_viconObject = find_closest_elements(rgb_timestamp,
                                            timestamp_vicon_object)  # Output in format (rgb_timestamp, timestamp_vicon_object)
        #vicon_coord = []
        timestamps_closest_object = list(dict_rgb_viconObject.values())
        dict_rgb_ec_left = find_closest_elements(rgb_timestamp, event_cam_left_timestamp)
        dict_rgb_ec_right = find_closest_elements(rgb_timestamp, event_cam_right_timestamp)

        # remove delayed timestamps. There could be timestamps which are further apart than the expected time difference between the images.
        #dict_rgb_ec_left = remove_delayed_timestamps(dict_rgb_ec_left, threshold)
        #dict_rgb_ec_right = remove_delayed_timestamps(dict_rgb_ec_right, threshold)
        #result_dict = remove_delayed_timestamps(dict_rgb_viconObject, threshold)

        timestamp_closest_ec_left = list(dict_rgb_ec_left.values())
        timestamp_closest_ec_right = list(dict_rgb_ec_right.values())

        # vicon object data contains timestamped data which is much larger than rgb timestamped images. We only need
        # the data for vicon object that corresponds to rgb timestamps. Hence, we extract the data for the trans and rot for the closest timestamps of the object
        vicon_object_translations_with_timestamps = {
            timestamp: np.array(vicon_object_data[str(timestamp)]["translation"])
            for timestamp in timestamps_closest_object}
        vicon_object_rotations_with_timestamps = {
            timestamp: np.array(vicon_object_data[str(timestamp)]["rotation"])
            for timestamp in timestamps_closest_object
        }

        # load camera parameters and transformation data from json file
        with open(root_dir + '/camera_params.json', 'r') as file:
            data = json.load(file)
        # cam1 is event camera left and cam2 is event camera right
        camera_mtx_left = np.array(data['camera_matrix'])
        distortion_coefficients_left = np.array(data['distortion_coefficients'])
        camera_mtx_rgb = np.array(data['camera_mtx_cam1'])
        distortion_coefficients_rgb = np.array(data['distortion_coeffs_cam1'])
        camera_mtx_right = np.array(data['camera_mtx_cam2'])
        distortion_coefficients_right = np.array(data['distortion_coeffs_cam2'])

        H_cam_sys_2_rgb = np.array(data['H_cam_sys_2_rgb'])
        H_rgb_2_left = np.array(data['H_rgb_2_left'])
        H_right_2_rgb = np.array(data['H_right_2_rgb'])
        # inverse of H_rgb_2_left using numpy
        H_left_2_rgb = np.linalg.inv(H_rgb_2_left)
        #compute H_right_2_left using existing transformations
        H_right_2_left = np.dot(H_right_2_rgb, H_rgb_2_left)
        #H_right_2_left = np.array(data['H_right_2_left'])

        ######### Here we save the transformations for the dataset. Annotations are not happening here. #########
        # Read the vicon coordinates of the even camera system. Traverse through the coordinates
        with open(json_path_camera_sys, 'r') as f:
            vicon_data_camera_sys = json.load(f)
        save_transformations_human(vicon_data_camera_sys, H_cam_sys_2_rgb, vicon_object_data.copy(), H_left_2_rgb, H_right_2_left,
                             path)

        ################## ANNOTATIONS #################
        count = 0
        with open(path + '_transformations.json', 'r') as file:
            projected_point_rgb_ec1_ec2 = json.load(file)

        timestamp_closest_ec_right = sorted(timestamp_closest_ec_right)
        timestamp_closest_ec_left = sorted(timestamp_closest_ec_left)
        rgb_timestamp = sorted(rgb_timestamp)
        #global data_human_bbox
        with open(human_bbox_path, 'r') as f:
            data_human_bbox = json.load(f)
        previous_t = timestamps_closest_object[0]

        for (kr, vr), (k, v) in zip(vicon_object_rotations_with_timestamps.items(), vicon_object_translations_with_timestamps.items()):
            # kr and k are timestamps for respective values
            if count>= len(timestamp_closest_ec_right) or count>= len(timestamp_closest_ec_left) or count>= len(rgb_timestamp):
                continue
            print('timestamp ', kr , ' image ', count)
            ############# Defining paths for images #############
            rgb_t = rgb_timestamp[count]
            ec_left = timestamp_closest_ec_left[count]
            ec_right = timestamp_closest_ec_right[count]
            rgb_img_path = rgb_image_path + str(rgb_t) + ".jpg"

            event_cam_left = path_event_cam_left_img + str(ec_left) + ".png"
            event_cam_right = path_event_cam_right_img + str(ec_right) + ".png"

            data_human_bbox,k = check_human_bbox_data(data_human_bbox, k, previous_t)
            vertices = get_BBox_vertices(data_human_bbox, k)
            previous_t = k
            points_3d = vertices

            ############ RGB Image ############
            H_rgb_2_vicon = np.array(projected_point_rgb_ec1_ec2[str(k)]['H_rgb_2_vicon'])
            # True is given if you want to save the bounding box and pose data of the object.
            img_rgb = project_points_to_image_plane(obj, H_rgb_2_vicon, k, 0, rgb_img_path, points_3d, vertices,
                                                            camera_mtx_rgb, distortion_coefficients_rgb, output_dir_rgb, root_dir, obj_iter, True, human)
            #cv2.imwrite(rgb_img_path, img_rgb)

            ############ Event camera 1 ############
            H_cam1_2_object = np.array(projected_point_rgb_ec1_ec2[str(k)]['H_cam1_2_vicon'])
            img_event_cam_1 = project_points_to_image_plane(obj, H_cam1_2_object, ec_left, 0,event_cam_left, points_3d, vertices,
                                                            camera_mtx_left, distortion_coefficients_left,output_dir_event_cam_left, root_dir, obj_iter, True, human)
            #cv2.imwrite(event_cam_left, img_event_cam_1)

            ############ Event camera 2 ############
            H_cam2_2_object = np.array(projected_point_rgb_ec1_ec2[str(k)]['H_cam2_2_vicon'])
            img_event_cam_2 = project_points_to_image_plane(obj, H_cam2_2_object, ec_right, 0, event_cam_right, points_3d, vertices,
                                                            camera_mtx_right, distortion_coefficients_right, output_dir_event_cam_right, root_dir, obj_iter, True, human)
            #cv2.imwrite(event_cam_right, img_event_cam_2)
            count += 1


            ########### Display the images ###########
            img_rgb = cv2.resize(img_rgb, (568, 426))
            img_event_cam_1 = cv2.resize(img_event_cam_1, (568, 426))
            img_event_cam_2 = cv2.resize(img_event_cam_2, (568, 426))

            concatenated_images = np.hstack((img_event_cam_1, img_rgb, img_event_cam_2))
            output_path = os.path.join(output_dir, f'image_{count:03d}.jpg')
            cv2.imwrite(output_path, concatenated_images)

            cv2.waitKey(0)

            cv2.destroyAllWindows()