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
# 6: "kronen_bier_crate", 7: "brinkhoff_bier_crate", 8: "zivid_cardboard_box", 9: "dell_carboard_box", 10: "ciatronic_carboard_box"

object_name = 'scene_12'
#obj_name = 'blue_klt'
objects = ['MR6D4']

threshold = 10000000
# import object data from json file
with open('/home/eventcamera/RGB_Event_cam_system/Annotation/Annotation_rgb_ec/obj_model/models_info.json', 'r') as file:
    obj_model_data = json.load(file)
obj_iter = 0
for obj_name in objects:
    if obj_name == 'MR6D1':
         object_id = 1
    elif obj_name == 'MR6D2':
            object_id = 2
    elif obj_name == 'MR6D3':
            object_id = 3
    elif obj_name == 'MR6D4':
            object_id = 4
    elif obj_name == 'MR6D5':
            object_id = 5
    elif obj_name == 'MR6D6':
            object_id = 6
    elif obj_name == 'MR6D7':
            object_id = 7
    elif obj_name == 'MR6D8':
            object_id = 8
    elif obj_name == 'MR6D9':
            object_id = 9
    elif obj_name == 'MR6D10':
            object_id = 10
    elif obj_name == 'MR6D11':
            object_id = 11
    elif obj_name == 'MR6D12':
            object_id = 12
    elif obj_name == 'MR6D13':
            object_id = 13
    elif obj_name == 'MR6D14':
            object_id = 14
    elif obj_name == 'MR6D15':
            object_id = 15
    elif obj_name == 'MR6DD16':
            object_id = 16

    object_len_x = obj_model_data[str(object_id)]['size_x']
    object_len_y = obj_model_data[str(object_id)]['size_y']
    object_len_z = obj_model_data[str(object_id)]['size_z']
    root_dir = '/media/eventcamera/Windows/dataset_7_feb/'
    path = root_dir + object_name + '/' + object_name
    json_path_camera_sys = root_dir + object_name + '/vicon_data/event_cam_sys.json'
    json_path_object = root_dir + object_name + '/vicon_data/' + obj_name + '.json'
    path_event_cam_left_img = root_dir + object_name + '/event_images/'
    path_event_cam_right_img = root_dir + object_name + '/event_cam_right/e2calib/'
    output_dir = root_dir + object_name + '/annotation/'
    output_dir_rgb = root_dir + object_name + '/annotation/rgb_' + obj_name + '_'
    output_dir_event_cam_left = root_dir + object_name + '/annotation/ec_left_' + obj_name + '_'
    output_dir_event_cam_right = root_dir + object_name + '/annotation/ec_right_' + obj_name + '_'
    rgb_image_path = root_dir + object_name + '/rgb/'
    object_id_padded = f"{object_id:06d}"
    obj_path = '/home/eventcamera/RGB_Event_cam_system/Annotation/Annotation_rgb_ec/obj_model/obj_' + str(object_id_padded) + '.ply'

    # if any of the above paths does not exist, create the path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if obj_iter == 0:
                os.unlink(file_path)
            #os.unlink(file_path)

    # extract the time stamps from the images from all 3 cameras
    rgb_timestamp = os.listdir(rgb_image_path)
    rgb_timestamp.sort()

    event_cam_left_timestamp = os.listdir(path_event_cam_left_img)
    event_cam_left_timestamp.sort()

    event_cam_right_timestamp = os.listdir(path_event_cam_right_img)
    event_cam_right_timestamp.sort()

    rgb_timestamp = check_max_timestamp_rgb_event(rgb_timestamp, event_cam_left_timestamp, event_cam_right_timestamp)

    # Load the vicon data of the object that is recorded using vicon when the dataset was created.
    with open(json_path_object, 'r') as file:
        vicon_object_data = json.load(file)
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
    with open('/home/eventcamera/RGB_Event_cam_system/Annotation/camera_params.json', 'r') as file:
        data = json.load(file)
    # cam1 is event camera left and cam2 is event camera right
    camera_matrix = np.array(data['camera_matrix'])
    distortion_coefficients = np.array(data['distortion_coefficients'])
    camera_mtx_cam2 = np.array(data['camera_mtx_cam2'])
    distortion_coeffs_cam2 = np.array(data['distortion_coeffs_cam2'])
    camera_mtx_cam1 = np.array(data['camera_mtx_cam1'])
    distortion_coeffs_cam1 = np.array(data['distortion_coeffs_cam1'])
    H_cam_vicon_2_rgb = np.array(data['H_cam_vicon_2_cam_optical'])
    H_cam1_2_rgb = np.array(data['H_cam1_2_rgb'])
    H_cam2_cam1 = np.array(data['H_cam2_cam1'])

    ######### Here we save the transformations for the dataset. Annotations are not happening here. #########
    # Read the vicon coordinates of the event camera system.
    with open(json_path_camera_sys, 'r') as f:
        vicon_data_camera_sys = json.load(f)
    save_transformations(vicon_data_camera_sys, H_cam_vicon_2_rgb, vicon_object_data.copy(), H_cam1_2_rgb, H_cam2_cam1, path)

    ################## ANNOTATIONS #################
    count = 0
    with open(path + '_transformations.json', 'r') as file:
        projected_point_rgb_ec1_ec2 = json.load(file)

    timestamp_closest_ec_right = sorted(timestamp_closest_ec_right)
    timestamp_closest_ec_left = sorted(timestamp_closest_ec_left)
    rgb_timestamp = sorted(rgb_timestamp)

    for (kr, vr), (k, v) in zip(vicon_object_rotations_with_timestamps.items(), vicon_object_translations_with_timestamps.items()):

        # kr and k are timestamps for respective values
        print('timestamp ', k , 'obj_name', obj_name, ' object ', count)
        ############# Defining paths for images #############
        rgb_t = rgb_timestamp[count]
        ec_left = timestamp_closest_ec_left[count]
        ec_right = timestamp_closest_ec_right[count]
        rgb_img_path = rgb_image_path + str(rgb_t) + ".png"
        event_cam_left = path_event_cam_left_img + str(ec_left) + ".png"
        event_cam_right = path_event_cam_right_img + str(ec_right) + ".png"

        ####### Import object ply file and create a mesh for visualization #######1
        obj_geometry = trimesh.load_mesh(obj_path)
        if not isinstance(obj_geometry, trimesh.Trimesh):
            print("The object is not a Trimesh object. It is a", type(obj_geometry))
        trimesh_object = obj_geometry.convex_hull
        points_3d = np.array(trimesh_object.sample(50000)) / 1000
        vertices = np.array(trimesh_object.vertices) / 1000
        vertices, points_3d = get_translated_points_vertice(object_id, vertices, points_3d, object_len_z)

        folder_path = root_dir + object_name
        ############ RGB Image ############
        H_rgb_2_object = np.array(projected_point_rgb_ec1_ec2[str(k)]['H_rgb_2_object'])
        # True is given if you want to save the bounding box and pose data of the object.
        img_rgb = project_points_to_image_plane(H_rgb_2_object, k, rgb_t, rgb_img_path, points_3d, vertices,
                                                        camera_matrix, distortion_coefficients, output_dir_rgb, folder_path, obj_iter, True)

        if len(objects) > 1:
            cv2.imwrite(rgb_img_path, img_rgb)

        ############ Event camera 1 ############
        event_t = 0
        H_cam1_2_object = np.array(projected_point_rgb_ec1_ec2[str(k)]['H_cam1_2_object'])
        img_event_cam_1 = project_points_to_image_plane(H_cam1_2_object, ec_left, event_t, event_cam_left, points_3d, vertices,
                                                        camera_mtx_cam1, distortion_coeffs_cam1,output_dir_event_cam_left, root_dir, obj_iter, True)
        if len(objects) > 1:
            cv2.imwrite(event_cam_left, img_event_cam_1)

        ############ Event camera 2 ############
        H_cam2_2_object = np.array(projected_point_rgb_ec1_ec2[str(k)]['H_cam2_2_object'])
        img_event_cam_2 = project_points_to_image_plane(H_cam2_2_object, ec_right, event_t, event_cam_right, points_3d, vertices,
                                                        camera_mtx_cam2, distortion_coeffs_cam2, output_dir_event_cam_right, root_dir, obj_iter, True)
        if len(objects) > 1:
            cv2.imwrite(event_cam_right, img_event_cam_2)

        obj_iter += 1
        ########### Display the images ###########
        img_rgb = cv2.resize(img_rgb, (568, 426))
        img_event_cam_1 = cv2.resize(img_event_cam_1, (568, 426))
        img_event_cam_2 = cv2.resize(img_event_cam_2, (568, 426))

        concatenated_images = np.hstack((img_event_cam_1, img_rgb, img_event_cam_2))
        output_path = os.path.join(output_dir, f'image_{count:03d}.jpg')
        cv2.imwrite(output_path, concatenated_images)
        #cv2.imshow('Image', concatenated_images)
        cv2.waitKey(0)
        count += 1
        cv2.destroyAllWindows()