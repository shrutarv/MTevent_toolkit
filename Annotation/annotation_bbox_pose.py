# Transform the coordinates in RGB camera frame to event camera frame
# RGB camera is cam0, Event camera 1 is cam1 and event camera 2 is cam2
import numpy as np
import cv2
import os
import json
import trimesh
from utilities import *
import pyrender
import torch

# 1: "wooden_pallet", 2: "small_klt", 3: "big_klt", 4: "blue_klt", 5:  "Amazon basics luggage",
# 	6:  "IKEA_Dammang _bin_with_lid", 7: "IKEA vesken trolley", 8: "IKEA sortera waste sorting bin", 9: "IKEA Drona grey", 10: "IKEA Drona blue"
# 	11: "IKEA KNALLIG wooden box", 12: "IKEA MOPPE mini drawer", 13: "IKEA LABBSAL basket", 14: "IKEA IVAR box on wheels", 15: "IKEA SKUBB storage case",
# 	16: "IKEA SAMLA transparent box"

object_name = 'scene74'
#obj_name = 'blue_klt'
objects = ['MR6D12','MR6D2']
root_dir = '/media/eventcamera/event_data/dataset_20_march_zft/'
calib_obj_centre_vicon_geometric = False
threshold = 10000000

obj_model_path = '/home/eventcamera/RGB_Event_cam_system/Annotation/Annotation_rgb_ec/obj_model/'
# import object data from json file
with open(obj_model_path + 'models_info.json', 'r') as file:
    obj_model_data = json.load(file)
with open(root_dir + "scene_data.json", "r") as file:
    scenes_data = json.load(file)
obj_iter = 0
#for scenes,o in scenes_data.items():
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
    elif obj_name == 'MR6D16':
            object_id = 16
    if calib_obj_centre_vicon_geometric:
        with open(obj_model_path + 'dynamic_objects_calibration_diff.json', 'r') as json_file:
            H_vicon_to_object_uncalibrated_to_calibrated = json.load(json_file)
        H_vicon_to_object_uncalibrated_to_calibrated = np.array(H_vicon_to_object_uncalibrated_to_calibrated
                                                                [str(object_id)]['H_object_uncalibrated_to_calibrated'])
        H_vicon_to_object_uncalibrated_to_calibrated[:3, 3] = H_vicon_to_object_uncalibrated_to_calibrated[:3, 3] / 1000
        # invert H_vicon_to_object_uncalibrated_to_calibrated
        #H_vicon_to_object_uncalibrated_to_calibrated = np.linalg.inv(H_vicon_to_object_uncalibrated_to_calibrated)
    else:
        H_vicon_to_object_uncalibrated_to_calibrated = np.eye(4)

    object_len_x = obj_model_data[str(object_id)]['size_x']
    object_len_y = obj_model_data[str(object_id)]['size_y']
    object_len_z = obj_model_data[str(object_id)]['size_z']

    path = root_dir + object_name + '/' + object_name
    json_path_camera_sys = root_dir + object_name + '/vicon_data/event_cam_sys.json'
    json_path_object = root_dir + object_name + '/vicon_data/' + obj_name + '.json'
    path_event_cam_left_img = root_dir + object_name + '/event_images/left/'
    path_event_cam_right_img = root_dir + object_name + '/event_images/right/'
    output_dir = root_dir + object_name + '/annotation/'
    output_dir_rgb = root_dir + object_name + '/annotation/rgb_' + obj_name + '_'
    output_dir_event_cam_left = root_dir + object_name + '/annotation/ec_left_' + obj_name + '_'
    output_dir_event_cam_right = root_dir + object_name + '/annotation/ec_right_' + obj_name + '_'
    rgb_image_path = root_dir + object_name + '/rgb/'
    object_id_padded = f"{object_id:06d}"
    obj_path = obj_model_path + 'obj_' + str(object_id_padded) + '.ply'

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
    # save the dict_rgb_ec_left and dict_rgb_ec_right to a json file
    '''
    with open(path + '_closest_ec_left.json', 'w') as file:
        json.dump(dict_rgb_ec_left, file)
    with open(path + '_closest_ec_right.json', 'w') as file:
        json.dump(dict_rgb_ec_right, file)
    '''
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
    with open(root_dir + object_name + '/camera_params.json', 'r') as file:
        data = json.load(file)
    # cam1 is event camera left and cam2 is event camera right
    camera_mtx_left = np.array(data['camera_matrix'])
    distortion_coefficients_left = np.array(data['distortion_coefficients'])
    camera_mtx_rgb = np.array(data['camera_mtx_cam1'])
    distortion_coefficients_rgb = np.array(data['distortion_coeffs_cam1'])
    camera_mtx_right = np.array(data['camera_mtx_cam2'])
    distortion_coefficients_right = np.array(data['distortion_coeffs_cam2'])

    H_cam_sys_2_rgb = np.array(data['H_cam_sys_2_rgb'])  # got from eye in hand calibration
    H_rgb_2_left = np.array(data['H_rgb_2_left'])
    H_right_2_rgb = np.array(data['H_right_2_rgb'])
   # H_cam_sys_2_right = np.array(data['H_cam_sys_2_right'])

    ######### Here we save the transformations for the dataset. Annotations are not happening here. #########
    # Read the vicon coordinates of the event camera system.
    with open(json_path_camera_sys, 'r') as f:
        vicon_data_camera_sys = json.load(f)
    save_transformations(vicon_data_camera_sys, H_cam_sys_2_rgb, vicon_object_data.copy(), H_rgb_2_left, H_right_2_rgb, path, H_vicon_to_object_uncalibrated_to_calibrated)

    ################## ANNOTATIONS #################
    count = 0
    with open(path + '_transformations.json', 'r') as file:
        projected_point_rgb_ec1_ec2 = json.load(file)

    timestamp_closest_ec_right = sorted(timestamp_closest_ec_right)
    timestamp_closest_ec_left = sorted(timestamp_closest_ec_left)
    rgb_timestamp = sorted(rgb_timestamp)
    minimum_frames = min(len(timestamp_closest_ec_left), len(timestamp_closest_ec_right), len(rgb_timestamp))
    for (kr, vr), (k, v) in zip(vicon_object_rotations_with_timestamps.items(), vicon_object_translations_with_timestamps.items()):

        # kr and k are timestamps for respective values
        print('timestamp ', k , 'obj_name', obj_name, ' object ', count)
        '''
        if count>1028:
            count += 1
            continue
        '''
        if count >= minimum_frames:
            print('skipping frame because it is extra in one of the cameras', count)
            continue
        ############# Defining paths for images #############
        rgb_t = rgb_timestamp[count]
        ec_left = timestamp_closest_ec_left[count]
        ec_right = timestamp_closest_ec_right[count]
        rgb_img_path = rgb_image_path + str(rgb_t) + ".jpg"
        event_cam_left = path_event_cam_left_img + str(ec_left) + ".png"
        event_cam_right = path_event_cam_right_img + str(ec_right) + ".png"

        ####### Import object ply file and create a mesh for visualization #######
        obj_geometry = trimesh.load_mesh(obj_path)
        if not isinstance(obj_geometry, trimesh.Trimesh):
            print("The object is not a Trimesh object. It is a", type(obj_geometry))

        trimesh_object = obj_geometry.convex_hull
        points_3d = np.array(trimesh_object.sample(50000)) / 1000
        vertices = np.array(trimesh_object.vertices) / 1000
        vertices, points_3d = get_translated_points_vertice(object_id, vertices, points_3d, object_len_z, object_len_x, object_len_y)
        vertices = get_vertices(vertices)
        folder_path = root_dir + object_name
        ############ RGB Image ############
        H_rgb_2_object = np.array(projected_point_rgb_ec1_ec2[str(k)]['H_rgb_2_object'])



        H_camera_object = np.linalg.inv(H_rgb_2_object)
        # Create a pyrender scene
        scene = pyrender.Scene()

        # Prepare mesh (reuse the already loaded trimesh object)
        mesh = pyrender.Mesh.from_trimesh(obj_geometry, smooth=False)
        scene.add(mesh, pose=H_camera_object)

        # Define camera intrinsics
        fx, fy = camera_mtx_rgb[0, 0], camera_mtx_rgb[1, 1]
        cx, cy = camera_mtx_rgb[0, 2], camera_mtx_rgb[1, 2]
        cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
        scene.add(cam, pose=np.eye(4))

        # Render offscreen to get depth and mask
        renderer = pyrender.OffscreenRenderer(viewport_width=1448, viewport_height=1086)
        color, depth = renderer.render(scene)

        # Create binary mask where object is visible (depth > 0)
        mask = (depth > 0).astype(np.uint8) * 255

        # Optional: save or use the mask
        cv2.imwrite(os.path.join(output_dir, f"mask_rgb_{count:03d}.png"), mask)

        # Optional overlay for visualization
        image = cv2.imread(rgb_img_path)
        overlay = image.copy()
        overlay[mask > 0] = [0, 255, 0]  # Green overlay
        blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

        cv2.imshow("RGB with Object Mask", blended)
        cv2.waitKey(0)





        # True is given if you want to save the bounding box and pose data of the object.
        img_rgb = project_points_to_image_plane(obj_name, H_rgb_2_object, k, rgb_t, rgb_img_path, points_3d, vertices,
                                                        camera_mtx_rgb, distortion_coefficients_rgb, output_dir_rgb, folder_path, obj_iter, True)

        if len(objects) > 1:
            cv2.imwrite(rgb_img_path, img_rgb)

        ############ Event camera left ############
        event_t = 0
        H_left_2_object = np.array(projected_point_rgb_ec1_ec2[str(k)]['H_left_2_object'])
        img_event_cam_left = project_points_to_image_plane(obj_name, H_left_2_object, ec_left, event_t, event_cam_left, points_3d, vertices,
                                                        camera_mtx_left, distortion_coefficients_left,output_dir_event_cam_left, root_dir, obj_iter, True)
        if len(objects) > 1:
            cv2.imwrite(event_cam_left, img_event_cam_left)

        ############ Event camera right ############
        H_right_2_object = np.array(projected_point_rgb_ec1_ec2[str(k)]['H_right_2_object'])
        # shift the transformation to 1 unit in positive z and 1 unit in negative y

        #H_right_2_object[0][3] = H_right_2_object[0][3] - 0.05
        #H_right_2_object[1][3] = H_right_2_object[1][3] - 0.05
        #H_right_2_object[2][3] = H_right_2_object[2][3] + 1.0

        img_event_cam_right = project_points_to_image_plane(obj_name, H_right_2_object, ec_right, event_t, event_cam_right, points_3d, vertices,
                                                        camera_mtx_right, distortion_coefficients_right, output_dir_event_cam_right, root_dir, obj_iter, True)
        if len(objects) > 1:
            cv2.imwrite(event_cam_right, img_event_cam_right)


        ########### Display the images ###########
        img_rgb = cv2.resize(img_rgb, (568, 426))
        img_event_cam_left = cv2.resize(img_event_cam_left, (568, 426))
        img_event_cam_right = cv2.resize(img_event_cam_right, (568, 426))

        concatenated_images = np.hstack((img_event_cam_left, img_rgb, img_event_cam_right))
        output_path = os.path.join(output_dir, f'image_{count:03d}.jpg')
        cv2.imwrite(output_path, concatenated_images)
        #cv2.imshow('Image', concatenated_images)
        cv2.waitKey(0)
        count += 1
        cv2.destroyAllWindows()
    obj_iter += 1
'''
rgb_timestamp = os.listdir(root_dir + object_name + '/rgb/')
rgb_timestamp.sort()
number_of_objects = len(objects)

if number_of_objects > 1:
    for obj in objects:
        # subtract the maks of all other objects from the mask of the current object
        # get the mask of the object
        mask_dir = os.listdir(root_dir + object_name + '/masks_rgb_' + obj + '/')
        mask_dir.sort()
        # remove all png files from mask_dir
        mask_dir = [x for x in mask_dir if x.endswith(".npy")]
        # get the mask of the other objects
        other_objects = [x for x in objects if x != obj]
        for other_obj in other_objects:
            mask_dir_other = os.listdir(root_dir + object_name + '/masks_rgb_' + other_obj + '/')
            mask_dir_other.sort()
            mask_dir_other = [x for x in mask_dir_other if x.endswith(".npy")]
            for i in mask_dir:
                mask_other = np.load(root_dir + object_name + '/masks_rgb_' + other_obj + '/' + i)
                mask = np.load(root_dir + object_name + '/masks_rgb_' + obj + '/' + i)
                # change mask_other to 0 and 1 instead of 0 and 255
                mask_other = mask_other / 255
                mask = mask * (1 - mask_other)
                np.save(root_dir + object_name + '/masks_rgb_' + obj + '/' + i, mask)
                # remove .npy extension from i and add.jpy extension
                i = i[:-4] + '.jpg'
                cv2.imwrite(root_dir + object_name + '/masks_rgb_' + obj + '/' + i, mask)



    #get_masks_visible_object(objects, root_dir, object_name)

'''