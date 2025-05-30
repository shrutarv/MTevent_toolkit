import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import os
import cv2

# if the timestamp recorded in rgb camera are for a longer duration than event cameras then delete
# all the rgb timestamps which are greater than max timestamp in event camera.
def check_max_timestamp_rgb_event(rgb_timestamp, event_cam_left_timestamp, event_cam_right_timestamp):
    rgb_timestamp = remove_extension(rgb_timestamp)
    event_cam_left_timestamp = remove_extension(event_cam_left_timestamp)
    event_cam_right_timestamp = remove_extension(event_cam_right_timestamp)

    indexes = []
    if (int(rgb_timestamp[-1]) > int(event_cam_left_timestamp[-1])) or (int(rgb_timestamp[-1]) > int(event_cam_right_timestamp[-1])):
        for i in range(len(rgb_timestamp)):
            #print(i)
            if (int(rgb_timestamp[i]) > (int(event_cam_left_timestamp[-1]))) or (int(rgb_timestamp[i]) > int(event_cam_right_timestamp[-1])):
                # store these indexes in a list and delete all the elements in rgb_timestamp corresponding to these indexes
                indexes.append(i)
        # sort indexes from highest to lowest
        indexes.sort(reverse=True)
        for i in indexes:
            del rgb_timestamp[i]
    #convert rgb_timestamp to int
    rgb_timestamp = [int(i) for i in rgb_timestamp]
    return rgb_timestamp

def find_closest_elements(A, B):
    result = {}

    for a in A:
        # check if the first element in in A (rgb timestamp) is less than the first element in B (event camera timestamp).
        # If it is then start from a that is larger than the first element in B
        if a < B[0] or a>B[-1]:
            continue

        closest_b = min(B, key=lambda x: abs(x - a))
        result[a] = closest_b
        B.remove(closest_b)

    return result

def remove_extension(arr1):
    modified_arr = [file_name[:-4] for file_name in arr1 if file_name.endswith('.png') or file_name.endswith('.jpg')]
    return modified_arr

def remove_extension_and_convert_to_int(arr):
    # Remove ".png" extension and convert to integers
    modified_arr = [int(file_name[:-4]) for file_name in arr if file_name.endswith('.png') or file_name.endswith('.jpg')]
    return modified_arr

def remove_delayed_timestamps(result_dict, threshold):
    keys_to_remove = []
    for key, value in result_dict.items():
        deviation = abs(int(key) - int(value))

        if deviation > threshold:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del result_dict[key]
    return result_dict

def save_transformations_human(vicon_data_camera_sys, H_cam_vicon_2_rgb, vicon_object_data, H_cam1_2_rgb, H_cam2_cam1, path):
    transformations = {}
    for i, v in vicon_data_camera_sys.items():
        if i == str(len(vicon_data_camera_sys) - 1):
            continue
        vicon_cam_translation = vicon_data_camera_sys[str(i)]['translation']
        vicon_cam_rotation_quat = vicon_data_camera_sys[str(i)]['rotation']

        # get rotation matrix from quaternion
        rotation_vicon_cam = R.from_quat(vicon_cam_rotation_quat).as_matrix()
        # make homogeneous transformation matrix
        H_world_2_cam_vicon = np.eye(4)
        H_world_2_cam_vicon[:3, :3] = rotation_vicon_cam
        H_world_2_cam_vicon[:3, 3] = vicon_cam_translation

        # make homogeneous transformation matrix from vicon to camera optical frame
        H_world_2_rgb = np.matmul(H_world_2_cam_vicon, H_cam_vicon_2_rgb)

        # invert H_vicon_2_cam_optical to get H_cam_optical_2_vicon
        H_rgb_2_world = np.eye(4)
        H_rgb_2_world[:3, :3] = np.transpose(H_world_2_rgb[:3, :3])
        H_rgb_2_world[:3, 3] = -np.matmul(np.transpose(H_world_2_rgb[:3, :3]),
                                                 H_world_2_rgb[:3, 3])
        # Sometimes a few msgs from start and end go missing. Hence, we need to check if the timestamp exists in the vicon_object_data
        if str(v['timestamp']) not in vicon_object_data.keys():
            print('timestamp not found in vicon_object_data')
            continue
        t_x = vicon_object_data[str(v['timestamp'])]['translation'][0]
        t_y = vicon_object_data[str(v['timestamp'])]['translation'][1]
        t_z = vicon_object_data[str(v['timestamp'])]['translation'][2]
        r_x = vicon_object_data[str(v['timestamp'])]['rotation'][0]
        r_y = vicon_object_data[str(v['timestamp'])]['rotation'][1]
        r_z = vicon_object_data[str(v['timestamp'])]['rotation'][2]
        r_w = vicon_object_data[str(v['timestamp'])]['rotation'][3]
        rotation = R.from_quat([r_x, r_y, r_z, r_w]).as_matrix()
        # world_2_object are the recorded vicon values for the object
        H_world_2_object = np.eye(4)
        H_world_2_object[:3, :3] = rotation
        H_world_2_object[:3, 3] = [t_x, t_y, t_z]

        H_rgb_2_object = np.matmul(H_rgb_2_world, H_world_2_object)
        t_rgb_2_object = H_rgb_2_object[:3, 3]

        # project object (x,y,z) in rgb coordinate to cam1 coordinate
        H_cam1_2_object = np.matmul(H_cam1_2_rgb, H_rgb_2_object)
        t_cam1_2_object = H_cam1_2_object[:3, 3]
        H_cam2_2_object = np.matmul(H_cam2_cam1, H_cam1_2_object)
        t_cam2_2_object = H_cam2_2_object[:3, 3]
        H_cam1_2_vicon = np.matmul(H_cam1_2_rgb, H_rgb_2_world)
        H_cam2_2_vicon = np.matmul(H_cam2_cam1, H_cam1_2_vicon)
        transformations[str(vicon_data_camera_sys[str(i)]['timestamp'])] = {'H_rgb_2_vicon': H_rgb_2_world.tolist(),
                                                           'H_rgb_2_object': H_rgb_2_object.tolist(),
                                                           'H_world_2_cam_vicon': H_world_2_cam_vicon.tolist(),
                                                           't_rgb_2_object': t_rgb_2_object.tolist(),
                                                           'H_cam1_2_object': H_cam1_2_object.tolist(),
                                                           'H_cam2_2_object': H_cam2_2_object.tolist(),
                                                            'H_cam1_2_vicon': H_cam1_2_vicon.tolist(),
                                                            'H_cam2_2_vicon': H_cam2_2_vicon.tolist(),
                                                           'rotation': rotation.tolist(),
                                                           'timestamp': str(v['timestamp'])
                                                           }
    with open(path + '_transformations.json', 'w') as json_file:
        json.dump(transformations, json_file, indent=2)
    print('saved transformations data')

def save_transformations(vicon_2_camera_sys_data, H_cam_sys_2_rgb, vicon_object_data, H_rgb_2_left, H_right_2_rgb, path, H_vicon_to_object_uncalibrated_to_calibrated):
    transformations = {}
    for i, v in vicon_2_camera_sys_data.items():
        # TODO:what does this do
        #if i == str(len(vicon_2_camera_sys_data) - 1):
        #    continue

        # H_vicon_2_camera_sys - make homogeneous transformation matrix
        rot = vicon_2_camera_sys_data[str(i)]['rotation'] # get rotation matrix from quaternion
        H = np.eye(4)
        H[:3, :3] = R.from_quat(rot).as_matrix()
        H[:3, 3] = vicon_2_camera_sys_data[str(i)]['translation']
        H_vicon_2_camera_sys = H

        # make homogeneous transformation matrix from vicon to rgb optical frame
        H_world_2_rgb = np.matmul(H_vicon_2_camera_sys, H_cam_sys_2_rgb)
        #H_world_2_right = np.matmul(H_vicon_2_camera_sys, H_cam_sys_2_right)
        # H_right_2_world = np.eye(4)
        #H_right_2_world[:3, :3] = np.transpose(H_world_2_right[:3, :3])
        #H_right_2_world[:3, 3] = -np.matmul(np.transpose(H_world_2_right[:3, :3]), H_world_2_right[:3, 3])

        # invert
        H_rgb_2_world = np.eye(4)
        H_rgb_2_world[:3, :3] = np.transpose(H_world_2_rgb[:3, :3])
        H_rgb_2_world[:3, 3] = -np.matmul(np.transpose(H_world_2_rgb[:3, :3]), H_world_2_rgb[:3, 3])

        # Sometimes a few msgs from start and end go missing. Hence, we need to check if the timestamp exists in the vicon_object_data
        if str(v['timestamp']) not in vicon_object_data.keys():
            print('timestamp not found in vicon_object_data')
            continue
        t_x = vicon_object_data[str(v['timestamp'])]['translation'][0]
        t_y = vicon_object_data[str(v['timestamp'])]['translation'][1]
        t_z = vicon_object_data[str(v['timestamp'])]['translation'][2]
        r_x = vicon_object_data[str(v['timestamp'])]['rotation'][0]
        r_y = vicon_object_data[str(v['timestamp'])]['rotation'][1]
        r_z = vicon_object_data[str(v['timestamp'])]['rotation'][2]
        r_w = vicon_object_data[str(v['timestamp'])]['rotation'][3]
        rotation = R.from_quat([r_x, r_y, r_z, r_w]).as_matrix()
        # world_2_object are the recorded vicon values for the object
        H_world_2_object = np.eye(4)
        H_world_2_object[:3, :3] = rotation
        H_world_2_object[:3, 3] = [t_x, t_y, t_z]

        # tranform H_world_2_object to correct the vicon centre for object and the geometric centre.
        H_world_2_object_corrected = np.matmul(H_world_2_object, H_vicon_to_object_uncalibrated_to_calibrated)

        H_rgb_2_object = np.matmul(H_rgb_2_world, H_world_2_object_corrected)
        #H_right_2_object = np.matmul(H_right_2_world, H_world_2_object)

        # project object (x,y,z) in rgb coordinate to cam1 coordinate
        H_left_2_object = np.matmul(np.linalg.inv(H_rgb_2_left), H_rgb_2_object)
        H_right_2_object = np.matmul(H_right_2_rgb, H_rgb_2_object)
        transformations[str(vicon_2_camera_sys_data[str(i)]['timestamp'])] = {
            'H_rgb_2_vicon': H_rgb_2_world.tolist(),
            'H_vicon_2_camera_sys': H_vicon_2_camera_sys.tolist(),
            'H_rgb_2_object': H_rgb_2_object.tolist(),
            'H_left_2_object': H_left_2_object.tolist(),
            'H_right_2_object': H_right_2_object.tolist(),
            'timestamp': str(v['timestamp'])
        }
    with open(path + '_transformations.json', 'w') as json_file:
        json.dump(transformations, json_file, indent=2)
    print('saved transformations data')


def compute_cam_2_obj(vicon_data_camera_sys, H_cam_vicon_2_rgb, vicon_object_data, vertices, H_cam1_2_rgb, H_cam2_cam1, camera):
    transformations = {}
    for i, v in vicon_data_camera_sys.items():
        if i == str(len(vicon_data_camera_sys) - 1):
            continue
        vicon_cam_translation = vicon_data_camera_sys[str(i)]['translation']
        vicon_cam_rotation_quat = vicon_data_camera_sys[str(i)]['rotation']

        # get rotation matrix from quaternion
        rotation_vicon_cam = R.from_quat(vicon_cam_rotation_quat).as_matrix()
        # make homogeneous transformation matrix
        H_world_2_cam_vicon = np.eye(4)
        H_world_2_cam_vicon[:3, :3] = rotation_vicon_cam
        H_world_2_cam_vicon[:3, 3] = vicon_cam_translation

        # make homogeneous transformation matrix from vicon to camera optical frame
        H_world_2_rgb = np.matmul(H_world_2_cam_vicon, H_cam_vicon_2_rgb)

        # invert H_vicon_2_cam_optical to get H_cam_optical_2_vicon
        H_rgb_2_world = np.eye(4)
        H_rgb_2_world[:3, :3] = np.transpose(H_world_2_rgb[:3, :3])
        H_rgb_2_world[:3, 3] = -np.matmul(np.transpose(H_world_2_rgb[:3, :3]),
                                                 H_world_2_rgb[:3, 3])
        t_x = vertices[0]
        t_y = vertices[1]
        t_z = vertices[2]
        r_x = vicon_object_data[str(v['timestamp'])]['rotation'][0]
        r_y = vicon_object_data[str(v['timestamp'])]['rotation'][1]
        r_z = vicon_object_data[str(v['timestamp'])]['rotation'][2]
        r_w = vicon_object_data[str(v['timestamp'])]['rotation'][3]
        rotation = R.from_quat([r_x, r_y, r_z, r_w]).as_matrix()
        # world_2_object are the recorded vicon values for the object
        H_world_2_object = np.eye(4)
        H_world_2_object[:3, :3] = rotation
        H_world_2_object[:3, 3] = [t_x, t_y, t_z]
        H_rgb_2_object = np.matmul(H_rgb_2_world, H_world_2_object)
        H_cam1_2_object = np.matmul(H_cam1_2_rgb, H_rgb_2_object)
        H_cam2_2_object = np.matmul(H_cam2_cam1, H_cam1_2_object)
        if camera == 'rgb':
            return H_rgb_2_object
        elif camera == 'cam1':
            # project object (x,y,z) in rgb coordinate to cam1 coordinate
            return H_cam1_2_object
        else:
            return H_cam2_2_object


def get_translated_points_vertice(object_id, vertices, points_3d, object_len_z, object_len_x, object_len_y):
    if object_id == 1:
        #rotate the object by 90 degrees
        translation_vector = np.array([object_len_x/2000, object_len_y/2000, object_len_z/2000 + 0.025])
        vertices -= translation_vector
        points_3d -= translation_vector

    if object_id == 2:
        translation_vector = np.array([object_len_x/2000, -object_len_y/2000, (object_len_z)/2000 + 0.04])
        vertices -= translation_vector
        points_3d -= translation_vector

    if object_id == 3:
        translation_vector = np.array([object_len_x/2000, -object_len_y/2000, (object_len_z)/2000 + 0.025])
        vertices -= translation_vector
        points_3d -= translation_vector

    if object_id == 4:
        translation_vector = np.array([object_len_x/2000, -object_len_y/2000, object_len_z/2000 + 0.04])
        vertices -= translation_vector
        points_3d -= translation_vector

    if object_id == 5:
        translation_vector = np.array([object_len_x/2000, object_len_y/2000, object_len_z / 2000 - 0.04])
        vertices -= translation_vector
        points_3d -= translation_vector

    if object_id == 6:
        translation_vector = np.array([-object_len_x/2000, 0, object_len_z/2000])
        vertices -= translation_vector
        points_3d -= translation_vector

    if object_id == 7:
        translation_vector = np.array([object_len_x/2000, -object_len_y/2000, object_len_z / 2000])
        vertices -= translation_vector
        points_3d -= translation_vector

    if object_id == 8:
        translation_vector = np.array([-object_len_x/2000, object_len_y/2000, object_len_z / 2000])
        vertices -= translation_vector
        points_3d -= translation_vector

    if object_id == 9:
        translation_vector = np.array([object_len_x/2000, -object_len_y/2000, object_len_z / 2000])
        vertices -= translation_vector
        points_3d -= translation_vector

    if object_id == 10:
        translation_vector = np.array([object_len_x/2000, object_len_y/2000, object_len_z / 2000])
        vertices -= translation_vector
        points_3d -= translation_vector

    if object_id == 11:
        translation_vector = np.array([object_len_x/2000, object_len_y/2000, object_len_z / 2000 + 0.02])
        vertices -= translation_vector
        points_3d -= translation_vector

    if object_id == 12:
        translation_vector = np.array([-object_len_x/2000, object_len_y/2000, object_len_z / 2000])
        vertices -= translation_vector
        points_3d -= translation_vector

    if object_id == 13:
        translation_vector = np.array([object_len_x/2000, object_len_y/2000 + 0.03, object_len_z / 2000 ])
        vertices -= translation_vector
        points_3d -= translation_vector

    if object_id == 14:
        translation_vector = np.array([-object_len_x/2000, -object_len_y/2000, object_len_z / 2000 + 0.015])
        vertices -= translation_vector
        points_3d -= translation_vector

    if object_id == 15:
        translation_vector = np.array([object_len_x/2000 + 0.02, object_len_y/2000 + 0.02, -(object_len_z) / 2000 - 0.02])
        vertices -= translation_vector
        points_3d -= translation_vector

    if object_id == 16:
        translation_vector = np.array([object_len_x/2000, -object_len_y/2000, object_len_z / 2000 + 0.02])
        vertices -= translation_vector
        points_3d -= translation_vector

    return vertices, points_3d

def save_bbox_values(output_dir, object_2d_transform_points, timestamp):
    #################################### Exporting bounding box and pose values to a json file ####################################
    # transform object_3d_transform_points of size (8,1,2) to (8,2)
    object_2d_transform_points = object_2d_transform_points.reshape(-1, 2)
    # compute xmin, xmax, ymin, ymax, zmin, zmax
    xmin = float(np.min(object_2d_transform_points[:, 0]))
    xmax = float(np.max(object_2d_transform_points[:, 0]))
    ymin = float(np.min(object_2d_transform_points[:, 1]))
    ymax = float(np.max(object_2d_transform_points[:, 1]))

    #Bbox = np.array([timestamp, xmin, xmax, ymin, ymax, zmin, zmax])
    #Bbox = {'timestamp': timestamp, 'xmin': Bbox[1], 'xmax': Bbox[2], 'ymin': Bbox[3], 'ymax': Bbox[4], 'zmin': Bbox[5], zmax: Bbox[6]}
    #Bbox = np.array([timestamp, xmin, xmax, ymin, ymax])
    Bbox = {'timestamp': timestamp, 'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}
    # append the Bbox values to a json file row wise
    file = output_dir + "bounding_box_labels_2d.json"
    with open(file, 'a') as json_file:
        json_file.write(json.dumps(Bbox) + '\n')

def save_bbox_values_3D(object_name, output_dir, timestamp, object_3d_transform_vertices, object_2d_vertices, img_cam, time_rgb, root_dir):
    #################################### Exporting bounding box and pose values to a json file ####################################
    # transform object_3d_transform_points of size (8,1,2) to (8,2)
    object_3d_transform_vertices = object_3d_transform_vertices.reshape(-1, 3)
    # compute xmin, xmax, ymin, ymax, zmin, zmax
    xmin = float(np.min(object_3d_transform_vertices[:, 0]))
    xmax = float(np.max(object_3d_transform_vertices[:, 0]))
    ymin = float(np.min(object_3d_transform_vertices[:, 1]))
    ymax = float(np.max(object_3d_transform_vertices[:, 1]))
    zmin = np.min(object_3d_transform_vertices[:, 2])
    zmax = np.max(object_3d_transform_vertices[:, 2])
    status = 'good'

    # check if the bbox is too large or too small
    # x axis is the horizontal, y is the height of the human and z is the depth. Here the y threshold is 2.0 for taller human and 1.9 for shorter human
    # z for taller is 1.6 and shorter is 1.5
    if object_name == 'human':
        if xmax - xmin > 1.55 or xmax - xmin < 0.30 or ymax - ymin > 1.9 or ymax - ymin < 0.40 or zmax - zmin > 1.5 or zmax - zmin < 0.15:
            print('bbox size is too large', xmax - xmin, ymax - ymin, zmax - zmin)
            status = 'bad'

    # Create a blank mask (same size as image, single channel)
    #time_rgb = 0
    # create masks only for visible part of the objects for rgb images
    if time_rgb != 0:
        object_mask = np.zeros(img_cam.shape[:2], dtype=np.uint8)
        hull = cv2.convexHull(np.array(object_2d_vertices))
        cv2.fillPoly(object_mask, [hull], 255)
        # create folder to save masks
        save_dir = root_dir + "/masks_rgb_" + object_name + "/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Save or display the mask
        #cv2.imwrite(save_dir + "mask_" + str(time_rgb) + ".png", object_mask)
        # load masks for human
        data = np.load(
            root_dir + '/output_masks_human_img/' + str(time_rgb) + '.npy')
        # Convert data from (1,1536,2048) to (1536,2038. visualise it in a camera frame of size 1536x2048
        human_mask = data[0]
        #human_maksk has values as true and false. Convert this to 1 and 0
        human_mask = human_mask.astype(np.uint8)  # Convert True -> 1, False -> 0

        if os.path.exists(root_dir + '/output_masks_hupwagen_img/'):
            data_hupwagen = np.load(
                root_dir + '/output_masks_hupwagen_img/' + str(time_rgb) + '.npy')
            mask_hupwagen = data_hupwagen[0]
            mask_hupwagen = mask_hupwagen.astype(np.uint8)
            # mask_hupwagen = cv2.imread(root_dir + "/output_masks_hupwagen_img/" + str(time_rgb) + '.jpg', cv2.IMREAD_GRAYSCALE)
            object_mask = object_mask * (1 - mask_hupwagen)
        # Subtract human mask from object mask
        visible_object_mask = object_mask * (1 - human_mask)  # Remove overlapping region
        #save the mask as npy file
        np.save(save_dir + "mask_" + str(time_rgb) + "_visible_object.npy", visible_object_mask)
        # Save the mask
        cv2.imwrite(save_dir + "mask_" + str(time_rgb) + "_visible_object.jpg", visible_object_mask)
    #Bbox = np.array([timestamp, xmin, xmax, ymin, ymax])
    Bbox = {'timestamp': timestamp, 'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'zmin': zmin, 'zmax': zmax, 'status': status}
    # append the Bbox values to a json file row wise
    file = output_dir + "bounding_box_labels_3d.json"
    with open(file, 'a') as json_file:
        json_file.write(json.dumps(Bbox) + '\n')

def project_points_to_image_plane(obj_name, H_cam_2_object, k, rgb_t, img_path, points_3d, vertices,
                                  camera_matrix, distortion_coefficients, output_dir, root, obj_iter =0, save = False, human = False):
    #points_2d_cam1 = cv2.projectPoints(np.array([t_cam_2_point]), np.eye(3), np.zeros(3), camera_matrix,
     #                                  distortion_coefficients)
    #points_2d_cam1 = np.round(points_2d_cam1[0]).astype(int)
    # check if string rgb exists in rgb_image_path

    img_temp_cam = cv2.imread(img_path)
    # rectify the image with the intrinsic parameters only first time
    if obj_iter == 0:
        img_temp_cam = cv2.undistort(img_temp_cam, camera_matrix, distortion_coefficients)

    #img_temp_cam = cv2.circle(img_temp_cam, tuple(points_2d_cam1[0][0]), 5, (255, 0, 0), -1)
    object_3d_transform_points = np.matmul(H_cam_2_object, np.vstack((points_3d.T, np.ones(points_3d.shape[0]))))[
                                 :3, :].T
    object_3d_transform_vertices = np.matmul(H_cam_2_object, np.vstack((vertices.T, np.ones(vertices.shape[0]))))[
                                   :3, :].T
    center_3d = np.mean(object_3d_transform_points, axis=0)
    # create 3d bbox using vertices. object_3d_transform_vertices are the 3D vertices of the 3D object


    # project 3d points to image plane
    object_2d_points, _ = cv2.projectPoints(object_3d_transform_points, np.eye(3), np.zeros(3), camera_matrix,
                                         distortion_coefficients)
    object_2d_points = np.round(object_2d_points).astype(int)
    object_2d_vertices, _ = cv2.projectPoints(object_3d_transform_vertices, np.eye(3), np.zeros(3), camera_matrix,
                                           distortion_coefficients)
    object_2d_vertices = np.round(object_2d_vertices).astype(int)

    center_2d, _ = cv2.projectPoints(np.array([center_3d]), np.eye(3), np.zeros(3), camera_matrix,
                                     distortion_coefficients)
    center_2d = center_2d[0, 0]
    if save:
        save_bbox_values(output_dir, object_2d_vertices, k)
        save_bbox_values_3D(obj_name, output_dir, k, object_3d_transform_vertices, object_2d_vertices, img_temp_cam, rgb_t, root)
        save_pose(H_cam_2_object, center_3d, output_dir, k)

    # create a mask
    mask = np.zeros_like(img_temp_cam)
    for point in object_2d_points:
        mask = cv2.circle(mask, tuple(point[0].astype(int)), 2, (255, 255, 255), -1)
    # fill the mask with the projected points
    #img_temp_cam = cv2.addWeighted(img_temp_cam, 1, mask, 0.3, 0)
    if rgb_t != 0:
        line_weight = 2
    else:
        line_weight = 1
    if human:
        for i in range(4):
            img_temp_cam = cv2.line(img_temp_cam, tuple(object_2d_vertices[i][0].astype(int)),
                                    tuple(object_2d_vertices[(i + 1) % 4][0].astype(int)), (0, 255, 0), 3)
            img_temp_cam = cv2.line(img_temp_cam, tuple(object_2d_vertices[i + 4][0].astype(int)),
                                    tuple(object_2d_vertices[(i + 1) % 4 + 4][0].astype(int)), (0, 255, 0), 3)
            img_temp_cam = cv2.line(img_temp_cam, tuple(object_2d_vertices[i][0].astype(int)),
                                    tuple(object_2d_vertices[i + 4][0].astype(int)), (0, 255, 0), 3)
        # draw the 3D bounding box using vertices in vert

    else:
        for i in range(4):
            img_temp_cam = cv2.line(img_temp_cam, tuple(object_2d_vertices[i][0].astype(int)),
                                    tuple(object_2d_vertices[(i + 1) % 4][0].astype(int)), (0, 255, 0), line_weight)
            img_temp_cam = cv2.line(img_temp_cam, tuple(object_2d_vertices[i + 4][0].astype(int)),
                                    tuple(object_2d_vertices[(i + 1) % 4 + 4][0].astype(int)), (0, 255, 0), line_weight)
            img_temp_cam = cv2.line(img_temp_cam, tuple(object_2d_vertices[i][0].astype(int)),
                                    tuple(object_2d_vertices[i + 4][0].astype(int)), (0, 255, 0), line_weight)



    '''
    # create 2D BBox using vertices. object_2d_vertices are the 3D vertices of the 3D object
    # Convert 3D BBox to 2Bbox vertices
    img_temp_cam = cv2.rectangle(img_temp_cam, tuple(object_2d_vertices[0][0].astype(int)),
                                    tuple(object_2d_vertices[2][0].astype(int)), (0, 255, 0), 3)
    # show the points on the image
    for point in object_2d_points:
        img_temp_cam = cv2.circle(img_temp_cam, tuple(point[0].astype(int)), 3, (0, 0, 255), -1)
    '''
    '''
    for point in object_2d_vertices:
        img_temp_cam = cv2.circle(img_temp_cam, tuple(point[0].astype(int)), 3, (0, 0, 255), -1)
    '''
    img_temp_cam = cv2.circle(img_temp_cam, tuple(center_2d.astype(int)), 5, (0, 255, 0), -1)
    #cv2.imshow('img', cv2.resize(img_temp_cam, (0, 0), fx=0.5, fy=0.5))
    return img_temp_cam


def save_pose(H_cam_optical_2_point, center_3d, output_dir, timestamp):
    rotation = H_cam_optical_2_point[:3, :3]
    rotmat = R.from_matrix(rotation)
    euler_angles = rotmat.as_euler('xyz', degrees=True)
    pose = np.concatenate((center_3d, euler_angles)).tolist()
    # save above to a json file
    data = [timestamp, pose]

    # append the Bbox values to a json file row wise
    file = output_dir + "_pose.json"
    with open(file, 'a') as json_file:
        json_file.write(json.dumps(data) + '\n')

def find_closest_timestamp(data, k):
    if str(k) in data.keys():  # Check if k exists in data
        return k
    elif data:  # Ensure data is not empty before searching
        print('Timestamp',k,'is missing. Probably a drop in event camera frame')
        closest_k = min(data.keys(), key=lambda x: abs(int(x) - k))  # Find the closest timestamp
        return closest_k
    else:
        return None  # Return None if data is empty

def check_human_bbox_data(data, k, previous_t):
    k = find_closest_timestamp(data, k)
    # compare all the x,y and z values at current time with the previous time. If the mod of difference is greater than 0.6 then keep the previous value
    # else keep the current value
    threshold = 6000
    if abs(data[str(k)]['min_x'] - data[str(previous_t)]['min_x']) > threshold:
        data[str(k)]['min_x'] = data[str(previous_t)]['min_x']
        print('inconsistent min_x')
    if abs(data[str(k)]['max_x'] - data[str(previous_t)]['max_x']) > threshold:
        data[str(k)]['max_x'] = data[str(previous_t)]['max_x']
        print('inconsistent max_x')
    if abs(data[str(k)]['min_y'] - data[str(previous_t)]['min_y']) > threshold:
        data[str(k)]['min_y'] = data[str(previous_t)]['min_y']
        print('inconsistent min_y')
    if abs(data[str(k)]['max_y'] - data[str(previous_t)]['max_y']) > threshold:
        data[str(k)]['max_y'] = data[str(previous_t)]['max_y']
        print('inconsistent max_y')
    if abs(data[str(k)]['min_z'] - data[str(previous_t)]['min_z']) > threshold:
        data[str(k)]['min_z'] = data[str(previous_t)]['min_z']
        print('inconsistent min_z')
    if abs(data[str(k)]['max_z'] - data[str(previous_t)]['max_z']) > threshold:
        data[str(k)]['max_z'] = data[str(previous_t)]['max_z']
        print('inconsistent max_z')
    return data,k

def get_BBox_vertices(data, k):
    # import json. It contains 3D bbox values of human as xmin xmax ymin ymax zmin zmax

    # extract xmin xmax ymin ymax zmin zmax for timestamp value = k
    vertices = data[str(k)]['min_x'], data[str(k)]['max_x'], data[str(k)]['min_y'], data[str(k)]['max_y'], data[str(k)]['min_z'], data[str(k)]['max_z']
    # get coordinates of the 3D bbox using above values
    vertices = np.array([[vertices[0], vertices[2], vertices[4]],
                          [vertices[0], vertices[3], vertices[4]],
                          [vertices[1], vertices[3], vertices[4]],
                          [vertices[1], vertices[2], vertices[4]],
                          [vertices[0], vertices[2], vertices[5]],
                          [vertices[0], vertices[3], vertices[5]],
                          [vertices[1], vertices[3], vertices[5]],
                          [vertices[1], vertices[2], vertices[5]]])
    # convert to metres
    vertices = vertices / 1000
    return vertices

def get_vertices(vert):
    xmin, ymin, zmin = np.min(vert, axis=0)
    xmax, ymax, zmax = np.max(vert, axis=0)

    # Define the 8 vertices of the bounding box
    vert = np.array([
        [xmin, ymin, zmin],  # Bottom-left-front
        [xmax, ymin, zmin],  # Bottom-right-front
        [xmax, ymax, zmin],  # Top-right-front
        [xmin, ymax, zmin],  # Top-left-front
        [xmin, ymin, zmax],  # Bottom-left-back
        [xmax, ymin, zmax],  # Bottom-right-back
        [xmax, ymax, zmax],  # Top-right-back
        [xmin, ymax, zmax]  # Top-left-back

    ])
    return vert
