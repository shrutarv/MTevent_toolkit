# read a yaml file
import yaml
import numpy as np
import json

with open('/media/eventcamera/event_data/calibration/mar_20/images-camchain.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    # cam 0: ec left, cam1: rgb, cam2:ec right
    camchain = yaml.load(file, Loader=yaml.FullLoader)
    distortion_coefficients = camchain['cam0']['distortion_coeffs']
    camera_matrix = camchain['cam0']['intrinsics']
    H_rgb_2_left = camchain['cam1']['T_cn_cnm1']
    distortion_coeffs_cam1 = camchain['cam1']['distortion_coeffs']
    camera_mtx_cam1 = camchain['cam1']['intrinsics']
    H_right_2_rgb = camchain['cam2']['T_cn_cnm1']
    distortion_coeffs_cam2 = camchain['cam2']['distortion_coeffs']
    camera_mtx_cam2 = camchain['cam2']['intrinsics']

# camera matrix
camera_matrix = np.array([[camera_matrix[0], 0, camera_matrix[2]], [0, camera_matrix[1], camera_matrix[3]], [0, 0, 1]]).tolist()
#distortion_coefficients = np.array(distortion_coefficients)
camera_mtx_cam1 = np.array([[camera_mtx_cam1[0], 0, camera_mtx_cam1[2]], [0, camera_mtx_cam1[1], camera_mtx_cam1[3]], [0, 0, 1]]).tolist()
#distortion_coeffs_cam1 = np.array(distortion_coeffs_cam1)
camera_mtx_cam2 = np.array([[camera_mtx_cam2[0], 0, camera_mtx_cam2[2]], [0, camera_mtx_cam2[1], camera_mtx_cam2[3]], [0, 0, 1]]).tolist()
#distortion_coeffs_cam2 = np.array(distortion_coeffs_cam2)

# save above to a json file
data = {'camera_matrix': camera_matrix, 'distortion_coefficients': distortion_coefficients,
        'camera_mtx_cam1': camera_mtx_cam1, 'distortion_coeffs_cam1': distortion_coeffs_cam1,
        'camera_mtx_cam2': camera_mtx_cam2, 'distortion_coeffs_cam2': distortion_coeffs_cam2,
        'H_rgb_2_left': H_rgb_2_left, 'H_right_2_rgb': H_right_2_rgb}
with open('/home/eventcamera/RGB_Event_cam_system/Annotation/camera_params.json', 'w') as file:
    json.dump(data, file)