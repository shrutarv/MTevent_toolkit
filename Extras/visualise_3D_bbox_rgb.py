import json
import cv2
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

root = '/home/eventcamera/data/dataset/dataset_23_jan/scene3_1'
# Load JSON File
with open(root + "/annotation/rgb_big_klt__bounding_box_labels_3d.json", "r") as f:
    bbox = [json.loads(line) for line in f.readlines()]

image_list = os.listdir(root + '/rgb_jpg/')
with open(root + '/annotation/rgb_big_klt__pose.json') as p:
    pose = [json.loads(line) for line in p.readlines()]
# sort image list in ascending order
image_list.sort()
with open('/home/eventcamera/RGB_Event_cam_system/Annotation/camera_params.json', 'r') as file:
    data = json.load(file)
# cam1 is event camera left and cam2 is event camera right
K = np.array(data['camera_matrix'])
dist = np.array(data['distortion_coefficents'])

# Function to project 3D points to 2D
def project_3d_to_2d(points_3d, K,dist):
    projected_points = []

    for point in points_3d:
        #x, y, z = point
        #point_2d = np.dot(K, np.array([x, y, z]))  # Apply intrinsic matrix
        #point_2d /= point_2d[2]  # Convert to 2D
        point_2d, _ = cv2.projectPoints(point, np.eye(3), np.zeros(3), K,
                                         dist)
        point_2d = np.round(point_2d).astype(int)
        projected_points.append((int(point_2d[0][0][0]), int(point_2d[0][0][1])))  # Store as pixel coords
    return projected_points


count = 0
for image_name in image_list:
    # Load the image
    image = cv2.imread(os.path.join(root + '/rgb_jpg/', image_name))
    #image = cv2.imread(os.path.join(root + '/event_cam_left/e2calib/', str(bbox[count]['timestamp']) + '.png'))
    xmin = bbox[count]['xmin']
    ymin = bbox[count]['ymin']
    xmax = bbox[count]['xmax']
    ymax = bbox[count]['ymax']
    zmin = bbox[count]['zmin']
    zmax = bbox[count]['zmax']
    points_3d = [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymax, zmin), (xmin, ymax, zmin),
                 (xmin, ymin, zmax), (xmax, ymin, zmax), (xmax, ymax, zmax), (xmin, ymax, zmax)]
    euler = [pose[count][1][3], pose[count][1][4], pose[count][1][5]]
    # rotate the points_3d as per the euler angles XYZ in degrees
    # convert euler angles to radians
    rotation_matrix = R.from_euler('xyz', euler, degrees=True).as_matrix()
    # Rotate each vertex
    center = np.mean(points_3d, axis=0)

    # Translate to origin, apply rotation, then translate back
    # points_3d = (points_3d - center) @ rotation_matrix.T + center

    points_2d = project_3d_to_2d(points_3d, K, dist)
    edges = [(0,1), (1,2), (2,3), (3,0),  # Front face
             (4,5), (5,6), (6,7), (7,4),  # Back face
             (0,4), (1,5), (2,6), (3,7)]
    for (start, end) in edges:
        cv2.line(image, points_2d[start], points_2d[end], (0, 255, 0), 2)
    count += 1


    # Save as JPEG
    output_path = os.path.join(root,'/rgb_visualised/', image_name.replace('.png', '.jpg'))
    # if output path does not exist, create the path
    if not os.path.exists(root + '/rgb_visualised/'):
        os.makedirs(root + '/rgb_visualised/')
    cv2.imshow("3D Bounding Box Projection", image)
    cv2.waitKey(0)
    cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print('visualised for image', image_name)




# Show the result

cv2.destroyAllWindows()
