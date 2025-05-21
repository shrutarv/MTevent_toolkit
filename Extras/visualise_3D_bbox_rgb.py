import json
import cv2
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2

root = '/mnt/smbshare/scene73'
with open(root + '/output_masks_human_img/bounding_boxes.json', 'r') as file:
    bounding_boxes = json.load(file)

def draw_point(image, c, K, color=(0, 255, 0), thickness=2):
    x = bounding_boxes[c]['X_left']
    y = bounding_boxes[c]['Y_left']
    z = bounding_boxes[c]['Z_left']
    # Project 3D point to 2D
    X = np.array([x, y, z, 1])
    x = K @ X[:3]
    x = x[:2] / x[2]
    x = x.astype(int)
    # Draw the point on the image
    cv2.circle(image, tuple(x), 5, color, thickness)
    # Draw the point on the image
    #cv2.putText(image, str(c), tuple(x), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    # Draw the point on the image
    return image



def draw_3d_bbox_on_image(image, bbox, K, color=(0, 255, 0), thickness=2):
    """
    image: numpy image
    bbox: dict with xmin, xmax, ymin, ymax, zmin, zmax
    K: camera intrinsic matrix (3x3)
    """

    # Create 8 corners of the bounding box
    corners_3d = np.array([
        [bbox["xmin"], bbox["ymin"], bbox["zmin"]],
        [bbox["xmax"], bbox["ymin"], bbox["zmin"]],
        [bbox["xmax"], bbox["ymax"], bbox["zmin"]],
        [bbox["xmin"], bbox["ymax"], bbox["zmin"]],
        [bbox["xmin"], bbox["ymin"], bbox["zmax"]],
        [bbox["xmax"], bbox["ymin"], bbox["zmax"]],
        [bbox["xmax"], bbox["ymax"], bbox["zmax"]],
        [bbox["xmin"], bbox["ymax"], bbox["zmax"]],
    ])

    # Project to 2D (assuming no rotation/translation, just K * XYZ)
    points_2d = []
    for point in corners_3d:
        X = np.array([point[0], point[1], point[2], 1])
        x = K @ X[:3]
        x = x[:2] / x[2]
        points_2d.append(x.astype(int))

    # Convert to array for drawing
    points_2d = np.array(points_2d)

    # Define connections between 3D corners (12 lines)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
    ]

    for i, j in edges:
        pt1 = tuple(points_2d[i])
        pt2 = tuple(points_2d[j])
        cv2.line(image, pt1, pt2, color, thickness)

    return image



# Load JSON File
# /media/eventcamera/event_data/dataset_31_march_zft/scene56/annotation_human/human_ec_left_bounding_box_labels_3d.json
#with open(root + '/annotation_human/human_ec_left_bounding_box_labels_3d.json', 'r') as file:
#    bbox_3d = [json.loads(line) for line in file]
cam = 'left'
output_path = root + '/smoothened/'
# check if the output path exists
if not os.path.exists(output_path):
    os.makedirs(output_path)
if cam == 'rgb':
    with open(root + "/smoothened/human_" + cam + "_bounding_box_labels_3d_smooth.json", 'r') as file:
        bbox_3d = json.load(file)
else:
    with open(root + "/smoothened/human_ec_" + cam + "_bounding_box_labels_3d_smooth.json", "r") as f:
        bbox_3d = json.load(f)

path = root + '/event_images/' + cam + '/'
# load camera_params.json
with open(root + '/camera_params.json', 'r') as file:
    cam_params = json.load(file)
# Extract camera intrinsic matrix
if cam == 'left':
    K = np.array(cam_params['camera_matrix'])
elif cam == 'right':
    K = np.array(cam_params['camera_mtx_cam2'])
elif cam == 'rgb':
    K = np.array(cam_params['camera_mtx_cam1'])
    path = root + '/rgb/'


files = os.listdir(path)
# sort the files
files.sort()
modified_files = [int(file_name[:-4]) for file_name in files if file_name.endswith('.png') or file_name.endswith('.jpg')]


# delete all jpg or png files in the output path
for file_name in os.listdir(output_path):
    if file_name.endswith('.jpg') or file_name.endswith('.png'):
        os.remove(os.path.join(output_path, file_name))

if not os.path.exists(output_path):
    os.makedirs(output_path)
count = 0
for i in bbox_3d:

    # find the bbox with the nearest timestamp in the files
    nearest_timestamp = min(modified_files, key=lambda x: abs(x - i['timestamp']))
    # read the image
    if cam == 'rgb':
        img = cv2.imread(os.path.join(path, str(nearest_timestamp) + '.jpg'))
    else:
        img = cv2.imread(os.path.join(path, str(nearest_timestamp) + '.png'))

    image_with_box = draw_3d_bbox_on_image(img, i, K)
    image_with_box = draw_point(image_with_box, count, K)
    count += 1
    # save the image to /media/eventcamera/event_data/dataset_31_march_zft/scene56/smoothened/

    cv2.imwrite(os.path.join(output_path, str(i['timestamp']) + '.jpg'), image_with_box)

    #cv2.imshow("3D BBox Projection", image_with_box)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


