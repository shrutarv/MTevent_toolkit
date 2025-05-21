import numpy as np
import json

import numpy as np
import json
import copy
import os

root = '/media/eventcamera/event_data/dataset_25_march_zft/scene75'

def apply_simple_threshold_filter(data, pose, threshold=0.5):
    smoothed_data = []

    # Initialize the previous bbox to be the first frame
    prev_bbox = data[0]
    smoothed_data.append(prev_bbox)

    for i in range(1, len(data)):
        current_bbox = data[i]
        smoothed_bbox = {}
        for coord in ["xmin", "xmax"]:
            if abs(current_bbox[coord] - prev_bbox[coord]) > threshold:
                smoothed_bbox[coord] = prev_bbox[coord]  # Retain previous value
                print(f"Outlier detected at index {i} for {coord}: {current_bbox[coord]} -> {prev_bbox[coord]}")
            else:
                smoothed_bbox[coord] = current_bbox[coord]  # Use current value

        for coord in ["ymin", "ymax", "zmin", "zmax"]:

            if abs(current_bbox[coord] - prev_bbox[coord]) > 0.25:
                smoothed_bbox[coord] = prev_bbox[coord]  # Retain previous value
                print(f"Outlier detected at index {i} for {coord}: {current_bbox[coord]} -> {prev_bbox[coord]}")
            else:
                smoothed_bbox[coord] = current_bbox[coord]  # Use current value
        # check the distance between the xmin and xmax
        if abs(smoothed_bbox["xmax"] - smoothed_bbox["xmin"]) > 1.30:
            smoothed_bbox["xmax"] = pose[i][1][0] + 0.5
            smoothed_bbox["xmin"] = pose[i][1][0] - 0.5

        # Add the timestamp from the current frame
        smoothed_bbox["timestamp"] = current_bbox["timestamp"]

        # Add the smoothed bbox to the list
        smoothed_data.append(smoothed_bbox)

        # Update the previous bbox for the next iteration
        prev_bbox = smoothed_bbox
    return smoothed_data

def apply_separate_threshold_filter(data, bounding_boxes, pose, camera, threshold=0.5):
    smoothed_data = []

    # Initialize the previous bbox to be the first frame
    prev_bbox = data[0]
    smoothed_data.append(prev_bbox)

    for i in range(1, len(data)):

        current_bbox = data[i]
        smoothed_bbox = {}

        timestamp_bbox = current_bbox["timestamp"]
        if camera == 'left':
            X = bounding_boxes[i]['X_left']
            Y = bounding_boxes[i]['Y_left']
            Z = bounding_boxes[i]['Z_left']
        elif camera == 'right':
            X = bounding_boxes[i]['X_right']
            Y = bounding_boxes[i]['Y_right']
            Z = bounding_boxes[i]['Z_right']
        else:
            X = bounding_boxes[i]['X']
            Y = bounding_boxes[i]['Y']
            Z = bounding_boxes[i]['Z']

        for coord in ["xmin", "xmax"]:
            # Check if the current coordinate is an outlier compared to the previous one
            if abs(X - current_bbox[coord]) > 0.6 or abs(X - current_bbox[coord]) <  0.6:
                temp_bbox = copy.deepcopy(data[i])
                temp_bbox["xmin"] = X - 0.3
                temp_bbox["xmax"] = X + 0.70
                smoothed_bbox[coord] = temp_bbox[coord]
            else:
                smoothed_bbox[coord] = current_bbox[coord]

        for coord in ["ymin", "ymax", "zmin", "zmax"]:

            if abs(current_bbox[coord] - prev_bbox[coord]) > 0.7:
                smoothed_bbox[coord] = prev_bbox[coord]  # Retain previous value
                #print(f"Outlier detected at index {i} for {coord}: {current_bbox[coord]} -> {prev_bbox[coord]}")
            else:
                smoothed_bbox[coord] = current_bbox[coord]  # Use current value
        # check the distance between the xmin and xmax
        if abs(smoothed_bbox["xmax"] - smoothed_bbox["xmin"]) > 1.30 or abs(smoothed_bbox["xmax"] - smoothed_bbox["xmin"]) < 0.40:
            print(i, "xmax and xmin are not in the range", smoothed_bbox["xmax"], smoothed_bbox["xmin"])
            smoothed_bbox["xmax"] = X + 0.5
            smoothed_bbox["xmin"] = X - 0.5

        if abs(smoothed_bbox["zmax"] - smoothed_bbox["zmin"]) > 1.00 or abs(smoothed_bbox["zmax"] - smoothed_bbox["zmin"]) < 0.30:
            print(i, "zmax and zmin are not in the range", smoothed_bbox["zmax"], smoothed_bbox["zmin"])
            #smoothed_bbox["zmax"] = bounding_boxes[i]['Z_left'] + 0.5
            smoothed_bbox["zmin"] = Z - 0.4

        # Add the timestamp from the current frame
        smoothed_bbox["timestamp"] = current_bbox["timestamp"]
        # if the original bboxes are good then keep the original values
        if data[i]['status'] == 'good':
            smoothed_bbox["xmin"] = current_bbox["xmin"]
            smoothed_bbox["xmax"] = current_bbox["xmax"]
            smoothed_bbox["ymin"] = current_bbox["ymin"]
            smoothed_bbox["ymax"] = current_bbox["ymax"]
            smoothed_bbox["zmin"] = current_bbox["zmin"]
            smoothed_bbox["zmax"] = current_bbox["zmax"]
        # get 6D pose from the bounding box. 3 for translation and 3 Euler angles for rotation
        # Do for pose
        pose[i][1][0] = smoothed_bbox["xmin"] + (smoothed_bbox["xmax"] - smoothed_bbox["xmin"]) / 2
        pose[i][1][1] = smoothed_bbox["ymin"] + (smoothed_bbox["ymax"] - smoothed_bbox["ymin"]) / 2
        pose[i][1][2] = smoothed_bbox["zmin"] + (smoothed_bbox["zmax"] - smoothed_bbox["zmin"]) / 2

        # Add the smoothed bbox to the list
        smoothed_data.append(smoothed_bbox)

        # Update the previous bbox for the next iteration
        prev_bbox = smoothed_bbox

    return smoothed_data, pose

def check_pose(pose):
    for i in range(1, len(pose)):
        # Check if the current coordinate is an outlier compared to the previous one
        if abs(pose[i][1][0] - pose[i-1][1][0]) > 0.5 or abs(pose[i][1][0] - pose[i-1][1][0]) < - 0.5:
            print(f"Outlier detected at index {i} for pose: {pose[i][1][0]} -> {pose[i - 1][1][0]}")
            pose[i][1][0] = pose[i-1][1][0]

    return pose

def check_bbox(bbox):
    for i in range(1, len(bbox)):
        # Check if the current coordinate is an outlier compared to the previous one
        if abs(bbox[i]['X'] - bbox[i-1]['X']) > 0.5 or abs(bbox[i]['X'] - bbox[i-1]['X']) < - 0.5:
            print(f"Outlier detected at index {i} for bbox: {bbox[i]['X']} -> {bbox[i - 1]['X']}")
            bbox[i]['X'] = bbox[i-1]['X']

    return bbox


# bbox_3d_data contains the 3D bounsing boxes which are converted from 2D bouncing boxes in rgb using script mask to bbox
with open(
        root + '/annotation_human/human_ec_left_bounding_box_labels_3d.json',
        'r') as file:
    bbox_3d_data_left = [json.loads(line) for line in file]
t_left = [entry['timestamp'] for entry in bbox_3d_data_left[:]]

with open(
        root + '/annotation_human/human_ec_right_bounding_box_labels_3d.json',
        'r') as file:
    bbox_3d_data_right = [json.loads(line) for line in file]
t_right = [entry['timestamp'] for entry in bbox_3d_data_right[:]]

with open(
        root + '/annotation_human/human_rgb_bounding_box_labels_3d.json',
        'r') as file:
    bbox_3d_data_rgb = [json.loads(line) for line in file]
t_rgb = [entry['timestamp'] for entry in bbox_3d_data_rgb[:]]

with open(root + '/output_masks_human_img/bounding_boxes.json', 'r') as file:
    bounding_boxes_seg = json.load(file)
t_bbox_seg = [entry['timestamp'] for entry in bounding_boxes_seg[:]]

with open(root + '/annotation_human/human_ec_left__pose.json', 'r') as file:
    pose_left = [json.loads(line) for line in file]

with open(root + '/annotation_human/human_ec_right__pose.json', 'r') as file:
    pose_right = [json.loads(line) for line in file]

with open(root + '/annotation_human/human_rgb__pose.json', 'r') as file:
    pose_rgb = [json.loads(line) for line in file]

bbox_seg = []

#bounding_boxes = check_bbox(bounding_boxes)
# find the nearest timestamp from t_bbox_seg to the timestamp in bbox_3d_data_left
for i in range(len(bbox_3d_data_right)):
    # find the nearest timestamp to timestamp_bbox in time array
    nearest_timestamp = min(t_bbox_seg, key=lambda x: abs(x - t_left[i]))
    bbox_seg.append(bounding_boxes_seg[t_bbox_seg.index(nearest_timestamp)])


smoothed_bboxes_left, pose_l = apply_separate_threshold_filter(bbox_3d_data_left, bbox_seg, pose_left, 'left', threshold=1)
smoothed_bboxes_right, pose_r = apply_separate_threshold_filter(bbox_3d_data_right, bbox_seg, pose_right, 'right',  threshold=1)
smoothed_bboxes_rgb, pose_rgb = apply_separate_threshold_filter(bbox_3d_data_rgb, bbox_seg, pose_rgb, 'rgb', threshold=1)

if not os.path.exists(root + '/smoothened/'):
    os.makedirs(root + '/smoothened/')

output_path_left = root + '/smoothened/human_ec_left_bounding_box_labels_3d_smooth.json'
output_path_right = root + '/smoothened/human_ec_right_bounding_box_labels_3d_smooth.json'
output_path_rgb = root + '/smoothened/human_rgb_bounding_box_labels_3d_smooth.json'
output_path_pose_left = root + '/smoothened/human_ec_left__pose_smooth.json'
output_path_pose_right = root + '/smoothened/human_ec_right__pose_smooth.json'
output_path_pose_rgb = root + '/smoothened/human_rgb__pose_smooth.json'

# Save to JSON
with open(output_path_left, "w") as f:
    json.dump(smoothed_bboxes_left, f, indent=2)
with open(output_path_right, "w") as f:
    json.dump(smoothed_bboxes_right, f, indent=2)
with open(output_path_rgb, "w") as f:
    json.dump(smoothed_bboxes_rgb, f, indent=2)
with open(output_path_pose_left, "w") as f:
    json.dump(pose_l, f, indent=2)
with open(output_path_pose_right, "w") as f:
    json.dump(pose_r, f, indent=2)
with open(output_path_pose_rgb, "w") as f:
    json.dump(pose_rgb, f, indent=2)

print("✅ Smoothed + filtered bboxes saved to filtered_bboxes.json")










