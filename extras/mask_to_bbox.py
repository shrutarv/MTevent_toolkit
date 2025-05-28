import os
import json
import numpy as np

root = "/media/eventcamera/event_data/dataset_25_march_zft/scene75/"
# Directory containing .npy files
input_dir = root + "output_masks_human_img/"
output_json = root + "output_masks_human_img/bounding_boxes.json"

data = []

def pixel_to_camera(x_pixel, y_pixel, Z, fx, fy, cx, cy):
    """Convert pixel coordinates to camera coordinates."""
    X = (x_pixel - cx) * Z / fx
    Y = (y_pixel - cy) * Z / fy
    return X, Y, Z

# sort input_dir
input_dir_files = os.listdir(input_dir)
input_dir_files.sort()
first = True
for filename in input_dir_files:
    if not filename.endswith(".npy"):
        continue

    # Extract ROS timestamp from filename
    timestamp = os.path.splitext(filename)[0]
       # Load masks
    masks = np.load(os.path.join(input_dir, filename))

    # Process each mask
    for idx, mask in enumerate(masks):
        y_indices, x_indices = np.where(mask)
        if first:
            # Initialize previous values
            x_min_previous = np.min(x_indices)
            x_max_previous = np.max(x_indices)
            y_min_previous = np.min(y_indices)
            y_max_previous = np.max(y_indices)
            first = False
        if len(x_indices) == 0 or len(y_indices) == 0:
            continue

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # Check if values are noisy by comparing the present to previous value
        print(x_min - x_min_previous)
        if abs(x_min - x_min_previous) > 100:
            # If the difference is too large, use previous values
            x_min = x_min_previous
        if abs(x_max - x_max_previous) > 100:
            # If the difference is too large, use previous values
            x_max = x_max_previous
        if abs(y_min - y_min_previous) > 100:
            # If the difference is too large, use previous values
            y_min = y_min_previous
        if abs(y_max - y_max_previous) > 100:
            # If the difference is too large, use previous values
            y_max = y_max_previous

        x_min_previous = x_min
        x_max_previous = x_max
        y_min_previous = y_min
        y_max_previous = y_max

        # Create a dictionary entry
        entry = {
            "timestamp": int(timestamp),  # or str(timestamp)
            "bbox_id": idx,  # unique ID for boxes in the same timestamp
            "xmin": int(x_min),
            "xmax": int(x_max),
            "ymin": int(y_min),
            "ymax": int(y_max)
        }

        data.append(entry)

# Save to JSON
with open(output_json, "w") as f:
    json.dump(data, f, indent=2)

print(f"Saved {len(data)} bounding boxes to {output_json}")

# read the 3D bbox json
with open(root + 'annotation_human/human_rgb_bounding_box_labels_3d.json', 'r') as file:
    bbox_3d = [json.loads(line) for line in file]

# get the z value from the bbox 3d for the timestamp that is nearest to the timestamp in data. Add the zmin and zmax to data
for entry in data:
    timestamp = entry["timestamp"]
    # Find the closest bbox_3d entry
    closest_bbox = min(bbox_3d, key=lambda x: abs(int(x["timestamp"]) - timestamp))

    # Add zmin and zmax to the entry
    entry["zmin"] = closest_bbox["zmin"]
    entry["zmax"] = closest_bbox["zmax"]

# load camera params
with open(root + 'camera_params.json', 'r') as file:
    cam_params = json.load(file)
# Extract camera intrinsic matrix
K = np.array(cam_params['camera_mtx_cam1'])
fx = K[0][0]
fy = K[1][1]
cx = K[0][2]
cy = K[1][2]

H_rgb_to_left = np.array(cam_params['H_rgb_2_left'])
H_right_to_rgb = np.array(cam_params['H_right_2_rgb'])
H_rgb_to_right = np.linalg.inv(H_right_to_rgb)

# Convert pixel coordinates to camera coordinates
# loop through data
for entry in data:
    # Get the pixel coordinates
    x_pixel = (entry["xmin"] + entry["xmax"]) / 2
    y_pixel = (entry["ymin"] + entry["ymax"]) / 2
    z = entry["zmax"]

    X, Y, Z = pixel_to_camera(x_pixel, y_pixel, z, fx, fy, cx, cy)

    # Add to the entry
    entry["X"] = X
    entry["Y"] = Y
    entry["Z"] = Z

    coords_left = H_rgb_to_left @ np.array([X, Y, Z, 1])
    coords_right = H_rgb_to_right @ np.array([X, Y, Z, 1])

    entry["X_left"] = coords_left[0]
    entry["Y_left"] = coords_left[1]
    entry["Z_left"] = coords_left[2]

    entry["X_right"] = coords_right[0]
    entry["Y_right"] = coords_right[1]
    entry["Z_right"] = coords_right[2]

# Save to JSON
with open(output_json, "w") as f:
    json.dump(data, f, indent=2)

#