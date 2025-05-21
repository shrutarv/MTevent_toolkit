# convert all png files in a folder to jpg
import os
import cv2

folder_path = '/home/eventcamera/data/dataset/dataset_23_jan/scene1_1/event_cam_left_30/e2calib'
for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        # Load the image
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # Use this only if you want to convert a 1 channel gray scale image to 3 channel RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Save as JPEG
        output_path = os.path.join(folder_path, filename.replace('.png', '.jpg'))
        cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(f"Converted {filename} to JPEG")