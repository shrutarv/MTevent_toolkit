#!/usr/bin/env python3

import os
import rosbag
import cv2
import numpy as np
import queue
import threading

# Root directory where dataset is stored
root = "/mnt/smbshare/"
start_scene = 52
end_scene = 53
# Event camera topics
event_topics = {
    "left": "/dvxplorer_left/events",
    "right": "/dvxplorer_right/events"
}

# Time step for slicing (in nanoseconds) - Example: 10ms
time_step_ns = 7_000_000

# Check for GPU acceleration (CUDA support)
try:
    cv2.ocl.setUseOpenCL(True)
    use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
    print(f"CUDA enabled: {use_gpu}")
except:
    use_gpu = False


# Function to process events and generate images for a given camera
def process_events(camera_name, output_dir):
    while True:
        batch = event_queues[camera_name].get()
        if batch is None:
            break

        batch_timestamp, events = batch  # Get assigned timestamp and events
        img = np.zeros((480, 640), dtype=np.uint8)  # Blank grayscale image

        # Use GPU if available
        if use_gpu:
            gpu_img = cv2.cuda.GpuMat()
            gpu_img.upload(img)

            for x, y, polarity in events:
                color = 255 if polarity else 100  # White for positive, Gray for negative
                if 0 <= x < 640 and 0 <= y < 480:
                    gpu_img.setTo(color, mask=None)  # Set event pixels

            gpu_img.download(img)  # Copy processed image back to CPU memory
        else:
            for x, y, polarity in events:
                color = 255 if polarity else 100
                if 0 <= x < 640 and 0 <= y < 480:
                    img[y, x] = color  # Assign event intensity

        # Save image using a clean timestamp format (integer nanoseconds)
        image_filename = os.path.join(output_dir, camera_name, f"{int(batch_timestamp)}.png")
        cv2.imwrite(image_filename, img)
        #print(f"Saved {image_filename} with {len(events)} events.")


# Function to process a single ROS bag file
def process_bag_file(bag_path, camera_name):
    with rosbag.Bag(bag_path, "r") as bag:
        current_batch = []
        current_time_window = None

        # Read messages for this specific camera
        for topic, msg, t in bag.read_messages(topics=[event_topics[camera_name]]):
            for event in msg.events:
                event_time_ns = event.ts.to_nsec()  # Ensure this retrieves nanoseconds

                # Initialize time window on first event
                if current_time_window is None:
                    current_time_window = (event_time_ns // time_step_ns) * time_step_ns

                # If the event belongs to the current time window, add it to the batch
                if event_time_ns < current_time_window + time_step_ns:
                    current_batch.append((event.x, event.y, event.polarity))
                else:
                    # Send previous batch for processing
                    if current_batch:
                        event_queues[camera_name].put((current_time_window + time_step_ns // 2, current_batch))

                    # Move to the next time window
                    current_time_window = (event_time_ns // time_step_ns) * time_step_ns
                    current_batch = [(event.x, event.y, event.polarity)]  # Start new batch

        # Process remaining events
        if current_batch:
            event_queues[camera_name].put((current_time_window, current_batch))


# Process scenes from scene22 to scene36
for scene_id in range(start_scene, end_scene):
    object_name = f"scene{scene_id}"
    scene_path = os.path.join(root, object_name)

    # Define paths for left and right bag files
    left_bag_path = os.path.join(scene_path, "left.bag")
    right_bag_path = os.path.join(scene_path, "right.bag")

    # Skip if bag files are missing
    if not os.path.exists(left_bag_path) or not os.path.exists(right_bag_path):
        print(f"Skipping {object_name}: Bag files not found.")
        continue

    # Output directory for images
    output_dir = os.path.join(scene_path, "event_images")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "left"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "right"), exist_ok=True)

    # Event queues for both cameras
    event_queues = {
        "left": queue.Queue(maxsize=10),
        "right": queue.Queue(maxsize=10)
    }

    # Start processing threads for both cameras
    processing_threads = {}
    for cam in event_topics.keys():
        processing_threads[cam] = threading.Thread(target=process_events, args=(cam, output_dir))
        processing_threads[cam].start()

    # Create and start bag processing threads
    bag_threads = {
        "left": threading.Thread(target=process_bag_file, args=(left_bag_path, "left")),
        "right": threading.Thread(target=process_bag_file, args=(right_bag_path, "right"))
    }
    bag_threads["left"].start()
    bag_threads["right"].start()

    # Wait for bag processing to complete
    bag_threads["left"].join()
    bag_threads["right"].join()

    # Signal processing threads to stop
    for cam in event_topics.keys():
        event_queues[cam].put(None)

    # Wait for image processing threads to finish
    for cam in processing_threads.keys():
        processing_threads[cam].join()

    print(f"Event visualization completed for {object_name}.")

print("Processing completed for all scenes from", start_scene, "to", end_scene)
