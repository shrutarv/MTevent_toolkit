import rosbag
import numpy as np
import os
import cv2
import threading
import queue

objects = ['MR6D1','MR6D2','MR6D3']
object_name = 'scene_12'
bag = rosbag.Bag('//media/eventcamera/Windows/dataset_7_feb/' + object_name + '/' + object_name + '.bag')
path = '//media/eventcamera/Windows/dataset_7_feb/' + object_name + '/'
events_topic_left = '/dvxplorer_left/events'


# Use a queue for processing batches
event_queue = queue.Queue(maxsize=10)
# Output directory for images
output_dir = "//media/eventcamera/Windows/dataset_7_feb/" + object_name + "/event_images"
os.makedirs(output_dir, exist_ok=True)

# Open the ROS bag file
#bag = rosbag.Bag(bag_file, "r")

time_step_ns = 10_000_000

# Check for GPU acceleration (CUDA support)
try:
    cv2.ocl.setUseOpenCL(True)
    use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
    print(f"CUDA enabled: {use_gpu}")
except:
    use_gpu = False


# Function to process events and generate images
def process_events():
    while True:
        batch = event_queue.get()
        if batch is None:
            break

        batch_timestamp, events = batch  # Get assigned timestamp and events
        img = np.zeros((480, 640), dtype=np.uint8)  # Blank grayscale image

        # Use GPU if available
        if use_gpu:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)

            for x, y, polarity in events:
                color = 255 if polarity else 100  # White for positive, Gray for negative
                if 0 <= x < 640 and 0 <= y < 480:
                    img[y, x] = color  # Draw event

            gpu_img.download(img)  # Copy processed image back to CPU memory
        else:
            for x, y, polarity in events:
                color = 255 if polarity else 100
                if 0 <= x < 640 and 0 <= y < 480:
                    img[y, x] = color

        # Save image using a clean timestamp format (integer nanoseconds)
        image_filename = os.path.join(output_dir, f"{int(batch_timestamp)}.png")
        cv2.imwrite(image_filename, img)
        print(f"Saved {image_filename} with {len(events)} events.")


# Start the processing thread
processing_thread = threading.Thread(target=process_events)
processing_thread.start()

current_batch = []
current_time_window = None  # Start time of the current batch

for topic, msg, t in bag.read_messages(topics=[events_topic_left]):
    for event in msg.events:
        event_time_ns = event.ts.to_nsec()  # Get event timestamp in nanoseconds

        # Initialize time window on first event
        if current_time_window is None:
            current_time_window = (event_time_ns // time_step_ns) * time_step_ns

        # If the event belongs to the current time window, add it to the batch
        if event_time_ns < current_time_window + time_step_ns:
            current_batch.append((event.x, event.y, event.polarity))
        else:
            # Send previous batch for processing
            if current_batch:
                event_queue.put((current_time_window, current_batch))

            # Move to the next time window
            current_time_window = (event_time_ns // time_step_ns) * time_step_ns
            current_batch = [(event.x, event.y, event.polarity)]  # Start new batch

# Process remaining events
if current_batch:
    event_queue.put((current_time_window, current_batch))

# Signal processing thread to stop
event_queue.put(None)
processing_thread.join()

# Close the ROS bag
bag.close()
print("Event visualization completed.")
