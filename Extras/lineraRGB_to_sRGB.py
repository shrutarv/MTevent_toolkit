import os
import cv2
import tensorflow as tf
import numpy as np

# Define the manual Linear RGB to sRGB conversion function
def linear_to_srgb(image):
    return tf.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * tf.pow(image, 1.0 / 2.4) - 0.055,
    )

# Function to process and save images
def process_image(input_path, output_path):
    # Load the image
    rgb_image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if rgb_image is None:
        print(f"Failed to load image: {input_path}")
        return

    # Normalize the image to [0, 1]
    linear_rgb_image = rgb_image / 255.0

    # Convert to TensorFlow tensor
    linear_rgb_tensor = tf.convert_to_tensor(linear_rgb_image, dtype=tf.float32)

    # Convert Linear RGB to sRGB
    srgb_tensor = linear_to_srgb(linear_rgb_tensor)
    srgb_image = srgb_tensor.numpy()  # Convert back to NumPy array

    # Clip and scale back to [0, 255]
    srgb_image = np.clip(srgb_image * 255.0, 0, 255).astype(np.uint8)
    # if output path does not esist create it
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    # Save the output image
    cv2.imwrite(output_path, srgb_image)
    print(f"Converted image saved to: {output_path}")

# Main function
def main(input_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Processing: {filename}")
            process_image(input_path, output_path)

if __name__ == "__main__":
    # Input and output directories
    input_directory = "/home/eventcamera/data/dataset/dataset_23_jan/scene10_2/rgb"  # Replace with the path to your RGB images
    output_directory = "/home/eventcamera/data/dataset/dataset_23_jan/scene10_2/rgb"  # Replace with the path to save sRGB images

    main(input_directory, output_directory)