import numpy as np
import cv2
import json


def compute_homography(K_rgb, K_gray, R, T):
    """Computes homography from RGB to grayscale image using camera extrinsics."""
    K_rgb_h = np.eye(4)
    K_rgb_h[:3, :3] = K_rgb  # Embed K_rgb into 4x4 matrix
    K_gray_h = np.eye(4)
    K_gray_h[:3, :3] = K_gray  # Embed K_gray into 4x4 matrix

    # Convert 3x4 [R|T] to 4x4 transformation matrix
    RT_h = np.eye(4)
    RT_h[:3, :4] = np.hstack((R, T.reshape(-1, 1)))
    H = K_gray_h @ RT_h @ np.linalg.inv(K_rgb_h)  # Compute homography
    return H[:3, :3]  # Extract 3x3 homography matrix
'''
def transform_2d_point(rgb_point, K_rgb, K_gray, R, T):
    """
    Transforms a 2D point from RGB image to grayscale image using camera extrinsics.

    Parameters:
    - rgb_point: (u, v) 2D coordinates in RGB image
    - K_rgb: Intrinsic matrix of the RGB camera
    - K_gray: Intrinsic matrix of the grayscale camera
    - R, T: Extrinsic transformation from RGB to grayscale

    Returns:
    - gray_point: (u', v') Transformed 2D point in grayscale image
    """
    # Convert 2D point to homogeneous coordinates
    u, v = rgb_point
    rgb_homogeneous = np.array([u, v, 1])  # (u, v, 1)

    # Convert from pixel coordinates to normalized camera coordinates
    xyz_rgb = np.linalg.inv(K_rgb) @ rgb_homogeneous  # Now in RGB camera frame

    # Assume arbitrary depth (Z) = 1 (we need a 3D point for transformation)
    xyz_rgb = np.append(xyz_rgb, 1)  # Convert to (X, Y, Z, 1) homogeneous form

    # Transform to grayscale camera frame using R and T
    xyz_gray = R @ xyz_rgb[:3] + T  # Apply rotation & translation

    # Project onto the grayscale camera plane
    uvw_gray = K_gray @ xyz_gray  # Convert to 2D homogeneous
    u_gray, v_gray = uvw_gray[0] / uvw_gray[2], uvw_gray[1] / uvw_gray[2]  # Normalize

    return int(u_gray), int(v_gray)  # Return pixel coordinates in grayscale image
'''
rgb_image = cv2.imread('/home/eventcamera/data/dataset/dataset_23_jan/scene1_3/rgb/1737719127505615456.png')
gray_image = cv2.imread('/home/eventcamera/data/dataset/dataset_23_jan/scene1_3/event_cam_left/e2calib/1737719127275183000.png', cv2.IMREAD_GRAYSCALE)
with open('/home/eventcamera/RGB_Event_cam_system/Annotation/camera_params.json', 'r') as file:
    data = json.load(file)
# cam1 is event camera left and cam2 is event camera right
K_rgb = np.array(data['camera_matrix'])
K_gray = np.array(data['camera_mtx_cam1'])
H_cam1_2_rgb = np.array(data['H_cam1_2_rgb'])
# Compute the inverse homography for RGB to grayscale mapping
H_rgb_2_cam1 = np.eye(4)
H_rgb_2_cam1[:3, :3] = np.transpose(H_cam1_2_rgb[:3, :3])
H_rgb_2_cam1[:3, 3] = -np.matmul(np.transpose(H_cam1_2_rgb[:3, :3]), H_cam1_2_rgb[:3, 3])

# Use feature detection and matching to get the homography matrix
# This is useful if you have a set of matching keypoints between the images.
orb = cv2.ORB_create()
kp_rgb, des_rgb = orb.detectAndCompute(rgb_image, None)
kp_gray, des_gray = orb.detectAndCompute(gray_image, None)

# Use a feature matcher (like BFMatcher) to find the best matching keypoints
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_rgb, des_gray)

# Extract matching points
pts_rgb = np.array([kp_rgb[m.queryIdx].pt for m in matches], dtype=np.float32)
pts_gray = np.array([kp_gray[m.trainIdx].pt for m in matches], dtype=np.float32)

# Compute homography based on the matching points
H_refined, _ = cv2.findHomography(pts_rgb, pts_gray, cv2.RANSAC)

# Convert H_rgb_2_cam1 to 3x3 homography
H = compute_homography(K_rgb, K_gray, H_rgb_2_cam1[:3, :3], H_rgb_2_cam1[:3, 3])
H = H_refined

# Define image resolutions
rgb_size = (2048, 1536)  # (width, height)
gray_size = (640, 480)   # (width, height)

gray_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
# Warp RGB image to grayscale image plane
warped_rgb = cv2.warpPerspective(rgb_image, H, gray_size)


# Blend images (Weighted sum)
alpha = 0.6  # Transparency factor

#cv2.imwrite('/home/eventcamera/Downloads/warped_rgb.png', warped_rgb)
#cv2.imwrite('/home/eventcamera/Downloads/gray_bgr.png', gray_bgr)
blended = cv2.addWeighted(gray_bgr, alpha, warped_rgb, 1 - alpha, 0)

# Show and save the result
cv2.imshow("Overlapped Image", blended)
cv2.waitKey(0)
cv2.destroyAllWindows()