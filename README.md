# MTevent_Toolkit
Contains scrips to process and analyse the MTevent dataset.

The MTevent dataset is designed to advance event-based perception in dynamic environments. It addresses key challenges such as occlusions, varying lighting, extreme viewing angles, and long detection distances, providing a comprehensive benchmark for event-based vision across multiple tasks—including 6D pose estimation of static and moving rigid objects, 2D motion segmentation, 3D bounding box detection, optical flow estimation, and object tracking. Annotations included in the dataset:
1. 6D pose of rigid objects.
2. 3D/2D bouding box coordinates of all moving objects.

Read our paper: [MTevent: A Multi-Task Event Camera Dataset for 6D Pose Estimation and Moving Object Detection](https://arxiv.org/abs/1234.56789)

<p align="center">
  <img src="media/scene52.gif" />
  
</p>

The dataset includes:
Rigid objects- Objects with ids from MR6D1 to MR6D16
Non-rigid objects - Human and forklift

The link to download the dataset: [MTevent](https://huggingface.co/datasets/anas-gouda/MTevent/tree/main)
Download all the scenes to a folder which would be the root folder.


## Data pre-processing
1. Execute script Data_pre_processing/extract_rgb_events_vicon_data_from_bag.py
   This script extracts rgb images, vicon data for moving rigid and non rigid objects objects. The json files are saved in the folder root/scene /vicon_data/. The rgb images in .jpg format are saved in the folder root/scene /rgb.

2. Execute bag_to_event_img.py
   Extracts the events from left and right event camera bag files as slices. Each slice is accumulated and saved as a jpg image for visualisation purpose. The event visualisation are saved in root/scene /event_images/left and root/scene /event_images/right


   
    

<p align="center">
  <img src="media/scene63_mask_human.gif" width="400"/>
  <img src="media/scene63_mask_obj.gif" width="300"/>
</p>

[Watch demo video](media/scene33_obj_bbox.mp4)
