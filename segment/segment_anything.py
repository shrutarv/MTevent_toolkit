# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""# Video segmentation with SAM 2

This notebook shows how to use SAM 2 for interactive segmentation in videos. It will cover the following:

- adding clicks (or box) on a frame to get and refine _masklets_ (spatio-temporal masks)
- propagating clicks (or box) to get _masklets_ throughout the video
- segmenting and tracking multiple objects at the same time

We use the terms _segment_ or _mask_ to refer to the model prediction for an object on a single frame, and _masklet_ to refer to the spatio-temporal masks across the entire video.

<a target="_blank" href="https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Environment Set-up

If running locally using jupyter, first install `sam2` in your environment using the [installation instructions](https://github.com/facebookresearch/sam2#installation) in the repository.

If running from Google Colab, set `using_colab=True` below and run the cell. In Colab, be sure to select 'GPU' under 'Edit'->'Notebook Settings'->'Hardware accelerator'. Note that it's recommended to use **A100 or L4 GPUs when running in Colab** (T4 GPUs might also work, but could be slow and might run out of memory in some cases).
"""

"""## Set-up"""

import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import json
import cv2

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

with open("/mnt/smbshare/scene_data.json", "r") as file:
    scenes_data = json.load(file)
objects = ['human','hupwagen']
# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
for scenes,o in scenes_data.items():
    print('scene:', scenes)
    root_dir = '/mnt/smbshare/' + scenes + '/'
    video_dir = root_dir + "rgb"


    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    """### Loading the SAM 2 video predictor"""

    from sam2.build_sam import build_sam2_video_predictor

    sam2_checkpoint = "/home/eventcamera/sam2/sam2/checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    def show_mask(mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def save_mask(mask, out_frame_idx, obj_id=None):

        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        save_path = mask_save_path + frame_names[out_frame_idx]
        # save mask image
        plt.imshow(mask_image)
        plt.savefig(save_path)


    def show_points(coords, labels, ax, marker_size=200):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    """#### Select an example video
    
    We assume that the video is stored as a list of JPEG frames with filenames like `<frame_index>.jpg`.
    
    For your custom videos, you can extract their JPEG frames using ffmpeg (https://ffmpeg.org/) as follows:
    ```
    ffmpeg -i <your_video>.mp4 -q:v 2 -start_number 0 <output_dir>/'%05d.jpg'
    ```
    where `-q:v` generates high-quality JPEG frames and `-start_number 0` asks ffmpeg to start the JPEG file from `00000.jpg`.
    """


    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
          if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # take a look the first video frame
    frame_idx = 0
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))


    """#### Initialize the inference state
    
    SAM 2 requires stateful inference for interactive video segmentation, so we need to initialize an **inference state** on this video.
    
    During initialization, it loads all the JPEG frames in `video_path` and stores their pixels in `inference_state` (as shown in the progress bar below).
    """

    first_iter = True
    for obj in objects:

        # if root_dir + 'annotation_' + obj + '/' directoy is empty, skip the object and delete the directory
        if not os.listdir(root_dir + 'annotation_' + obj + '/'):
            os.rmdir(root_dir + 'annotation_' + obj + '/')
            continue
        mask_save_path = root_dir + '/output_masks_' + obj + '_img/'
        if not os.path.exists(mask_save_path):
            os.makedirs(mask_save_path)
        # empty cuda memory
        torch.cuda.empty_cache()

        """#### Initialize the inference state
    
        SAM 2 requires stateful inference for interactive video segmentation, so we need to initialize an **inference state** on this video.
    
        During initialization, it loads all the JPEG frames in `video_path` and stores their pixels in `inference_state` (as shown in the progress bar below).
        """
        if first_iter:
            inference_state = predictor.init_state(video_path=video_dir,async_loading_frames=True)
            first_iter = False
        """### Example 1: Segment & track one object
        
        Note: if you have run any previous tracking using this `inference_state`, please reset it first via `reset_state`.
        
        (The cell below is just for illustration; it's not needed to call `reset_state` here as this `inference_state` is just freshly initialized above.)
        """

        predictor.reset_state(inference_state)

        """In addition to using clicks as inputs, SAM 2 also supports segmenting and tracking objects in a video via **bounding boxes**.
        
        In the example below, we segment the child on the right using a **box prompt** of (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) on frame 0 as input into the `add_new_points_or_box` API.
        """
        # Load bounding box data from json file
        bbox_data = []

        with open(root_dir + 'annotation_' + obj + '/' + obj + '_rgb_bounding_box_labels_2d.json', 'r') as file:
            for line in file:
                bbox_data.append(json.loads(line))
        first_frame = bbox_data[0]

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)

        # Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
        #box = np.array([1083, 183, 1220, 556], dtype=np.float32)
        box = np.array([first_frame['xmin'], first_frame['ymin'], first_frame['xmax'], first_frame['ymax']], dtype=np.float32)
        print('box:', box)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            box=box,
        )

        # show the results on the current (interacted) frame
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {ann_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        show_box(box, plt.gca())
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

        """Here, SAM 2 gets a pretty good segmentation mask of the entire human, even though the input bounding box is not perfectly tight around the object.
        
        Similar to the previous example, if the returned mask from is not perfect when using a box prompt, we can also further **refine** the output using positive or negative clicks. To illustrate this, here we make a **positive click** at (x, y) = (460, 60) with label `1` to expand the segment around the child's hair.
        
        Note: to refine the segmentation mask from a box prompt, we need to send **both the original box input and all subsequent refinement clicks and their labels** when calling `add_new_points_or_box`.
        """

        """Then, to get the masklet throughout the entire video, we propagate the prompts using the `propagate_in_video` API."""

        # run propagation throughout the video and collect the results in a dict
        save_dir = root_dir + '/output_masks_' + obj + '_img/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # render the segmentation results every few frames
        vis_frame_stride = 1
        plt.close("all")
        for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            plt.figure(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")
            plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                print('scene', scenes, 'saved mask for object', obj, 'with frame', frame_names[out_frame_idx])
                #show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
                save_mask(out_mask, out_frame_idx)
                # save out_mask as a npy file with name as frame_name without jpy extension
                np.save(os.path.join(save_dir, f"{frame_names[out_frame_idx][:-4]}.npy"), out_mask)
            #plt.show()
            plt.close("all")