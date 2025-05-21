import numpy as np
import matplotlib.pyplot as plt

# read a npy file
data = np.load('/home/eventcamera/data/dataset/dataset_23_jan/scene3_1/output_masks_human_img/1737644214511218091.npy')
# Convert data from (1,1536,2048) to (1536,2038. visualise it in a camera frame of size 1536x2048
data = data[0]
# visualise the data
plt.imshow(data, cmap='gray')
plt.show()
