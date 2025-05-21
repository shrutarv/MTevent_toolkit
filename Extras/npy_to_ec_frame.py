import numpy as np
import matplotlib.pyplot as plt
import os

# store all the npy files in a list
npy_files = os.listdir('/home/eventcamera/data/dataset/dataset_23_jan/scene3_1/event_cam_right_npy')
# iterate over all the npy files
for npy in npy_files:
    # import npy file
    data = np.load('/home/eventcamera/data/dataset/dataset_23_jan/scene3_1/event_cam_right_npy/' + npy, allow_pickle=True)
    # create a event camera frame of size 640x480
    event_cam = np.zeros((480, 640))
    x = []
    y = []
    pol = []
    # extract x,y and polarity from the data. x is stored at data[0][1]. store all x values in a list
    for i in range(len(data)):
        x.append(data[i][1])
        y.append(data[i][2])
        pol.append(data[i][3])
        event_cam[y[i],x[i]] = pol[i]

    # display the event camera frame
    #plt.imshow(event_cam)
    # remove .npy extension
    npy = npy[:-4]
    # save the figure
    plt.savefig('/home/eventcamera/data/dataset/dataset_23_jan/scene3_1/event_cam_right_npy_img/' + npy + '.png')
    print('saved', npy)
    #plt.show()

print('done')

