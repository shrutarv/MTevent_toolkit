import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import seaborn as sns
import pandas as pd

x = []
y = []
z = []
speed = []
time = []
angular_vel = []
smoothening_parameter = 8
with open("/media/eventcamera/event_data/dataset_27_feb/scene_data.json", "r") as file:
    scenes_data = json.load(file)
# Load JSON file
for scene, obj in scenes_data.items():
    x = []
    y = []
    z = []
    with open("/media/eventcamera/event_data/dataset_27_feb/" + scene + "/vicon_data/event_cam_sys.json", "r") as file:
        data = json.load(file)

    #print('file loaded', scene)
    # traverse the data dictionary and extract the values
    values = list(data.values())
    for i in range(len(values)):
        if i%smoothening_parameter == 0:
            x.append(values[i]['translation'][0])
            y.append(values[i]['translation'][1])
            z.append(values[i]['translation'][2])
            t = values[i]['timestamp']
            # convert to float
            t = float(t)
            # convert t from nano seconds to seconds
            t = t/1e9
            time.append(t)

    rotations = [R.from_quat([entry["rotation"][1], entry["rotation"][2], entry["rotation"][3], entry["rotation"][0]]) for entry in values]
    # keep every 5th element
    rotations = rotations[::smoothening_parameter]
    delta_time = np.diff(time)
    epsilon = 1e-6  # Small number to prevent division errors

    for i in range(len(rotations) - 1):
        #print(i)
        # Compute relative rotation
        relative_rotation = rotations[i + 1] * rotations[i].inv()
        # Convert to angular velocity
        omega = relative_rotation.as_rotvec() / (0.005 * smoothening_parameter + epsilon) # 0.005 comes from 200 fps
        angular_vel.append(omega)
    # print min and max angular velocity and translation velocity
    x_temp = np.diff(x) / (0.005 * smoothening_parameter)
    y_temp = np.diff(y) / (0.005 * smoothening_parameter)
    z_temp = np.diff(z) / (0.005 * smoothening_parameter)
    s_temp = np.sqrt(x_temp**2 + y_temp**2 + z_temp**2)
    speed.append(s_temp)
    s_temp = s_temp[s_temp < 3]
    print('Scene', scene,
          'min translation:', np.min(s_temp), 'max translation:', np.max(s_temp))

angular_vel = np.array(angular_vel)
# compute absolue angular velocity
angular_vel_absolute = np.linalg.norm(angular_vel, axis=1)
# compute speed
x = np.diff(x)/(0.005 * smoothening_parameter)
y = np.diff(y)/(0.005 * smoothening_parameter)
z = np.diff(z)/(0.005 * smoothening_parameter)
# compute absolute distance instead of separatelly for x,y and z
#speed.append(np.sqrt(x**2 + y**2 + z**2))
#remove any speed value greater than 3
speed = np.concatenate(speed)
speed = speed[speed < 2.1]

plt.rcParams.update({'font.size': 16})
    #time = time[:-1]
# plot a histogram of the speed
plt.hist(speed, bins=20, color='gray', linewidth=1.5, rwidth = 0.8)
plt.title("Translational velocity histogram")
plt.xlabel("Camera Translational Velocity (m/s)")
#plt.yscale('log')
plt.ylabel("Number of occurences")

plt.tight_layout()
# put xtick max to 2
plt.xlim(0, 2)

plt.savefig('/media/eventcamera/event_data/dataset_27_feb/' + 'translation_speed_histogram.png')
plt.show()
plt.hist(angular_vel_absolute, bins=20, color='gray', linewidth=1.5, rwidth = 0.8)
plt.title("Angular velocity histogram")
plt.xlabel("Camera Angular velocity (rad/s)")
plt.ylabel("Number of occurences")
#plt.yscale('log')

plt.tight_layout()
plt.xlim(0, 2)

plt.savefig('/media/eventcamera/event_data/dataset_27_feb/' + 'angular_velocity_histogram.png')
plt.show()
''' 
   # Nest 2 plots in one figure
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(time, speed, 'b')
plt.title("Translational speed")
plt.xlabel("Time (s)")
plt.ylabel("Speed (m/sec)")
plt.legend(['speed'])
plt.subplot(1, 2, 2)
plt.plot(time, angular_vel_absolute, 'r')

plt.title("Angular velocity")
plt.xlabel("Time (s)")
plt.ylabel("Angular velocity (rad/sec)")
plt.tight_layout()
plt.legend(['absolute angular velocity', 'x', 'y', 'z'])
# Save the plot
#plt.savefig('/home/eventcamera/data/dataset/dataset_23_jan/' + folder_name + '/vicon_data/speed_'+ folder_name + '.png')

plt.show()

'''





