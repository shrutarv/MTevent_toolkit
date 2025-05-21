import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
x = []
y = []
z = []
time = []
smoothening_parameter = 20
folder_name = 'scene4_3'
# Load JSON file
with open("/home/eventcamera/data/dataset/dataset_23_jan/" + folder_name + "/vicon_data/event_cam_sys.json", "r") as file:
    data = json.load(file)

print('file loaded')
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
angular_vel = []
delta_time = np.diff(time)
epsilon = 1e-6  # Small number to prevent division errors

for i in range(len(rotations) - 1):
    print(i)
    # Compute relative rotation
    relative_rotation = rotations[i + 1] * rotations[i].inv()
    # Convert to angular velocity
    omega = relative_rotation.as_rotvec() / (0.005 * smoothening_parameter + epsilon)
    angular_vel.append(omega)
angular_vel = np.array(angular_vel)
# compute absolue angular velocity
angular_vel_absolute = np.linalg.norm(angular_vel, axis=1)
# compute speed
x = np.diff(x)/(0.005 * smoothening_parameter)
y = np.diff(y)/(0.005 * smoothening_parameter)
z = np.diff(z)/(0.005 * smoothening_parameter)
# compute absolute distance instead of separatelly for x,y and z
speed = np.sqrt(x**2 + y**2 + z**2)
time = time[:-1]
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
'''
plt.plot(time, angular_vel[:,0], 'b')
plt.plot(time, angular_vel[:,1], 'g')
plt.plot(time, angular_vel[:,2], 'y')
'''
plt.title("Angular velocity")
plt.xlabel("Time (s)")
plt.ylabel("Angular velocity (rad/sec)")
plt.tight_layout()
plt.legend(['absolute angular velocity', 'x', 'y', 'z'])
# Save the plot
plt.savefig('/home/eventcamera/data/dataset/dataset_23_jan/' + folder_name + '/vicon_data/speed_'+ folder_name + '.png')

plt.show()







