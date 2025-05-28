import json

with open("/home/eventcamera/RGB_Event_cam_system/Annotation/Annotation_rgb_ec/obj_model/models_info.json", "r") as file:
    model_info = json.load(file)

# loop through the models
for i, info in model_info.items():
    print('volume', model_info[i]['size_x']/10 * model_info[i]['size_y']/10 * model_info[i]['size_z']/10)
