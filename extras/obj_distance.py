import json

root = '/media/eventcamera/event_data/dataset_27_feb/'
with open("/media/eventcamera/event_data/dataset_27_feb/scene_data.json", "r") as file:
    scenes_data = json.load(file)

for scene, objects in scenes_data.items():
    for obj_name, obj_data in objects.items():
        bbox_path = root + scene + "/annotation/rgb_" + obj_name + "_bounding_box_labels_3d.json"
        with open(bbox_path, "r") as file:
            bbox_data = json.load(file)


        print(f"Scene: {scene}, Object: {obj_name}")
        print(obj_data)
        print("\n")