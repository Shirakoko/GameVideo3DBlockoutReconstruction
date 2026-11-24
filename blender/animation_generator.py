"""Baked CFR keyframes. Blender UI/session state is never consulted."""
from blender.object_generator import apply_transform
from blender.camera_generator import set_camera_sample


def linear_curves(datablock):
    animation = datablock.animation_data
    if not animation or not animation.action:
        return
    action = animation.action
    curves = list(getattr(action, "fcurves", []))
    for layer in getattr(action, "layers", []):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", []):
                curves.extend(bag.fcurves)
    for curve in curves:
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"


def animate_objects(items, objects):
    for item in items:
        obj = objects[item["object_id"]]
        for sample in item["samples"]:
            apply_transform(obj, sample["transform"], item["dimensions"], mesh=item["primitive_type"] == "mesh")
            for path in ["location", "rotation_quaternion", "scale"]:
                obj.keyframe_insert(data_path=path, frame=sample["blender_frame"], group="Source Motion")
        linear_curves(obj)


def animate_cameras(items, cameras, scene):
    for item in items:
        obj = cameras[item["camera_id"]]
        for sample in item["samples"]:
            set_camera_sample(obj, sample, scene)
            for path in ["location", "rotation_quaternion"]:
                obj.keyframe_insert(data_path=path, frame=sample["blender_frame"], group="Source Camera Pose")
            for path in ["lens", "shift_x", "shift_y"]:
                obj.data.keyframe_insert(data_path=path, frame=sample["blender_frame"])
        linear_curves(obj)
        linear_curves(obj.data)
