"""Calibrated optical poses -> Blender cameras, preserving K and axes."""
import bpy
from mathutils import Quaternion


def set_camera_sample(camera, sample, scene):
    camera.location = sample["position"]
    x, y, z, w = sample["optical_rotation"]
    # Right multiply optical-from-Blender-camera diag(1,-1,-1).
    q = Quaternion((w, x, y, z)) @ Quaternion((0, 1, 0, 0))
    if camera.rotation_quaternion.dot(q) < 0:
        q.negate()
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = q
    k = sample["intrinsics"]
    width, height = scene.render.resolution_x, scene.render.resolution_y
    aspect = scene.render.pixel_aspect_y/scene.render.pixel_aspect_x
    data = camera.data
    data.type, data.sensor_fit, data.sensor_width = "PERSP", "HORIZONTAL", 36.
    data.lens = k[0][0]*data.sensor_width/width
    data.shift_x = (width/2-k[0][2])/width
    data.shift_y = (k[1][2]-height/2)*aspect/width


def create_camera(item, collection, scene):
    name = item["camera_id"]
    data = bpy.data.cameras.new(name)
    data.clip_start, data.clip_end = 0.01, 10000.
    camera = bpy.data.objects.new(name, data)
    collection.objects.link(camera)
    camera["source_camera_id"] = name
    camera["pose_source"] = item["pose_source"]
    set_camera_sample(camera, item["samples"][0], scene)
    return camera
