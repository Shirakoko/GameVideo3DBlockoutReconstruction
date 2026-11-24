"""Independent saved-scene inspection with Blender's projection oracle."""
import math
import json

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Quaternion, Vector

from blender.object_generator import COLORS


def verify_scene(plan):
    scene = bpy.context.scene
    objects = {o.get("source_object_id"): o for o in scene.objects if o.get("source_object_id")}
    cameras = {o.get("source_camera_id"): o for o in scene.objects if o.get("source_camera_id")}
    if len(objects) != len(plan["objects"]) or len(cameras) != len(plan["cameras"]):
        raise ValueError("Missing/duplicated source IDs in saved scene")
    if len(scene.objects) != len(objects)+len(cameras)+3:
        raise ValueError("Unexpected accumulated objects (expected reference + sun + area)")
    if (scene.frame_start, scene.frame_end) != (plan["frame_start"], plan["frame_end"]):
        raise ValueError("Saved scene frame range mismatch")
    if abs(scene.render.fps/scene.render.fps_base-float(__import__('fractions').Fraction(plan["fps"]))) > 1e-5:
        raise ValueError("Saved Blender FPS mismatch")
    if scene.camera != cameras[plan["active_camera"]]:
        raise ValueError("Saved active camera mismatch")
    if ([scene.render.resolution_x, scene.render.resolution_y] != plan["resolution"]
            or scene.render.resolution_percentage != 100):
        raise ValueError("Saved render resolution mismatch")
    if any(abs(v-e) > 1e-6 for v, e in zip(
            (scene.render.pixel_aspect_x, scene.render.pixel_aspect_y), plan["pixel_aspect"])):
        raise ValueError("Saved render pixel aspect mismatch")
    if scene.unit_settings.system != "METRIC" or abs(scene.unit_settings.scale_length-1) > 1e-6:
        raise ValueError("Saved scene unit scale mismatch")
    mapping = bpy.data.texts.get("FrameMapping.json")
    if mapping is None or json.loads(mapping.as_string()) != plan["frame_mapping"]:
        raise ValueError("Saved source frame/tick mapping mismatch")
    for item in plan["objects"]:
        obj = objects[item["object_id"]]
        if not len(obj.data.vertices) or not len(obj.data.polygons):
            raise ValueError("Empty saved mesh")
        extents = [max(v.co[i] for v in obj.data.vertices)-min(v.co[i] for v in obj.data.vertices) for i in range(3)]
        if item["primitive_type"] == "mesh":
            if any(abs(e-d) > max(1e-4, d*1e-5) for e, d in zip(extents, item["dimensions"])):
                raise ValueError("Mesh metre bounds mismatch; possible double scaling/axis conversion")
            if not obj.data.uv_layers or not obj.data.materials:
                raise ValueError("Textured mesh lost UV/material data")
            images = [node.image for mat in obj.data.materials if mat.use_nodes
                      for node in mat.node_tree.nodes if node.type == "TEX_IMAGE" and node.image]
            if not images or any(not image.packed_file for image in images):
                raise ValueError("Texture not packed in saved blend")
            continue
        if any(abs(e-1) > 1e-6 for e in extents):
            raise ValueError("Primitive is not a unit mesh; dimensions would be applied twice")
        if len(obj.data.materials) != 1:
            raise ValueError("Missing/extra primitive category material")
        mat = obj.data.materials[0]
        if mat.name != "Blockout_"+item["category"] or any(
                abs(v-e) > 1e-6 for v, e in zip(mat.diffuse_color, COLORS[item["category"]])):
            raise ValueError("Saved category material mismatch")
    max_projection = max_pose = max_object = 0.
    projection_cases = 0
    local = Matrix.Diagonal((1., -1., -1., 1.))
    for ordinal in range(plan["frame_end"]):
        scene.frame_set(ordinal+1)
        bpy.context.view_layer.update()
        for item in plan["cameras"]:
            sample = item["samples"][ordinal]
            camera = cameras[item["camera_id"]]
            x, y, z, w = sample["optical_rotation"]
            optical = Quaternion((w, x, y, z)).to_matrix().to_4x4()
            optical.translation = sample["position"]
            expected = optical @ local
            max_pose = max(max_pose, max(abs(camera.matrix_world[i][j]-expected[i][j]) for i in range(4) for j in range(4)))
            k = sample["intrinsics"]
            for p in [(0, 0, 5), (1, -.6, 5), (-.7, .4, 8)]:
                view = world_to_camera_view(scene, camera, optical @ Vector(p))
                expected_u = (k[0][0]*p[0]/p[2]+k[0][2])/plan["resolution"][0]
                expected_v = 1-(k[1][1]*p[1]/p[2]+k[1][2])/plan["resolution"][1]
                max_projection = max(max_projection, abs(view.x-expected_u), abs(view.y-expected_v))
                projection_cases += 1
        for item in plan["objects"]:
            obj = objects[item["object_id"]]
            transform = item["samples"][ordinal]["transform"] if item["samples"] else item["transform"]
            x, y, z, w = transform["rotation"]
            expected = Matrix.LocRotScale(Vector(transform["position"]), Quaternion((w, x, y, z)),
                Vector(transform["scale"] if item["primitive_type"] == "mesh" else [d*s for d, s in zip(item["dimensions"], transform["scale"])]))
            max_object = max(max_object, max(abs(obj.matrix_world[i][j]-expected[i][j]) for i in range(4) for j in range(4)))
            # Organizational parent links must never inherit another world pose.
            if obj.parent is not None:
                raise ValueError("Unexpected transform parent")
            target_collection = "Static" if item["is_static"] is True else "Dynamic" if item["is_static"] is False else "UnknownMotion"
            if target_collection not in {c.name for c in obj.users_collection}:
                raise ValueError("Incorrect static/dynamic collection")
    tolerance = 5e-5
    if any(not math.isfinite(v) or v > tolerance for v in [max_pose, max_object, max_projection]):
        raise ValueError(f"Saved scene numerical QA failed: camera={max_pose}, object={max_object}, projection={max_projection}")
    scene.frame_set(plan["frame_start"])
    return dict(status="passed", blender_version=bpy.app.version_string, frame_count=plan["frame_end"],
                objects=len(objects), cameras=len(cameras), projection_cases=projection_cases,
                max_camera_matrix_error=max_pose, max_object_matrix_error=max_object,
                max_projection_error_normalized=max_projection, tolerance=tolerance,
                evidence_boundary="Saved Blender transforms/projection only; not real-world reconstruction accuracy")
