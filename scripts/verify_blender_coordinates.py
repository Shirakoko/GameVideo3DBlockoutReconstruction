"""Run in Blender background: independent mathutils and camera projection QA.

No scene files, asset imports, meshes, animation, or rendering are produced.
Only an ephemeral camera and a numerical JSON receipt, separate from Stage E.
"""
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Euler, Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reconstruction.coordinate_transform import (blender_camera_from_optical_pose,
    transform_points, ue_rotator_matrix, world_from_optical_ue_pose)


def main():
    args = sys.argv[sys.argv.index("--")+1:]
    output = Path(args[0]).resolve()
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = 160, 90
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.
    data = bpy.data.cameras.new("StageB_coordinate_probe")
    data.type, data.sensor_fit, data.sensor_width, data.lens = "PERSP", "HORIZONTAL", 32., 16.
    camera = bpy.data.objects.new("StageB_coordinate_probe", data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    errors, rotation_errors = [], []
    for p, y, r in [(0., 0., 0.), (30., 45., 60.), (-10., 90., -31.), (89.9, -170., 17.)]:
        # Blender Euler uses standard RH XYZ: independent product implementation.
        oracle = np.asarray(Euler(tuple(np.deg2rad([-r, -p, y])), "XYZ").to_matrix())
        rotation_errors.append(float(np.max(np.abs(oracle-ue_rotator_matrix(pitch_deg=p, yaw_deg=y, roll_deg=r)))))
        t = world_from_optical_ue_pose([100, 200, 300], pitch_deg=p, yaw_deg=y, roll_deg=r, world_units_per_meter=100)
        camera.matrix_world = Matrix(blender_camera_from_optical_pose(t).tolist())
        bpy.context.view_layer.update()
        for optical in ([0., 0., 4.], [1., -0.5, 4.], [-1., 0.5, 6.]):
            world = transform_points([optical], t)[0]
            projected = world_to_camera_view(scene, camera, Vector(world.tolist()))
            x, yy, z = optical
            expected = np.array([0.5+80*x/z/160, 0.5-80*yy/z/90, z])
            errors.append(float(np.max(np.abs(np.asarray(projected)-expected))))
    receipt = {"status": "passed", "blender_version": bpy.app.version_string,
               "scope": "independent mathutils Euler and world_to_camera_view; NOT dataset render-pose verification",
               "camera_local": "+X right, +Y up, -Z forward", "projection_cases": len(errors),
               "rotation_cases": len(rotation_errors), "max_projection_error": max(errors),
               "max_rotation_matrix_error": max(rotation_errors), "tolerance": 2e-5,
               "scene_file_created": False, "rendering_performed": False}
    if max(errors+rotation_errors) > receipt["tolerance"]:
        receipt["status"] = "failed"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "passed":
        raise RuntimeError("Blender coordinate probe failed")


if __name__ == "__main__":
    main()
