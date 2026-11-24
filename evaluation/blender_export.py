"""Export evaluated saved Blender cameras in Stage B's optical world convention."""
import numpy as np

from evaluation.report import write_json
from reconstruction.coordinate_transform import validate_rigid


def export_camera_poses(plan, output):
    import bpy
    scene = bpy.context.scene
    cameras = {obj.get("source_camera_id"): obj for obj in scene.objects if obj.get("source_camera_id")}
    samples, rounding = [], 0.
    for frame in plan["frame_mapping"]:
        scene.frame_set(frame["blender_frame"])
        bpy.context.view_layer.update()
        for identity in sorted(cameras):
            raw = np.array(cameras[identity].matrix_world, dtype=np.float64) @ np.diag([1., -1., -1., 1.])
            u, _, vt = np.linalg.svd(raw[:3, :3])
            proper = u @ vt
            difference = float(np.max(np.abs(raw[:3, :3]-proper)))
            if np.linalg.det(proper) < 0 or difference > 1e-5:
                raise ValueError("Saved camera is not rigid; numerical orthogonalization cannot repair scale/reflection")
            rounding = max(rounding, difference)
            raw[:3, :3] = proper
            samples.append(dict(camera_id=identity, frame_index=frame["source_frame_index"], timestamp_seconds=frame["timestamp_seconds"],
                world_from_camera=validate_rigid(raw).tolist()))
    scene.frame_set(plan["frame_start"])
    document = dict(schema="stage_g.camera_prediction/1", coordinate_system="blender_rh_z_up", translation_unit="m",
        pose_direction="world_from_optical_camera", samples=samples, provenance="inferred",
        source="Evaluated camera.matrix_world in reopened scene.blend, not input track or build-plan target",
        numerical_rounding=dict(method="nearest SO3 via SVD to remove Blender float32 matrix rounding only", max_rotation_entry_adjustment=rounding))
    write_json(output, document)
    return document
