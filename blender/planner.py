"""Validated Scene JSON -> fully sampled portable background-build plan."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

import numpy as np
from jsonschema import Draft202012Validator

from blender.timeline import fps_fraction, frame_mapping, sample_camera, sample_transform
from reconstruction.coordinate_transform import validate_rigid, world_from_optical_ue_pose
from scene_schema import validate_scene


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_camera_poses(path):
    """Optional dense original poses: verified named contract, not raw telemetry.

    world_from_optical_camera matrices OR explicit UE FRotator degrees. No
    inference of units, handedness, SDK rotation or render-pose latch.
    """
    document = read_json(path)
    if document.get("schema") != "stage_e.camera_poses/1" or document.get("verified") is not True:
        raise ValueError("Dense Camera Pose requires verified stage_e.camera_poses/1; raw telemetry refused")
    json.dumps(document, allow_nan=False)
    schema = read_json(Path(__file__).resolve().parents[1]/"config/camera_pose_schema.json")
    errors = list(Draft202012Validator(schema).iter_errors(document))
    if errors:
        raise ValueError("Invalid reviewed Camera Pose: "+"; ".join(e.message for e in errors[:5]))
    if not isinstance(document.get("evidence"), str) or not document["evidence"].strip():
        raise ValueError("Reviewed Camera Pose requires contract/synchronization evidence")
    mode = document.get("pose_format")
    if mode == "world_from_optical_camera":
        if document.get("coordinate_system") != "blender_rh_z_up" or document.get("translation_unit") != "m":
            raise ValueError("Optical Camera Pose must explicitly use Blender world and metres")
    elif mode == "ue_frotator_degrees":
        if document.get("coordinate_system") != "ue_lh_z_up" or "world_units_per_meter" not in document:
            raise ValueError("UE poses need explicit axes and world units per metre")
    else:
        raise ValueError("Unsupported Camera Pose direction/format")
    result = {}
    for sample in document["samples"]:
        row = deepcopy(sample)
        if mode == "ue_frotator_degrees":
            p = row["pose"]
            row["world_from_camera"] = world_from_optical_ue_pose(p["position"], pitch_deg=p["pitch"],
                yaw_deg=p["yaw"], roll_deg=p["roll"], world_units_per_meter=document["world_units_per_meter"]).tolist()
        else:
            row["world_from_camera"] = validate_rigid(row["world_from_camera"]).tolist()
        validate_intrinsics(row["intrinsics"])
        result.setdefault(row["camera_id"], []).append(row)
    for rows in result.values():
        if any(b["frame_index"] <= a["frame_index"] or b["timestamp_seconds"] <= a["timestamp_seconds"] for a, b in zip(rows, rows[1:])):
            raise ValueError("Dense Camera Pose indices/timestamps must strictly increase")
    return result


def validate_intrinsics(k):
    k = np.asarray(k, dtype=float)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or abs(k[0, 1]) > 1e-10 or abs(k[1, 0]) > 1e-10 or not np.allclose(k[2], [0, 0, 1])):
        raise ValueError("Blender pinhole camera requires finite positive focal lengths and zero skew")
    return k


def prepare_plan(scene_path, video, *, resolution=None, camera_poses=None, active_camera=None,
                 engine="BLENDER_WORKBENCH", samples=16, point_cloud=None, max_reference_points=50000):
    scene_path = Path(scene_path).resolve()
    scene = read_json(scene_path)
    validate_scene(scene)
    if not scene["reconstruction_metadata"]["geometry_contract_verified"]:
        raise ValueError("Stage E blocked: Scene geometry/Camera Pose contract unverified; no fabricated transforms")
    if any(o["transform"] is None for o in scene["objects"]):
        raise ValueError("Stage E requires measured transform/dimensions for every object")
    if not scene["objects"]:
        raise ValueError("No geometry to generate")
    if samples < 1 or max_reference_points < 1:
        raise ValueError("Positive samples/reference point budget required")
    if engine not in {"BLENDER_WORKBENCH", "BLENDER_EEVEE", "CYCLES"}:
        raise ValueError("Unsupported render engine")
    w, h = resolution or (video["width"], video["height"])
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (w, h)) or min(w, h) <= 0 or w % 2 or h % 2:
        raise ValueError("H.264 yuv420p output needs positive even width/height")
    # K lives in the observed native RGB raster, not an arbitrary target size.
    rasters = {tuple(r["raster_size"]) for o in scene["objects"] for r in o["observations"]}
    if rasters != {(video["width"], video["height"])}:
        raise ValueError("Scene native raster differs from original video; explicit upstream K/raster conversion required")
    camera_ids = {c["camera_id"] for c in scene["cameras"]}
    active = active_camera or scene["timeline"]["frames"][0]["camera_id"]
    if active not in camera_ids:
        raise ValueError("Unknown active Camera ID")
    mapping = frame_mapping(scene["timeline"], video, active)
    overrides = load_camera_poses(camera_poses) if camera_poses else {}
    if set(overrides)-camera_ids:
        raise ValueError("Dense poses contain unknown Camera ID")
    dense_ticks = {r["frame_index"]: r["original_tick"] for r in overrides.get(active, []) if "original_tick" in r}
    for frame in mapping:
        tick = dense_ticks.get(frame["source_frame_index"])
        if tick is not None:
            if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
                raise ValueError("Original Camera Pose ticks must be nonnegative integers")
            if frame["original_tick"] is not None and frame["original_tick"] != tick:
                raise ValueError("Dense Camera Pose original tick disagrees with Scene JSON")
            frame["original_tick"] = tick
    cameras, aspect = [], None
    for camera in scene["cameras"]:
        rows = overrides.get(camera["camera_id"], camera["track"])
        for row in rows:
            if row.get("world_from_camera") is None or row.get("intrinsics") is None:
                raise ValueError("Camera pose/intrinsics unknown; raw_source is provenance, not a usable transform")
            validate_rigid(row["world_from_camera"])
            validate_intrinsics(row["intrinsics"])
        cam_map = frame_mapping(scene["timeline"], video, camera["camera_id"])
        origin = cam_map[0]["source_frame_index"]
        for row in rows:
            offset = (row["timestamp_seconds"]-scene["timeline"]["start_seconds"])*float(fps_fraction(scene["timeline"]["fps"]))
            if abs(offset-round(offset)) > 1e-6 or row["frame_index"] != origin+round(offset):
                raise ValueError("Camera Pose Frame Index/time mismatch")
        sampled = []
        for frame in mapping:
            value = sample_camera(rows, frame["timestamp_seconds"])
            k = validate_intrinsics(value["intrinsics"]).copy()
            k[0] *= w/video["width"]
            k[1] *= h/video["height"]
            ratio = float(k[0, 0]/k[1, 1])
            if aspect is None:
                aspect = ratio
            if abs(ratio-aspect) > 1e-6:
                raise ValueError("Changing fx/fy or incompatible camera pixel aspects not supported")
            value.update(blender_frame=frame["blender_frame"], intrinsics=k.tolist())
            sampled.append(value)
        cameras.append(dict(camera_id=camera["camera_id"], samples=sampled,
                            pose_source=str(Path(camera_poses).resolve()) if camera["camera_id"] in overrides else str(scene_path)))
    objects = []
    for obj in scene["objects"]:
        item = {k: deepcopy(obj[k]) for k in ["object_id", "category", "primitive_type", "transform", "dimensions", "parent_id", "is_static"]}
        item["samples"] = [dict(blender_frame=f["blender_frame"], transform=sample_transform(obj["motion_track"], f["timestamp_seconds"]))
                           for f in mapping] if obj["motion_track"] else []
        objects.append(item)
    references, reference_sources = [], []
    paths = [Path(point_cloud).resolve()] if point_cloud else [scene_path.parent/o["geometry_source"]["point_cloud_path"] for o in scene["objects"]]
    # Read original bytes with Open3D on the host; Blender needs only vertices.
    import open3d as o3d
    for path in dict.fromkeys(paths):
        if not path.is_file():
            raise ValueError(f"Missing geometric source: {path}")
        cloud = o3d.io.read_point_cloud(str(path))
        p = np.asarray(cloud.points)
        if not len(p) or not np.isfinite(p).all():
            raise ValueError(f"Invalid/empty source point cloud: {path}")
        reference_sources.append(str(path.resolve()))
        references.append(p[::max(1, int(np.ceil(len(p)/max_reference_points)))])
    points = np.concatenate(references)
    points = points[::max(1, int(np.ceil(len(points)/max_reference_points)))]
    fps = fps_fraction(scene["timeline"]["fps"])
    # Blender stores integer fps and float fps_base; actual rational retained.
    fps_integer = round(float(fps))
    return dict(schema="stage_e.build_plan/1", scene_id=scene["scene_id"], source_scene=str(scene_path), source_scene_json=scene,
                data_source=scene["reconstruction_metadata"]["data_source"], resolution=[w, h],
                fps=str(fps), fps_integer=fps_integer, fps_base=fps_integer/float(fps),
                frame_start=1, frame_end=len(mapping), frame_mapping=mapping, active_camera=active,
                pixel_aspect=[1., aspect] if aspect >= 1 else [1/aspect, 1.], cameras=cameras, objects=objects, reference_points=points.tolist(),
                reference_sources=reference_sources, engine=engine, samples=samples,
                conventions=dict(positions="metres, Blender RH Z up", object_dimensions="unscaled primitive dimensions; scale applied once",
                    parents="organizational world-space grouping", camera="optical +X right +Y down +Z forward -> Blender +X right +Y up -Z forward",
                    interpolation="per-render-frame linear translation/K/scale + shortest-arc quaternion SLERP",
                    visibility="not_detected is NOT occlusion; never used to hide objects",
                    motion_boundary="hold first/last object pose outside observed motion; camera extrapolation refused"))
