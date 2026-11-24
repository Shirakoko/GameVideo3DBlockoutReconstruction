"""Stage C RGB -> masks -> measured geometry -> tracks -> validated Stage D."""
from __future__ import annotations

from datetime import datetime
from itertools import combinations
from pathlib import Path
from zoneinfo import ZoneInfo
import json

import numpy as np
from PIL import Image, ImageDraw

from agents.object_detection import PRIMITIVES, confidence, unknown
from agents.tracking import Observation, PersistentTracker, appearance, classify_motion, independent_observations
from reconstruction.object_geometry import frame_points, load_pixel_cloud, measure, object_transform, static_bounds
from reconstruction.pointcloud_fusion import input_digest, point_cloud, write_verified_ply
from scene_schema import validate_scene


def json_write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def relationships(objects):
    result = []
    # Simultaneous observations only: dynamic objects are never related at
    # unrelated timestamps using the scene's representative transform.
    for a, b in combinations(objects, 2):
        lookup = {(r["frame_index"], r["timestamp_seconds"], r["camera_id"]): r for r in b["observations"] if r["position"] is not None}
        for ra in a["observations"]:
            rb = lookup.get((ra["frame_index"], ra["timestamp_seconds"], ra["camera_id"]))
            if rb is None or ra["position"] is None:
                continue
            pa, pb = np.array(ra["position"]), np.array(rb["position"])
            da, db = np.array(ra["dimensions"]), np.array(rb["dimensions"])
            low_a, high_a, low_b, high_b = pa-da/2, pa+da/2, pb-db/2, pb+db/2
            predicates = []
            if high_a[0] < low_b[0]:
                predicates.append((a, "negative_x_of", b))
            elif high_b[0] < low_a[0]:
                predicates.append((b, "negative_x_of", a))
            if low_a[2] > high_b[2]+0.05:
                predicates.append((a, "above", b))
            elif low_b[2] > high_a[2]+0.05:
                predicates.append((b, "above", a))
            separation = np.maximum(np.maximum(low_a-high_b, low_b-high_a), 0)
            if np.linalg.norm(separation) < 1.:
                predicates.append((a, "near", b))
            if np.all(np.minimum(high_a, high_b)-np.maximum(low_a, low_b) > 0.02):
                predicates.append((a, "intersects", b))
            for subject, predicate, target in predicates:
                result.append(dict(subject_id=subject["object_id"], predicate=predicate, object_id=target["object_id"],
                    frame_index=ra["frame_index"], timestamp_seconds=ra["timestamp_seconds"],
                    confidence=confidence(0.65, "inferred", "simultaneous_observed_aabb_relation", "World axes (+Z up); partial surface bounds, not complete physical extents")))
    return result


def build_scene(frames, provider, output, *, scene_id, data_source, fps, metric_frames=None,
                min_depth_m=None, max_depth_m=None, pixel_cloud_paths=None, raw_camera_source=None, input_manifest=None):
    """frames: RGB keyframe dictionaries; metric_frames: reviewed Stage B inputs.

    The whole batch is checked before generating output. Explicitly partial
    RGB-only input is supported; a failed metric contract NEVER falls back.
    """
    if not frames or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Nonempty frames and explicit positive FPS required")
    if any(b["frame_index"] <= a["frame_index"] or b["timestamp_seconds"] <= a["timestamp_seconds"] for a, b in zip(frames, frames[1:])):
        raise ValueError("Keyframes must have strictly increasing source indices/timestamps")
    metric = metric_frames is not None
    mf = {f.frame_index: f for f in metric_frames or []}
    if metric:
        if len(mf) != len(metric_frames) or set(mf) != {r["frame_index"] for r in frames}:
            raise ValueError("Metric and RGB frame sets must match exactly")
        if min_depth_m is None or max_depth_m is None or not np.isfinite([min_depth_m, max_depth_m]).all() or not 0 < min_depth_m < max_depth_m:
            raise ValueError("Reviewed geometry requires explicit finite depth limits")
        seen = {}
        for f in metric_frames:
            f.validate()
            digest = input_digest(f)
            if f.source_observation_id in seen and seen[f.source_observation_id] != digest:
                raise ValueError("Duplicate source identity contradicts RGB/depth/pose")
            seen[f.source_observation_id] = digest
    for row in frames:
        rgb = row["rgb"]
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Native RGB uint8 HxWx3 required")
        for field in ("frame_index", "original_tick"):
            if isinstance(row[field], bool) or not isinstance(row[field], int) or row[field] < 0:
                raise ValueError("Explicit nonnegative source index/tick required")
        if not np.isfinite(row["timestamp_seconds"]) or row["timestamp_seconds"] < 0 or not row["source_observation_id"]:
            raise ValueError("Source seconds and independent observation identity required")
        if metric:
            f = mf[row["frame_index"]]
            if not np.array_equal(rgb, f.rgb) or row["timestamp_seconds"] != f.timestamp_seconds or row["original_tick"] != f.original_tick or row["source_observation_id"] != f.source_observation_id:
                raise ValueError("RGB records contradict metric frame provenance")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/"masks").mkdir(exist_ok=True)
    (output/"debug").mkdir(exist_ok=True)
    tracker = PersistentTracker(scene_id)
    camera_track, timeline_rows = [], []
    for row in frames:
        index, timestamp, rgb = row["frame_index"], row["timestamp_seconds"], row["rgb"]
        points = pixels = None
        if metric:
            path = (pixel_cloud_paths or {}).get(index)
            points, pixels = load_pixel_cloud(path, mf[index], min_depth_m, max_depth_m) if path else frame_points(mf[index], min_depth_m, max_depth_m)
        detections = provider.detect(rgb, index)
        observations = []
        canvas = Image.fromarray(rgb.copy())
        draw = ImageDraw.Draw(canvas)
        for ordinal, d in enumerate(detections):
            d.validate(rgb)
            mask_path = f"masks/frame_{index:06d}_{ordinal:03d}.png"
            Image.fromarray(d.mask.astype(np.uint8)*255).save(output/mask_path)
            measurement, c = measure(d, points, pixels)
            observations.append(Observation(index, timestamp, row["source_observation_id"], d, measurement, c, appearance(rgb, d.mask), mask_path))
            draw.rectangle(d.bbox, outline="red", width=max(1, rgb.shape[1]//600))
            draw.text((d.bbox[0], max(0, d.bbox[1]-12)), d.category+" "+str(round(d.confidence["score"], 2)), fill="white")
        tracker.update(observations, timestamp)
        # Overlay masks, boxes and evidence labels, not just point-cloud plots.
        tint = np.asarray(canvas).copy().astype(float)
        for d in detections:
            tint[d.mask] = tint[d.mask]*0.75+np.array([255, 40, 70])*0.25
        Image.fromarray(tint.astype(np.uint8)).save(output/f"debug/frame_{index:06d}.png")
        timeline_rows.append(dict(frame_index=index, timestamp_seconds=timestamp, original_tick=row["original_tick"], camera_id="camera_main", source_observation_id=row["source_observation_id"]))
        camera_track.append(dict(frame_index=index, timestamp_seconds=timestamp,
            world_from_camera=mf[index].world_from_camera.tolist() if metric else None,
            intrinsics=mf[index].intrinsics.tolist() if metric else None,
            confidence=confidence(0.95, "inferred", "reviewed_input_contract", mf[index].sync_evidence) if metric else unknown("Source SDK rotation/render-pose contract unverified; raw telemetry retained separately")))
    objects = []
    point_folder = output/"objects"
    point_folder.mkdir(exist_ok=True)
    for track in tracker.tracks:
        static, motion_conf = classify_motion(track)
        measured = [o for o in independent_observations(track) if o.measurement is not None]
        pose = dims = cloud_path = None
        geometry_kind = "unobserved"
        geom_conf = unknown("No reviewed mask/depth geometry")
        if measured:
            if static is True:
                position, dimensions, selected = static_bounds([o.measurement for o in measured])
            else:
                # Dynamic/unknown tracks NEVER pool world points across time.
                first = measured[0].measurement
                position, dimensions, selected = first["position"], first["dimensions"], first["points"]
            pose, dims = object_transform(position), dimensions.tolist()
            cloud_path = f"objects/{track.object_id}.ply"
            cloud = point_cloud(selected, np.full(selected.shape, 160, dtype=np.uint8))
            write_verified_ply(output/cloud_path, cloud, data_source=data_source)
            geometry_kind = "point_cloud_mask" if pixel_cloud_paths else "depth_mask"
            geom_conf = measured[0].geometry_confidence.copy()
        obs_rows = []
        for o in track.observations:
            m = o.measurement
            obs_rows.append(dict(frame_index=o.frame_index, timestamp_seconds=o.timestamp_seconds, camera_id=o.camera_id,
                bbox_xyxy=o.detection.bbox, raster_size=[o.detection.mask.shape[1], o.detection.mask.shape[0]],
                mask=dict(path=o.mask_path, kind=o.detection.mask_kind, confidence=o.detection.mask_confidence),
                position=m["position"].tolist() if m else None, dimensions=m["dimensions"].tolist() if m else None,
                geometry_confidence=o.geometry_confidence, point_count=len(m["points"]) if m else 0))
        motion_track = []
        if static is not True:
            for o in measured:
                motion_track.append(dict(frame_index=o.frame_index, timestamp_seconds=o.timestamp_seconds,
                    transform=object_transform(o.measurement["position"]), confidence=o.geometry_confidence))
        source_frames = sorted({o.frame_index for o in track.observations})
        objects.append(dict(object_id=track.object_id, category=track.category, primitive_type=PRIMITIVES[track.category],
            transform=pose, dimensions=dims, parent_id=None, is_static=static, classification_confidence=motion_conf,
            confidence=track.observations[0].detection.confidence,
            tracking_confidence=confidence(0.65 if track.hint else 0.7, "inferred", "assisted_identity_hint" if track.hint else track.association_method,
                "Scene-scoped deterministic ID; conservative category/world-motion/appearance association; ambiguous matches abstain. No zero-ID-switch claim on real RGB"),
            geometry_source=dict(kind=geometry_kind, point_cloud_path=cloud_path, bounds_type="observed_surface_aabb" if pose else "unobserved", confidence=geom_conf),
            source_frames=source_frames, observations=obs_rows, motion_track=motion_track,
            visibility=[dict(frame_index=r["frame_index"], timestamp_seconds=r["timestamp_seconds"], camera_id="camera_main", state="visible" if r["frame_index"] in source_frames else "not_detected") for r in timeline_rows]))
    complete = metric and bool(objects) and all(o["transform"] is not None for o in objects)
    categories = sorted({o["category"] for o in objects})
    outdoor = {"tree", "ground"}.issubset(categories)
    scene = dict(schema_version="1.0.0", scene_id=scene_id,
        scene_type=dict(value="outdoor_natural" if outdoor else "unknown",
            confidence=confidence(0.65, "inferred", "rgb_category_context", "Scene context from RGB categories; not a benchmarked scene classifier") if outdoor else unknown("Insufficient RGB category context")),
        created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), coordinate_system="blender_rh_z_up", unit_scale=1.,
        objects=objects, cameras=[dict(camera_id="camera_main", local_axes="optical_x_right_y_down_z_forward", track=camera_track, raw_source=raw_camera_source)],
        timeline=dict(fps=float(fps), start_seconds=frames[0]["timestamp_seconds"], end_seconds=frames[-1]["timestamp_seconds"], time_origin="source clip seconds, not QPC", frames=timeline_rows),
        relationships=relationships(objects), reconstruction_metadata=dict(status="complete" if complete else "partial_observations" if metric else "partial_geometry_unverified",
            data_source=data_source, geometry_contract_verified=metric, provider=provider.name, input_manifest=input_manifest,
            limitations=["Supported categories do not certify detector accuracy; annotations are assisted, not benchmarked ground truth.", "Dimensions/centres are observed-surface AABBs; hidden extent and true orientation are not recovered. Identity rotation is blockout convention.", "Static classification can be biased by partial visibility; ambiguous motion remains null.", "Sparse keyframes do not prove full-clip recall or continuous visibility; not_detected is not proof of occlusion.", "Camera matrices use optical local axes; Stage E must convert to Blender camera-local axes.", "Real sample remains local internal evaluation; no upload or additional sample download."]))
    validation = validate_scene(scene)
    json_write(output/"scene_schema.json", scene)
    report = dict(status=scene["reconstruction_metadata"]["status"], schema_validation=validation,
        categories=categories, object_count=len(objects), objects_with_geometry=sum(o["transform"] is not None for o in objects),
        static=sum(o["is_static"] is True for o in objects), dynamic=sum(o["is_static"] is False for o in objects), unknown_motion=sum(o["is_static"] is None for o in objects),
        observations=sum(len(o["observations"]) for o in objects), relationships=len(scene["relationships"]),
        real_semantic_accuracy_validated=False, evidence_boundary="Synthetic tests verify code, not real detector accuracy or dataset depth contract")
    json_write(output/"validation_report.json", report)
    # Small geometric debug view with world axes and camera positions.
    if metric and objects:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        for o in objects:
            coords = [r["position"] for r in o["observations"] if r["position"] is not None]
            if coords:
                p = np.asarray(coords)
                ax.plot(p[:, 0], p[:, 1], ".-", label=o["category"]+" "+o["object_id"][-4:])
        c = np.array([mf[r["frame_index"]].world_from_camera[:3, 3] for r in frames])
        ax.plot(c[:, 0], c[:, 1], "kx--", label="camera")
        ax.set(xlabel="World X (m)", ylabel="World Y (m)", title="Measured object centres / camera track (XY)")
        ax.axis("equal")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output/"debug/world_tracks.png", dpi=130)
        plt.close(fig)
    return scene, report
