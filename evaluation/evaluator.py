"""Reviewed dataset GT or explicit synthetic fixtures; partial reports on missing evidence."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.geometry_metrics import geometry_metrics
from evaluation.report import add_comparison, add_metric, new_report, write_reports
from evaluation.temporal_metrics import camera_metrics, synchronization_metrics
from evaluation.semantic_metrics import semantic_metrics
from evaluation.visual_metrics import lpips, ssim
from reconstruction.backprojection import backproject
from reconstruction.coordinate_transform import transform_points


MISSING = dict(geometry=["chamfer_distance_m", "f_score", "point_cloud_coverage"], camera=["camera_translation_error_m_rmse", "camera_rotation_error_deg_rmse", "trajectory_ate_m_rmse", "trajectory_rpe_translation_m_rmse"],
               semantic=["detection_precision", "detection_recall", "category_accuracy", "static_dynamic_accuracy", "object_tracking_id_switches"],
               synchronization=["sync_error_s_max"], visual=["ssim", "lpips"])


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def truth_contract(descriptor):
    if not isinstance(descriptor, dict) or descriptor.get("provenance") != "ground_truth" or descriptor.get("verified") is not True or not isinstance(descriptor.get("evidence"), str) or not descriptor["evidence"].strip():
        raise ValueError("Ground Truth requires explicit ground_truth provenance, verified=true and review evidence; inferred is not GT")


def load_points(path):
    path = Path(path)
    if path.suffix == ".npz":
        from zipfile import ZipFile
        with ZipFile(path) as archive:
            if sum(i.file_size for i in archive.infolist()) > 512*1024**2:
                raise ValueError("Point cloud NPZ exceeds 512 MiB decoded budget")
        with np.load(path, allow_pickle=False) as data:
            return data["points_world_m"].copy()
    if path.suffix == ".ply":
        import open3d as o3d
        if path.stat().st_size > 512*1024**2:
            raise ValueError("PLY exceeds 512 MiB input budget")
        return np.asarray(o3d.io.read_point_cloud(str(path)).points)
    raise ValueError("Point clouds must be PLY or NPZ with points_world_m")


def scene_observations(path):
    scene = read_json(path)
    result = []
    for obj in scene["objects"]:
        for row in obj["observations"]:
            result.append(dict(frame_index=row["frame_index"], object_id=obj["object_id"], category=obj["category"],
                bbox_xyxy=row["bbox_xyxy"], is_static=obj["is_static"], raster_size=row["raster_size"]))
    return result


def evaluate(config, output, *, base=None):
    if config.get("schema") != "stage_g.config/1" or not isinstance(config.get("data_source"), str):
        raise ValueError("Explicit stage_g.config/1 and data_source required")
    base = Path(base or ".").resolve()
    path = lambda value: (base/value).resolve()
    report = new_report(config["data_source"])
    report["evidence"] = config
    prediction = config.get("prediction", {})
    provenance = prediction.get("provenance", "unobserved")
    if provenance not in ("predicted", "inferred", "unobserved"):
        raise ValueError("Prediction must be predicted, inferred or unobserved, never ground_truth")

    def missing(group, reason, *, failure=False):
        for name in MISSING[group]:
            add_metric(report, group, name, None, reason=reason, provenance=provenance)
        if failure:
            report["failures"].append(dict(group=group, error=reason))

    frames = None
    gt = config.get("ground_truth")
    if gt and gt.get("metric_manifest"):
        try:
            truth_contract(gt)
            from scripts.reconstruct_scene import load_metric_manifest
            frames, _ = load_metric_manifest(path(gt["metric_manifest"]))
        except (ValueError, KeyError, TypeError, OSError) as error:
            report["failures"].append(dict(group="ground_truth", error=str(error)))
    try:
        if frames is None or not prediction.get("point_cloud"):
            missing("geometry", "No verified GT metric Depth/render-pose and candidate point cloud")
        else:
            if prediction.get("coordinate_system") != "blender_rh_z_up" or prediction.get("unit") != "m":
                raise ValueError("Candidate point cloud must explicitly declare blender_rh_z_up and m; no scale fitting")
            options = config.get("geometry", {})
            if "min_depth_m" not in options or "max_depth_m" not in options:
                raise ValueError("Evaluation requires explicit reviewed depth limits")
            clouds, seen = [], set()
            for frame in frames:
                if frame.source_observation_id in seen:
                    continue
                seen.add(frame.source_observation_id)
                bp = backproject(frame.depth, frame.intrinsics, spec=frame.depth_spec, valid_mask=frame.valid_mask, rgb=frame.rgb,
                    min_depth_m=options["min_depth_m"], max_depth_m=options["max_depth_m"], stride=options.get("stride", 2))
                clouds.append(transform_points(bp.points_camera_m, frame.world_from_camera))
                if sum(len(c) for c in clouds) > 2000000:
                    raise ValueError("GT accumulation exceeds 2M points; increase explicit sampling stride")
            comparison = geometry_metrics(load_points(path(prediction["point_cloud"])), np.concatenate(clouds),
                thresholds_m=options.get("thresholds_m", [.05, .1]), max_points=options.get("max_points", 200000), seed=options.get("seed", 0))
            add_comparison(report, "geometry", comparison, provenance=provenance,
                scope="Candidate reconstruction cloud versus accumulated observed GT Depth surfaces; dynamic samples are not a static scene mesh")
    except (ValueError, KeyError, OSError, TypeError) as error:
        missing("geometry", str(error), failure=True)
    try:
        gt_poses = None
        if gt and gt.get("camera_poses"):
            truth_contract(gt)
            from blender.planner import load_camera_poses
            gt_poses = [r for rows in load_camera_poses(path(gt["camera_poses"])).values() for r in rows]
        elif frames is not None:
            gt_poses = [dict(frame_index=f.frame_index, camera_id="camera_main", timestamp_seconds=f.timestamp_seconds, world_from_camera=f.world_from_camera.tolist()) for f in frames]
        pred_poses = None
        if prediction.get("camera_poses"):
            # Predictions have a separate evaluation pose format; verified
            # stage_e poses are intentionally NOT used as estimated poses.
            document = read_json(path(prediction["camera_poses"]))
            if document.get("schema") != "stage_g.camera_prediction/1" or document.get("coordinate_system") != "blender_rh_z_up" or document.get("translation_unit") != "m" or document.get("pose_direction") != "world_from_optical_camera":
                raise ValueError("Explicit stage_g.camera_prediction/1 metric optical-to-world contract required")
            pred_poses = document["samples"]
        elif prediction.get("scene_json"):
            scene = read_json(path(prediction["scene_json"]))
            if scene.get("coordinate_system") != "blender_rh_z_up" or scene.get("unit_scale") != 1.:
                raise ValueError("Candidate Scene must use Blender world/metres")
            pred_poses = [dict(r, camera_id=c["camera_id"]) for c in scene["cameras"] for r in c["track"] if r.get("world_from_camera") is not None]
        if gt_poses is None or pred_poses is None:
            missing("camera", "No verified GT Camera Pose or independent candidate pose track")
        else:
            comparison = camera_metrics(pred_poses, gt_poses, **config.get("camera", {}))
            if prediction.get("camera_poses"):
                comparison["details"]["prediction_metadata"] = {key: value for key, value in document.items() if key != "samples"}
            add_comparison(report, "camera", comparison, provenance=provenance,
                scope="When cameras are copied from GT this measures pipeline pose retention, not pose estimation accuracy")
    except (ValueError, KeyError, OSError, TypeError) as error:
        missing("camera", str(error), failure=True)
    try:
        semantic = config.get("semantic")
        if not semantic or not prediction.get("scene_json"):
            missing("semantic", "No exhaustive GT instance annotations; assisted/inferred detections are not ground truth")
        else:
            truth_contract(semantic)
            document = read_json(path(semantic["annotations"]))
            if document.get("schema") != "stage_g.semantic_truth/1" or document.get("exhaustive") is not True:
                raise ValueError("Exhaustive stage_g.semantic_truth/1 required, with explicit empty annotation frames")
            truth_contract(document)
            pred = scene_observations(path(prediction["scene_json"]))
            sizes = {row["frame_index"]: row["raster_size"] for row in document["frames"]}
            if any(r["frame_index"] in sizes and r["raster_size"] != sizes[r["frame_index"]] for r in pred):
                raise ValueError("Semantic GT/candidate raster mismatch; implicit scaling forbidden")
            truth = [dict(obj, frame_index=row["frame_index"], raster_size=row["raster_size"]) for row in document["frames"] for obj in row["objects"]]
            add_comparison(report, "semantic", semantic_metrics(pred, truth, evaluated_frames=[r["frame_index"] for r in document["frames"]],
                iou_threshold=semantic.get("iou_threshold", .5), max_id_gap_frames=semantic.get("max_id_gap_frames")), provenance=provenance)
    except (ValueError, KeyError, OSError, TypeError) as error:
        missing("semantic", str(error), failure=True)
    try:
        sync = config.get("synchronization")
        if not sync:
            missing("synchronization", "No reviewed common-origin stream timestamps")
        else:
            if sync.get("verified") is not True or sync.get("unit") != "s" or not sync.get("evidence") or not sync.get("time_origin"):
                raise ValueError("Synchronization needs verified common-origin seconds and evidence; never raw QPC")
            streams = read_json(path(sync["streams"]))["streams"] if isinstance(sync["streams"], str) else sync["streams"]
            for name, comparison in synchronization_metrics(streams, reference=sync.get("reference", "rgb")).items():
                add_comparison(report, "synchronization."+name, comparison, provenance=provenance, scope=sync["evidence"])
    except (ValueError, KeyError, OSError, TypeError) as error:
        missing("synchronization", str(error), failure=True)
    visual = config.get("visual", {})
    pairs = visual.get("pairs", [])
    scores, perceptual = [], []
    for pair in pairs:
        try:
            with Image.open(path(pair["predicted"])) as image:
                p = np.asarray(image.convert("RGB"))
            with Image.open(path(pair["reference"])) as image:
                g = np.asarray(image.convert("RGB"))
            scores.append(ssim(p, g))
            if visual.get("lpips_backbone_weights") and visual.get("lpips_linear_weights"):
                perceptual.append(lpips(p, g, backbone_weights=path(visual["lpips_backbone_weights"]), linear_weights=path(visual["lpips_linear_weights"])))
        except (ValueError, KeyError, OSError, ImportError, RuntimeError) as error:
            report["failures"].append(dict(group="visual", error=str(error)))
    add_metric(report, "visual", "ssim", float(np.mean(scores)) if scores else None, reason="No valid equal-raster source-frame image pairs", provenance=provenance)
    add_metric(report, "visual", "lpips", float(np.mean(perceptual)) if perceptual else None, reason="LPIPS optional: needs local lpips package plus explicit backbone AND linear weights; no auto-download", provenance=provenance)
    report["visual_pair_counts"] = dict(requested=len(pairs), ssim_evaluated=len(scores), lpips_evaluated=len(perceptual))
    report["status"] = "partial" if report["failures"] or any(r["status"] != "measured" for r in report["metrics"] if r["group"] != "visual") else "evaluated"
    write_reports(output, report)
    return report


def evaluate_existing_sample(prepared, output):
    """Honest local dataset evidence, without using raw log codes as metric GT."""
    prepared = Path(prepared)
    audit = read_json(prepared/"validation_report.json")
    report = new_report("REAL RESTRICTED DATASET SAMPLE — local internal evaluation; original retention/deletion terms apply")
    report["evidence"] = dict(prepared=str(prepared.resolve()), audit=audit, download_performed=False)
    for group in ("geometry", "camera", "semantic"):
        for name in MISSING[group]:
            add_metric(report, group, name, None, reason="Dataset metric Depth/render-pose not verified or exhaustive semantic GT absent; raw telemetry consistency is not reconstruction accuracy", provenance="unobserved")
    value = audit.get("timestamp_max_rgb_depth_meta_error_seconds")
    add_metric(report, "synchronization.depth", "sync_error_s_max", value, unit="s", scope="Existing Stage A decoded RGB PTS versus Depth Metadata delivery timestamps; NOT depth render-camera latch verification")
    mismatches = audit.get("camera_table_pose_mismatch_frames")
    add_metric(report, "dataset_consistency", "camera_table_pose_mismatch_count", len(mismatches) if isinstance(mismatches, list) else None, unit="count", scope="Raw JSONL vs Parquet storage consistency, NOT estimated Pose Error")
    for name in MISSING["visual"]:
        add_metric(report, "visual", name, None, reason="No independently paired candidate frames/optional LPIPS weights", provenance="unobserved")
    report["limitations"].append("Current Session v3 inverse/depth units and SDK render-pose latch remain unverified; this partial report is not real A-G acceptance.")
    write_reports(output, report)
    return report
