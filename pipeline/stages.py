"""Adapters reuse existing A-F kernels, with Stage E and F separate checkpoints."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image

from evaluation.evaluator import evaluate, evaluate_existing_sample, read_json
from evaluation.report import new_report, write_json, write_reports

ROOT = Path(__file__).resolve().parents[1]


def _ground_truth_annotations(frames, output):
    """Known synthetic scene labels, not GT made by evaluating a detector."""
    from agents.object_detection import SyntheticColorProvider, mask_bbox
    rows = []
    for frame in frames:
        objects = []
        # Palette was assigned by the independent analytic renderer's labels.
        # Only this synthetic fixture may use its exact colors as an oracle.
        for category, color in SyntheticColorProvider.palette.items():
            mask = np.all(frame.rgb == color, axis=-1)
            if mask.any():
                objects.append(dict(object_id="analytic-"+category, category=category, bbox_xyxy=mask_bbox(mask), is_static=category != "person"))
        rows.append(dict(frame_index=frame.frame_index, raster_size=[frame.rgb.shape[1], frame.rgb.shape[0]], objects=objects))
    write_json(output/"semantic_truth.json", dict(schema="stage_g.semantic_truth/1", provenance="ground_truth", verified=True,
        evidence="SYNTHETIC analytic renderer instance labels only; exact palette encodes oracle masks, not real model accuracy", exhaustive=True, frames=rows))


def prepare_synthetic(config, output):
    from blender.synthetic_demo import demo_frame
    from scripts.reconstruct_scene import save_synthetic_inputs
    from stage_a.media import ffmpeg_path
    options = config.get("synthetic", {})
    fps, count, stride = options.get("fps", 60), options.get("frame_count", 720), options.get("keyframe_stride", 120)
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (fps, count, stride)) or count > 900 or count < 2:
        raise ValueError("Synthetic FPS/count/stride must be positive integers, count 2..900")
    indices = sorted({0, count-1, *range(0, count, stride)})
    frames = [demo_frame(i, fps) for i in indices]
    source = "SYNTHETIC analytic ray/AABB fixture ONLY — not dataset acceptance"
    save_synthetic_inputs(output, frames, dict(data_source=source))
    camera, timestamps = [], []
    command = [ffmpeg_path(), "-y", "-nostdin", "-hide_banner", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "240x140",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output/"original_video.mp4")]
    with (output/"encode_original.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log, stdout=log)
        try:
            for i in range(count):
                frame = demo_frame(i, fps)
                process.stdin.write(frame.rgb.tobytes())
                camera.append(dict(camera_id="camera_main", frame_index=frame.frame_index, original_tick=frame.original_tick,
                    timestamp_seconds=frame.timestamp_seconds, world_from_camera=frame.world_from_camera.tolist(), intrinsics=frame.intrinsics.tolist()))
                timestamps.append(dict(frame_index=frame.frame_index, timestamp_seconds=frame.timestamp_seconds))
            process.stdin.close()
            if process.wait(timeout=120):
                raise RuntimeError("Synthetic source video encoding failed")
        finally:
            if not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    write_json(output/"camera_poses.json", dict(schema="stage_e.camera_poses/1", verified=True, evidence=source,
        pose_format="world_from_optical_camera", coordinate_system="blender_rh_z_up", translation_unit="m", samples=camera))
    write_json(output/"streams.json", dict(streams={key: timestamps for key in ("rgb", "depth", "camera")}))
    _ground_truth_annotations(frames, output)
    for frame in frames:
        Image.fromarray(frame.rgb).save(output/f"rgb_{frame.frame_index:06d}.png")
    prepared = dict(mode="synthetic", data_source=source, metric_manifest=str(output/"synthetic_input_manifest.json"),
        original_video=str(output/"original_video.mp4"), camera_poses=str(output/"camera_poses.json"), fps=fps, geometry_contract_verified=True)
    write_json(output/"prepared.json", prepared)
    return prepared


def execute_stage(stage, config, stages, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    folders = {key: Path(value["output_directory"]) for key, value in stages.items() if value["status"] in ("completed", "cached")}
    mode = config["mode"]
    if stage == "A":
        if mode == "synthetic":
            prepare_synthetic(config, output)
            return 0
        if mode == "existing_sample":
            from reconstruction.validation import audit_sample
            prepared = Path(config["prepared_sample"]) if not config.get("prepare_raw_sample") else output/"prepared_sample"
            if config.get("prepare_raw_sample"):
                from scripts.prepare_sample import main as prepare_sample
                prepare_sample(source=config["sample_source"], output=prepared)
            report = audit_sample(Path(config["sample_source"]), prepared)
            report.pop("candidate_poses", None)
            write_json(output/"dataset_validation.json", report)
            write_json(output/"prepared.json", dict(mode=mode, prepared_sample=str(prepared), status="blocked", blockers=report["blockers"]))
            return 2
        from scripts.reconstruct_scene import load_metric_manifest
        from blender.media import probe_video
        frames, source = load_metric_manifest(Path(config["input_manifest"]))
        video = probe_video(config["original_video"], log_path=output/"source_video.log", timeout=config.get("timeout_seconds", 3600),
            start_frame=config.get("video_start_frame", 0), frame_count=config.get("video_frame_count"))
        fps = float(__import__("fractions").Fraction(video["fps"]))
        # Validate the supplied source index/time lattice against this clip,
        # without changing timestamps or interpreting QPC.
        origin_index, origin_time = frames[0].frame_index, frames[0].timestamp_seconds
        for frame in frames:
            offset = (frame.timestamp_seconds-origin_time)*fps
            if abs(offset-round(offset)) > 1e-6 or frame.frame_index != origin_index+round(offset):
                raise ValueError("Reviewed RGB/Depth/Camera index-time lattice mismatch")
        if frames[-1].timestamp_seconds-origin_time > (video["frame_count"]-1)/fps+1e-6:
            raise ValueError("Reviewed frames exceed video window")
        write_json(output/"prepared.json", dict(mode=mode, data_source=source["data_source"], metric_manifest=config["input_manifest"],
            original_video=config["original_video"], camera_poses=config.get("camera_poses"), fps=fps, geometry_contract_verified=True, video=video))
        return 0
    if stage == "G":
        if mode == "existing_sample":
            prepared = config.get("prepared_sample")
            if config.get("prepare_raw_sample"):
                attempt = stages.get("A", {}).get("last_attempt_output_directory")
                snapshot = Path(attempt)/"prepared.json" if attempt else None
                if snapshot is None or not snapshot.is_file():
                    report = new_report("Dataset preparation failed; no stale sample substituted")
                    report["failures"].append(dict(stage="A", error=stages.get("A", {}).get("error", "Preparation unavailable")))
                    write_reports(output, report)
                    return 0
                prepared = read_json(snapshot)["prepared_sample"]
            evaluate_existing_sample(prepared, output)
            return 0
        if "A" not in folders:
            report = new_report("Pipeline input unavailable; no truth substituted")
            report["failures"] = [dict(stage=k, status=v["status"], error=v.get("error", "")) for k, v in stages.items() if v["status"] not in ("completed", "cached")]
            write_reports(output, report)
            return 0
        settings = evaluation_config(config, folders)
        write_json(output/"evaluation_config.json", settings)
        evaluate(settings, output)
        return 0
    prepared = read_json(folders["A"]/"prepared.json")
    manifest = Path(prepared["metric_manifest"])
    reconstruction = config.get("reconstruction", {})
    limits = dict(min_depth_m=reconstruction["min_depth_m"], max_depth_m=reconstruction["max_depth_m"])
    if stage == "B":
        from scripts.reconstruct_scene import load_metric_manifest, reconstruct
        frames, metadata = load_metric_manifest(manifest)
        return reconstruct(frames, metadata, output, **limits, voxel_size_m=reconstruction.get("voxel_size_m", .08), stride=reconstruction.get("stride", 2))
    if stage == "C":
        from scripts.reconstruct_scene import load_metric_manifest
        from scripts.understand_scene import rgb_records
        from agents.object_detection import AnnotationProvider, LocalVisionProvider, SyntheticColorProvider
        from agents.scene_builder import build_scene
        frames, _ = load_metric_manifest(manifest)
        if mode == "synthetic":
            provider = SyntheticColorProvider()
        elif config.get("annotations"):
            provider = AnnotationProvider(config["annotations"])
        else:
            provider = LocalVisionProvider(config["local_detector"], config.get("local_sam"))
        _, report = build_scene(rgb_records(frames), provider, output, scene_id=config.get("scene_id", "pipeline-scene"), data_source=prepared["data_source"],
            fps=prepared["fps"], metric_frames=frames, input_manifest=str(manifest), **limits)
        return 0 if report["status"] == "complete" else 2
    if stage == "D":
        from scene_schema import validate_scene
        scene = read_json(folders["C"]/"scene_schema.json")
        validation = validate_scene(scene)
        write_json(output/"schema_validation.json", validation)
        return 0
    options = config.get("render", {})
    if stage == "E":
        from blender.pipeline import run_pipeline
        run_pipeline(folders["C"]/"scene_schema.json", prepared["original_video"], output,
            blender=config.get("blender"), camera_poses=prepared.get("camera_poses"), point_cloud=folders["B"]/"scene_pointcloud.ply",
            resolution=options.get("resolution"), engine=options.get("engine", "BLENDER_WORKBENCH"), samples=options.get("samples", 16),
            video_start_frame=config.get("video_start_frame", 0), video_frame_count=config.get("video_frame_count"), generate_only=True,
            timeout=config.get("timeout_seconds", 3600))
        return 0
    if stage == "F":
        from blender.pipeline import background, blender_path
        from blender.media import encode_outputs
        generation = read_json(folders["E"]/"pipeline_report.json")
        for name in ("scene.blend", "scene_schema.json", "build_plan.json", "frame_mapping.json"):
            shutil.copy2(folders["E"]/name, output/name)
        background(blender_path(config.get("blender")), ROOT/"blender/render.py", output, blend=output/"scene.blend", evaluation_poses=True, timeout=config.get("timeout_seconds", 3600))
        plan = read_json(output/"build_plan.json")
        media = encode_outputs(output, generation["source_video"], plan["resolution"], timeout=config.get("timeout_seconds", 3600))
        write_json(output/"render_report.json", dict(status="passed", media=media, source_generation=str(folders["E"])))
        return 0
    raise ValueError("Unknown pipeline stage")


def evaluation_config(config, folders):
    settings = deepcopy(config.get("evaluation", {}))
    prepared = read_json(folders["A"]/"prepared.json")
    settings.setdefault("schema", "stage_g.config/1")
    settings.setdefault("data_source", prepared["data_source"])
    settings.setdefault("prediction", dict(provenance="inferred", coordinate_system="blender_rh_z_up", unit="m"))
    if "B" in folders:
        settings["prediction"].setdefault("point_cloud", str(folders["B"]/"scene_pointcloud.ply"))
    if "C" in folders:
        settings["prediction"].setdefault("scene_json", str(folders["C"]/"scene_schema.json"))
    if "F" in folders:
        settings["prediction"].setdefault("camera_poses", str(folders["F"]/"camera_pose_evaluation.json"))
    settings.setdefault("geometry", dict(config["reconstruction"]))
    # Keep only evaluator options, never silently forward fusion parameters.
    settings["geometry"] = {key: value for key, value in settings["geometry"].items() if key in {"min_depth_m", "max_depth_m", "stride", "thresholds_m", "max_points", "seed"}}
    if config["mode"] == "synthetic":
        settings["ground_truth"] = dict(provenance="ground_truth", verified=True, evidence="SYNTHETIC analytic fixture; GT input consistency only, not independent pose estimation", metric_manifest=prepared["metric_manifest"],
                                       camera_poses=prepared["camera_poses"] if "F" in folders else None)
        settings.setdefault("camera", dict(rpe_delta_frames=config.get("synthetic", {}).get("keyframe_stride", 120)))
        settings["semantic"] = dict(provenance="ground_truth", verified=True, evidence="Synthetic renderer instance oracle only", annotations=str(folders["A"]/"semantic_truth.json"))
        settings["synchronization"] = dict(verified=True, unit="s", time_origin="synthetic clip seconds", evidence="Same-call analytic RGB/depth/camera timestamps, not SDK latch inference", streams=str(folders["A"]/"streams.json"))
        if "F" in folders:
            source = read_json(prepared["metric_manifest"])
            origin = source["frames"][0]["frame_index"]
            settings.setdefault("visual", {})["pairs"] = [dict(frame_index=row["frame_index"],
                reference=str(folders["A"]/f"rgb_{row['frame_index']:06d}.png"),
                predicted=str(folders["F"]/f"frames/frame_{row['frame_index']-origin+1:06d}.png")) for row in source["frames"]]
    return settings
