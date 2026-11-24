"""Refine an existing local Session without uploading or downloading video.

python -m reconstruction.refine --allow-inferred --preview-only
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import gc
import json
import os
from pathlib import Path
from functools import lru_cache
import subprocess
import tempfile
from zoneinfo import ZoneInfo

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

from agents.object_detection import confidence
from blender.media import probe_video
from blender.pipeline import ROOT, blender_path, json_write, run_pipeline, sha256
from reconstruction.refined_calibration import HOLDOUT, align_depth, calibrate, decode_empirical, features, pose_track, selected_frames
from reconstruction.refined_geometry import SurfaceFusion, bake_texture, split_chunks, write_glb
from reconstruction.refined_segmentation import CATEGORIES, LocalSegmenter
from reconstruction.refined_voxels import VoxelFusion, observation_views, palette_mesh
from scene_schema import validate_scene
from stage_a.media import iter_frames

SESSION = "66ad9535-583f-4a25-b655-6b1697c83a40"


@contextmanager
def refinement_lock(output):
    path = output/".refined.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise ValueError("Refined output is already locked by another run") from error
    with os.fdopen(fd, "w") as stream:
        stream.write(str(os.getpid()))
    try:
        yield
    except Exception as error:
        # Persist a bounded error kind, never remote URL/credential exception
        # payloads. A failed run cannot masquerade as a previous success.
        report_path = output/"refinement_report.json"
        report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        report.update(schema="refined.run/1", status="failed", error_type=type(error).__name__,
            full_video_rendered=False, failed_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
        json_write(report_path, report)
        raise
    finally:
        path.unlink()


def verified_frame_cache(cache, indices):
    path = cache/"frames.json"
    if not path.is_file():
        return None
    prior = json.loads(path.read_text())
    expected = {f"frame_{i:06d}.npz" for i in indices}
    if set(prior.get("hashes", {})) != expected:
        return None
    if not all((cache/name).is_file() and sha256(cache/name) == digest for name, digest in prior["hashes"].items()):
        return None
    return prior


def prepare_frames(video, depth_source, indices, cache, segmenter):
    """Stream both full native-raster decoders; retain only selected frames."""
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache/"frames.json"
    prior = verified_frame_cache(cache, indices)
    if prior:
        print("Verified selected-frame cache", flush=True)
        return prior
    keys, reports, hashes = set(indices), {}, {}
    rgb_frames = iter_frames(Path(video["path"]), start=0, count=video["frame_count"], max_decoded_bytes=5*1024**3)
    depth_frames = iter_frames(depth_source, start=0, count=video["frame_count"], depth=True, max_decoded_bytes=5*1024**3)
    try:
        for (index, rgb), (depth_index, codes) in zip(rgb_frames, depth_frames, strict=True):
            if depth_index != index or rgb.shape[:2] != codes.shape:
                raise ValueError("Native RGB/Depth decoded ordinals/raster disagree")
            if index not in keys:
                continue
            labels, report = segmenter.segment(rgb)
            path = cache/f"frame_{index:06d}.npz"
            np.savez_compressed(path, rgb=rgb, codes=codes, labels=labels, frame_index=index)
            hashes[path.name] = sha256(path)
            reports[str(index)] = report
            print(f"Segmented real frame {index}/{video['frame_count']-1}", flush=True)
    finally:
        rgb_frames.close()
        depth_frames.close()
    if len(hashes) != len(indices):
        raise ValueError("Selected RGB/Depth window is incomplete")
    record = dict(schema="refined.frames/1", hashes=hashes, segmentation=reports, remote_upload=False)
    json_write(manifest_path, record)
    return record


def make_scene(bundle, meshes, training, frame_loader, poses, calibration, source_rows, k, video, source_paths, *, style="textured", voxel_size=.2):
    assets = bundle/"assets"
    assets.mkdir()
    c = confidence(.65, "inferred", "empirical_static_mesh", "Held-out reprojection passed; not a verified physical Session contract")
    objects = []
    for category, mesh in meshes.items():
        for key, chunk in split_chunks(mesh, preserve_voxels=style == "voxel", voxel_size=voxel_size):
            try:
                if style == "voxel":
                    points, faces, uv, texture = palette_mesh(chunk)
                    source_frames, voxel_masks = observation_views(points, category, training, frame_loader, poses, k)
                    texture_report = dict(source_frames=source_frames, texture_mode="quantized_voxel_color")
                else:
                    points, faces, uv, texture, texture_report = bake_texture(chunk, training, frame_loader, poses, k)
            except ValueError:
                continue  # Unobserved/untexturable chunks are omitted and counted.
            low, high = points.min(0), points.max(0)
            center, dimensions = (low+high)/2, high-low
            if (dimensions <= 1e-6).any():
                continue
            name = category+"_"+"_".join(map(str, key))
            asset, cloud_path = assets/(name+".glb"), assets/(name+".ply")
            write_glb(asset, points-center, faces, uv, texture)
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[::max(1, len(points)//5000)]))
            if not o3d.io.write_point_cloud(str(cloud_path), cloud):
                raise ValueError("Reference point cloud export failed")
            source_frames = sorted(texture_report["source_frames"])
            observation_rows = []
            visibility = []
            # Representative masks suffice for static observation receipts;
            # every geometry-training frame is retained in asset provenance.
            for index in source_frames:
                rgb, depth, valid, labels = frame_loader(index)
                mask = voxel_masks[index] if style == "voxel" else valid & (labels == CATEGORIES.index(category))
                if not mask.any():
                    mask = valid & (labels != CATEGORIES.index("person"))
                y, x = np.where(mask)
                mask_path = assets/f"{name}_frame_{index:06d}_mask.png"
                Image.fromarray(mask.astype(np.uint8)*255).save(mask_path)
                time = source_rows[index]["timestamp_seconds"]
                observation_rows.append(dict(frame_index=index, timestamp_seconds=time, camera_id="camera_main",
                    bbox_xyxy=[int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)], raster_size=[video["width"], video["height"]],
                    mask=dict(path=str(mask_path.relative_to(bundle)), kind="instance", confidence=c),
                    position=center.tolist(), dimensions=dimensions.tolist(), geometry_confidence=c, point_count=len(points)))
                visibility.append(dict(frame_index=index, timestamp_seconds=time, camera_id="camera_main", state="visible"))
            objects.append(dict(object_id=name, category=category, primitive_type="mesh",
                mesh_asset=dict(path=str(asset.relative_to(bundle)), sha256=sha256(asset), format="glb"),
                transform=dict(position=center.tolist(), rotation=[0, 0, 0, 1], scale=[1, 1, 1]), dimensions=dimensions.tolist(),
                parent_id=None, is_static=True, classification_confidence=c, confidence=c, tracking_confidence=c,
                geometry_source=dict(kind="surface_mesh" if style == "voxel" or category in {"tree", "barrier", "grass", "shrub"} else "tsdf_mesh",
                    point_cloud_path=str(cloud_path.relative_to(bundle)), bounds_type="observed_mesh_aabb", confidence=c),
                source_frames=source_frames, observations=observation_rows, visibility=visibility, motion_track=[]))
            print(f"Baked editable {name}: {len(faces)} faces", flush=True)
            json_write(assets/(name+".provenance.json"), dict(style=style, geometry_training_frames=training,
                appearance_source_frames=source_frames, physical_contract_verified=False, inferred=True))
    if not objects:
        raise ValueError("No valid static mesh survived visibility/edge filtering")
    calibration_path = bundle/"calibration_report.json"
    json_write(calibration_path, calibration)
    frames = [dict(frame_index=i, timestamp_seconds=row["timestamp_seconds"], original_tick=row["original_tick"],
                   camera_id="camera_main", source_observation_id=f"real-frame-{i}") for i, row in enumerate(source_rows)]
    cameras = [dict(camera_id="camera_main", local_axes="optical_x_right_y_down_z_forward", raw_source=str(source_paths["poses"]),
                    track=[dict(frame_index=i, timestamp_seconds=row["timestamp_seconds"], world_from_camera=poses[i].tolist(),
                                intrinsics=k.tolist(), confidence=c) for i, row in enumerate(source_rows)])]
    # Background estimate uses observed sky only, never a synthetic original.
    rgb, depth, valid, labels = frame_loader(training[0])
    sky = ~valid & (np.indices(valid.shape)[0] < valid.shape[0]//3)
    color = np.median(rgb[sky], axis=0)/255 if sky.any() else np.array([.35, .4, .45])
    scene = dict(schema_version="1.1.0", scene_id="real-refined-static-"+SESSION[:8],
        scene_type=dict(value="observed_static_environment", confidence=c),
        created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), coordinate_system="blender_rh_z_up", unit_scale=1.,
        objects=objects, cameras=cameras, timeline=dict(fps=float(video["fps"]), start_seconds=frames[0]["timestamp_seconds"],
            end_seconds=frames[-1]["timestamp_seconds"], time_origin="original clip CFR seconds, source ticks retained", frames=frames),
        relationships=[], appearance=dict(texture_mode="baked_rgb", background_color=color.tolist()),
        reconstruction_metadata=dict(status="complete_inferred", data_source="REAL LOCAL SESSION / INFERRED STATIC ENVIRONMENT",
            geometry_contract_verified=False, provider="local_grounding_dino_sam+empirical_depth+"+("voxel" if style == "voxel" else "TSDF"),
            reconstruction_mode="inferred", empirical_validation=dict(path=calibration_path.name, sha256=sha256(calibration_path)),
            input_manifest=None, limitations=["Physical depth/SDK render-pose contracts remain unverified.",
                "Only observed static environment; character, skeletal animation and unseen completion excluded.",
                "Observed RGB colors retain source lighting; not relightable PBR material recovery.",
                "Flat palette voxel shells, no solid interior completion." if style == "voxel" else "Observed textured surface meshes.",
                "Model segmentation is predicted, not ground truth.", "Delete with restricted Session by 2026-12-24 Asia/Shanghai."]))
    validate_scene(scene)
    json_write(bundle/"scene_schema.json", scene)
    return scene


def render_previews(directory, frames, *, executable=None):
    directory = Path(directory).resolve()
    log_path = directory/"preview.log"
    args = [executable or blender_path(), "--background", "--factory-startup", "--threads", "4", str(directory/"scene.blend"),
            "--python-exit-code", "1", "--python", str(ROOT/"blender/render.py"), "--", str(directory/"build_plan.json"),
            str(directory), "--preview-frames", ",".join(str(i+1) for i in frames)]
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
    if result.returncode:
        raise RuntimeError(f"Background preview failed; see {log_path}")


def masked_ssim(a, b, mask):
    import cv2
    a, b = a.astype(np.float64)/255, b.astype(np.float64)/255
    ma, mb = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    va = cv2.GaussianBlur(a*a, (11, 11), 1.5)-ma*ma
    vb = cv2.GaussianBlur(b*b, (11, 11), 1.5)-mb*mb
    cov = cv2.GaussianBlur(a*b, (11, 11), 1.5)-ma*mb
    score = ((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
    interior = cv2.erode(mask.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
    return float(score[interior].mean()) if interior.any() else None


def preview_quality(output, baseline, frames, frame_loader, *, style="textured"):
    rows = []
    for index in frames:
        original, depth, valid, labels = frame_loader(index)
        refined = np.array(Image.open(output/f"frames/frame_{index+1:06d}.png").convert("RGB"))
        coarse = np.array(Image.open(baseline/f"frames/frame_{index+1:06d}.png").convert("RGB"))
        mask = valid & (labels != CATEGORIES.index("person"))
        rows.append(dict(frame_index=index, refined_masked_ssim=masked_ssim(original, refined, mask),
                         coarse_masked_ssim=masked_ssim(original, coarse, mask), evaluated_pixels=int(mask.sum())))
        if index in {0, 360, frames[-1]}:
            sheet = Image.new("RGB", (original.shape[1]*3, original.shape[0]+40), "white")
            draw = ImageDraw.Draw(sheet)
            label = "INFERRED COLOR VOXELS" if style == "voxel" else "INFERRED TEXTURED MESH"
            for slot, (name, image) in enumerate((("REAL ORIGINAL", original), ("SAME-POSE PRIMITIVES", coarse), (label, refined))):
                draw.text((slot*original.shape[1]+12, 12), name, fill="black")
                sheet.paste(Image.fromarray(image), (slot*original.shape[1], 40))
            sheet.save(output/f"comparison_frame_{index:06d}.jpg", quality=95)
    held = [r for r in rows if r["frame_index"] in HOLDOUT]
    good = [r for r in held if r["refined_masked_ssim"] is not None and r["coarse_masked_ssim"] is not None]
    improved = bool(good) and np.mean([r["refined_masked_ssim"] for r in good]) > np.mean([r["coarse_masked_ssim"] for r in good])
    report = dict(status="passed" if improved else "failed", frames=rows, heldout_frames=list(HOLDOUT),
                  comparison="same source, poses, timeline and inferred geometry bounds; not synthetic demo",
                  evidence_boundary="Masked image agreement, not ground-truth geometry or semantics")
    json_write(output/"quality_report.json", report)
    return report


def run(args):
    if not args.allow_inferred:
        raise ValueError("Refinement requires explicit --allow-inferred; original strict path remains unchanged")
    source, prepared, output = args.source_session.resolve(), args.prepared.resolve(), args.output.resolve()
    if output == prepared or output == source or source in output.parents or prepared in output.parents:
        raise ValueError("Choose an independent refined output, not the source/prepared directory")
    report_path = output/"refinement_report.json"
    if output.is_dir() and any(output.iterdir()):
        if not report_path.is_file() or json.loads(report_path.read_text()).get("schema") != "refined.run/1":
            raise ValueError("Nonempty unmanaged output; choose a new refined output directory")
    output.mkdir(parents=True, exist_ok=True)
    with refinement_lock(output):
        prior = {}
        if (output/"refinement_report.json").exists():
            prior = json.loads((output/"refinement_report.json").read_text())
            if prior.get("schema") != "refined.run/1":
                raise ValueError("Unmanaged refinement_report.json; choose a different output")
        paths = dict(video=prepared/"original_video.mp4", poses=prepared/"camera/poses.json",
                     depth=source/"depth/depth_prefix.partial.hevc", contract=source/"depth/decode_contract.json")
        for path in paths.values():
            if not path.is_file():
                raise ValueError(f"Missing local input: {path}")
        video = probe_video(paths["video"])
        rows = json.loads(paths["poses"].read_text())["frames"]
        contract = json.loads(paths["contract"].read_text())
        if video["fps"] != "60" or video["frame_count"] != 720 or len(rows) != 720:
            raise ValueError("This adapter expects the existing validated 720-frame/60-FPS sample")
        if (video["width"], video["height"]) != (1920, 1080) or any(r["clip_frame"] != i or r["rgb_frame_index"] != i
                or r["depth_frame_index"] != i or abs(r["timestamp_seconds"]-i/60) > 1e-9 for i, r in enumerate(rows)):
            raise ValueError("Source raster/frame/timestamp contract changed")
        intr = contract["intrinsics"]
        if (intr["width"], intr["height"]) != (video["width"], video["height"]):
            raise ValueError("Native depth intrinsics/raster disagree")
        k = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1.]])
        world = json.loads((source/"telemetry/world.json").read_text())
        if world["coordinate_system"] != "lh_z_up" or rows[0]["camera_raw"] is None:
            raise ValueError("Unsupported camera source")
        inputs = {str(p): sha256(p) for p in [*paths.values(), source/"telemetry/world.json", args.models/"models.lock.json"]}
        code = {str(p): sha256(p) for p in (ROOT/"reconstruction").glob("refin*.py")}
        fingerprint = hashlib.sha256(json.dumps(dict(inputs=inputs, code=code, step=args.frame_step, voxel=args.voxel_size, style=args.style), sort_keys=True).encode()).hexdigest()
        segment_code = str(ROOT/"reconstruction/refined_segmentation.py")
        cache_key = hashlib.sha256(json.dumps(dict(inputs=inputs, segmentation=code[segment_code],
            decoder=sha256(ROOT/"stage_a/media.py"), step=args.frame_step), sort_keys=True).encode()).hexdigest()
        cache = output/".refined_cache"/cache_key
        indices = selected_frames(720, args.frame_step)
        indices = sorted(set(indices) | set(HOLDOUT))
        # Safely adopt the already completed legacy cache when only downstream
        # geometry/calibration code changed. Never trust directory name alone.
        if not verified_frame_cache(cache, indices) and prior.get("input_hashes") == inputs \
                and prior.get("code_hashes", {}).get(segment_code) == code[segment_code]:
            legacy = Path(prior["frame_cache"]).resolve()
            if output/".refined_cache" in legacy.parents and verified_frame_cache(legacy, indices):
                cache = legacy
        frame_report = verified_frame_cache(cache, indices)
        if frame_report:
            print("Reusing hash-verified segmentation without loading models", flush=True)
        else:
            segmenter = LocalSegmenter(args.models)
            frame_report = prepare_frames(video, paths["depth"], indices, cache, segmenter)
            del segmenter
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        @lru_cache(maxsize=8)
        def raw(index):
            with np.load(cache/f"frame_{index:06d}.npz", allow_pickle=False) as data:
                return data["rgb"].copy(), data["codes"].copy(), data["labels"].copy()
        @lru_cache(maxsize=70)
        def feature_loader(index):
            feature_key = hashlib.sha256((frame_report["hashes"][f"frame_{index:06d}.npz"]+
                code[str(ROOT/"reconstruction/refined_calibration.py")]+"sift8000-static-rgb-v2").encode()).hexdigest()[:16]
            path = cache/f"features_{feature_key}_{index:06d}.npz"
            digest_path = path.with_suffix(".sha256")
            if path.is_file() and digest_path.is_file() and sha256(path) == digest_path.read_text():
                with np.load(path, allow_pickle=False) as data:
                    return data["points"].copy(), data["descriptors"].copy()
            rgb, codes, labels = raw(index)
            uv, desc = features(rgb, labels != CATEGORIES.index("person"))
            desc = desc if desc is not None else np.empty((0, 128), np.float32)
            np.savez_compressed(path, points=uv, descriptors=desc)
            digest_path.write_text(sha256(path))
            return uv, desc
        print("Calibrating empirical depth/camera hypotheses; holdouts excluded", flush=True)
        poses, curve, calibration = calibrate(rows, world["world_to_meters"], k, indices, feature_loader, lambda i: raw(i)[1])
        k = np.asarray(calibration["intrinsics"], float)
        json_write(output/"calibration_report.json", calibration)
        report = dict(schema="refined.run/1", status=calibration["status"], fingerprint=fingerprint, input_hashes=inputs,
            code_hashes=code, calibration=calibration, frame_cache=str(cache), frame_cache_key=cache_key, models=str(args.models.resolve()),
            segmentation=frame_report["segmentation"], style=args.style, real_physical_contract_verified=False,
            full_video_rendered=False, data_policy="Local internal evaluation; no Session downloads/uploads; delete by 2026-12-24 Asia/Shanghai")
        json_write(output/"refinement_report.json", report)
        if calibration["status"] != "passed":
            print("Real calibration rejected; no geometry fabricated and no synthetic fallback", flush=True)
            return 2
        if args.calibrate_only:
            return 0
        convention = {key: calibration["camera_convention"][key] for key in ("angular_unit", "yaw_sign", "pitch_sign")}
        depth_poses = pose_track(rows, world["world_to_meters"], field="depth_referenced_camera_raw", **convention) \
            if calibration["depth_pose_mode"] == "depth_referenced_snapshot" else poses
        @lru_cache(maxsize=8)
        def frame_loader(index):
            rgb, codes, labels = raw(index)
            depth, valid = decode_empirical(codes, curve)
            depth, valid = align_depth(depth, valid, depth_poses[index], poses[index], k)
            valid &= labels != CATEGORIES.index("person")
            return rgb, depth, valid, labels
        training = [i for i in indices if i not in HOLDOUT]
        fusion = VoxelFusion(k, args.voxel_size) if args.style == "voxel" else SurfaceFusion(k, voxel=args.voxel_size, truncation=args.voxel_size*3)
        for index in training:
            rgb, depth, valid, labels = frame_loader(index)
            if args.style == "voxel":
                fusion.integrate(rgb, depth, valid, labels, poses[index], index)
            else:
                fusion.integrate(rgb, depth, valid, labels, poses[index])
            print(f"Fused static real frame {index}", flush=True)
        meshes = fusion.extract()
        if args.style == "voxel":
            report["voxel_geometry"] = fusion.report
        bundle = Path(tempfile.mkdtemp(prefix="source_bundle_", dir=output))
        scene = make_scene(bundle, meshes, training, frame_loader, poses, calibration, rows, k, video, paths, style=args.style, voxel_size=args.voxel_size)
        executable = blender_path(args.blender)
        run_pipeline(bundle/"scene_schema.json", paths["video"], output, blender=executable,
                     engine="BLENDER_EEVEE", samples=32, allow_inferred=True, generate_only=True)
        baseline_scene = json.loads(json.dumps(scene))
        for obj in baseline_scene["objects"]:
            obj.pop("mesh_asset")
            obj["primitive_type"] = "plane" if obj["category"] == "ground" else "box"
            # Resolve source references because baseline JSON is saved elsewhere.
            obj["geometry_source"]["point_cloud_path"] = str((bundle/obj["geometry_source"]["point_cloud_path"]).resolve())
            for row in obj["observations"]:
                row["mask"]["path"] = str((bundle/row["mask"]["path"]).resolve())
        baseline_scene["reconstruction_metadata"]["empirical_validation"]["path"] = str(bundle/"calibration_report.json")
        json_write(bundle/"baseline_scene.json", baseline_scene)
        baseline = output/"baseline"
        run_pipeline(bundle/"baseline_scene.json", paths["video"], baseline, blender=executable,
                     allow_inferred=True, generate_only=True)
        previews = sorted(set(HOLDOUT) | {0, 360, 719})
        render_previews(output, previews, executable=executable)
        render_previews(baseline, previews, executable=executable)
        quality = preview_quality(output, baseline, previews, frame_loader, style=args.style)
        report.update(status="preview_passed" if quality["status"] == "passed" else "quality_rejected",
                      source_bundle=str(bundle), mesh_objects=len(scene["objects"]), quality=quality,
                      full_video_rendered=False)
        json_write(output/"refinement_report.json", report)
        if quality["status"] != "passed":
            print("Held-out image quality did not improve; keeping diagnostic previews, not rendering full video", flush=True)
            return 2
        if not args.preview_only:
            run_pipeline(bundle/"scene_schema.json", paths["video"], output, blender=executable,
                         engine="BLENDER_EEVEE", samples=32, allow_inferred=True)
            report.update(status="passed", full_video_rendered=True)
            json_write(output/"refinement_report.json", report)
        return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-session", type=Path, default=ROOT/"data/input/samples"/SESSION)
    p.add_argument("--prepared", type=Path, default=ROOT/"outputs/stage_a/sample_66ad")
    p.add_argument("--output", type=Path, default=ROOT/"outputs/stage_ef/real_refined")
    p.add_argument("--models", type=Path, default=ROOT/"data/models/refined")
    p.add_argument("--blender", type=Path)
    p.add_argument("--allow-inferred", action="store_true")
    p.add_argument("--preview-only", action="store_true")
    p.add_argument("--calibrate-only", action="store_true")
    p.add_argument("--frame-step", type=int, default=12)
    p.add_argument("--voxel-size", type=float, default=None, help="Metres; default 0.2 voxel / 0.04 textured")
    p.add_argument("--style", choices=("voxel", "textured"), default="voxel")
    args = p.parse_args(argv)
    if args.voxel_size is None:
        args.voxel_size = .2 if args.style == "voxel" else .04
    low, high = (.05, 1.) if args.style == "voxel" else (.01, .2)
    if args.frame_step < 1 or not low <= args.voxel_size <= high:
        p.error(f"Positive frame step and voxel size in [{low}, {high}] metres required")
    try:
        return run(args)
    except (ValueError, RuntimeError, OSError, ImportError, KeyError, subprocess.SubprocessError) as error:
        # Never put credentials, HTTP headers or signed model URLs into reports.
        print(f"Refined reconstruction stopped: {error}", file=__import__('sys').stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
