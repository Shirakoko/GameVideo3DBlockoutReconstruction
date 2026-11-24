"""Stage C/D: local assisted real RGB, verified metric input, or explicit demo."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.object_detection import AnnotationProvider, LocalVisionProvider, SyntheticColorProvider
from agents.scene_builder import build_scene, json_write
from agents.synthetic_scene import check_fixture, make_scene_fixture
from scripts.reconstruct_scene import load_metric_manifest, save_synthetic_inputs


def rgb_records(metric_frames):
    return [dict(frame_index=f.frame_index, timestamp_seconds=f.timestamp_seconds, original_tick=f.original_tick,
        source_observation_id=f.source_observation_id, rgb=f.rgb) for f in metric_frames]


def load_rgb_manifest(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "stage_c.rgb_frames/1":
        raise ValueError("Explicit stage_c.rgb_frames/1 manifest required")
    rows = []
    if not 1 <= len(document["frames"]) <= 720:
        raise ValueError("RGB manifest requires 1..720 keyframes")
    budget = 0
    for item in document["frames"]:
        with Image.open(path.parent/item["rgb_path"]) as image:
            budget += image.width*image.height*3
            if budget > 512*1024**2:
                raise ValueError("RGB decoded byte budget exceeds 512 MiB")
            rgb = np.asarray(image.convert("RGB")).copy()
        rows.append(dict(frame_index=item["frame_index"], timestamp_seconds=item["timestamp_seconds"], original_tick=item["original_tick"],
            source_observation_id=item["source_observation_id"], rgb=rgb))
    return rows, document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--synthetic", action="store_true")
    mode.add_argument("--input-manifest", type=Path, help="reviewed Stage B metric_frames/1")
    mode.add_argument("--rgb-manifest", type=Path, help="explicit RGB-only partial output")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--local-detector", type=Path)
    parser.add_argument("--local-sam", type=Path)
    parser.add_argument("--scene-id")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--min-depth-m", type=float)
    parser.add_argument("--max-depth-m", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.annotations and args.local_detector:
            raise ValueError("Choose annotations OR local detector")
        if args.local_sam and not args.local_detector:
            raise ValueError("--local-sam requires --local-detector")
        metric_frames, clouds = None, None
        raw_source = None
        input_path = args.input_manifest or args.rgb_manifest
        if args.synthetic:
            if args.annotations or args.local_detector:
                raise ValueError("Synthetic demo uses its explicit fixture-only RGB provider")
            metric_frames, truth = make_scene_fixture()
            rows = rgb_records(metric_frames)
            provider = SyntheticColorProvider()
            source, scene_id, fps = "SYNTHETIC FIVE-CATEGORY FIXTURE ONLY — NOT HUGGING FACE", args.scene_id or "synthetic-five-category", 60. if args.fps is None else args.fps
            min_depth = 0.1 if args.min_depth_m is None else args.min_depth_m
            max_depth = 35. if args.max_depth_m is None else args.max_depth_m
            output = args.output or ROOT/"outputs/stage_cd/synthetic_demo"
            output.mkdir(parents=True, exist_ok=True)
            save_synthetic_inputs(output, metric_frames, dict(data_source=source))
            input_path = output/"synthetic_input_manifest.json"
        elif args.input_manifest or args.rgb_manifest:
            if args.scene_id is None or args.fps is None:
                raise ValueError("External input requires explicit --scene-id and --fps")
            if not args.annotations and not args.local_detector:
                raise ValueError("External RGB needs --annotations or explicit --local-detector weights")
            if args.input_manifest:
                if args.min_depth_m is None or args.max_depth_m is None:
                    raise ValueError("Reviewed input requires explicit --min-depth-m and --max-depth-m")
                metric_frames, metadata = load_metric_manifest(args.input_manifest.resolve())
                rows = rgb_records(metric_frames)
                source = metadata["data_source"]
                manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
                clouds = {r["frame_index"]: args.input_manifest.resolve().parent/r["point_cloud_npz"] for r in manifest["frames"] if r.get("point_cloud_npz")}
            else:
                rows, metadata = load_rgb_manifest(args.rgb_manifest.resolve())
                source = metadata["data_source"]
            provider = AnnotationProvider(args.annotations) if args.annotations else LocalVisionProvider(str(args.local_detector), str(args.local_sam) if args.local_sam else None)
            scene_id, fps = args.scene_id, args.fps
            min_depth, max_depth = args.min_depth_m, args.max_depth_m
            output = args.output or ROOT/"outputs/stage_cd/reviewed_input"
        else:
            sample = ROOT/"outputs/stage_a/sample_66ad"
            provider = LocalVisionProvider(str(args.local_detector), str(args.local_sam) if args.local_sam else None) if args.local_detector else AnnotationProvider(args.annotations or ROOT/"config/sample_66ad_annotations.json")
            poses = json.loads((sample/"camera/poses.json").read_text(encoding="utf-8"))["frames"]
            by_index = {p["rgb_frame_index"]: p for p in poses}
            rows = []
            for index in sorted(provider.rows) if isinstance(provider, AnnotationProvider) else [0, 120, 240, 360, 600, 719]:
                p = by_index[index]
                with Image.open(sample/f"rgb/frame_{index:06d}.png") as image:
                    rgb = np.asarray(image.convert("RGB")).copy()
                rows.append(dict(frame_index=index, timestamp_seconds=p["timestamp_seconds"], original_tick=p["original_tick"],
                    source_observation_id=f"sample-rgb-delivery-{index}", rgb=rgb))
            raw_source = str((sample/"camera/poses.json").resolve())
            scene_id, fps, source = args.scene_id or "sample-66ad-rgb-partial", 60. if args.fps is None else args.fps, "REAL RESTRICTED SAMPLE 66ad — assisted RGB only, geometry unverified; internal evaluation; delete by 2026-12-24 Asia/Shanghai"
            output = args.output or ROOT/"outputs/stage_cd/sample_66ad"
            min_depth = max_depth = None
        scene, report = build_scene(rows, provider, output, scene_id=scene_id, data_source=source, fps=fps, metric_frames=metric_frames,
            min_depth_m=min_depth, max_depth_m=max_depth, pixel_cloud_paths=clouds, raw_camera_source=raw_source,
            input_manifest=str(input_path.resolve()) if input_path else None)
        if args.synthetic:
            report["fixture_checks"] = check_fixture(scene, metric_frames, truth)
            json_write(output/"validation_report.json", report)
            annotation_rows = []
            for frame in metric_frames:
                items = []
                for obj in scene["objects"]:
                    obs = next(r for r in obj["observations"] if r["frame_index"] == frame.frame_index)
                    items.append(dict(key=obj["category"], category=obj["category"], score=1.,
                        mask=dict(kind="file", path=obs["mask"]["path"], score=1.)))
                annotation_rows.append(dict(frame_index=frame.frame_index, raster_size=[frame.rgb.shape[1], frame.rgb.shape[0]], objects=items))
            json_write(output/"synthetic_annotations.json", dict(schema="stage_c.rgb_annotations/1",
                evidence="SYNTHETIC palette-derived masks ONLY; inferred labels; saved for generic reviewed-input replay, not human/real ground truth", frames=annotation_rows))
        print(json.dumps(dict(report=report, scene_path=str((output/"scene_schema.json").resolve())), ensure_ascii=False, indent=2))
        return 0 if report["status"] == "complete" else 2
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(f"Stage C/D failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
