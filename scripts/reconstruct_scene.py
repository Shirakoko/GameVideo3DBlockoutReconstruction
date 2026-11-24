"""Stage B: audit existing raw sample, or reconstruct explicitly verified input.

Default is the real audit (exit 2: geometry blocked), never synthetic fallback.
--synthetic is a separately labelled analytical demonstration, not Stage A
acceptance. --input-manifest accepts reviewed metric frames, NOT HEVC codes.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reconstruction.backprojection import DepthSpec, backproject
from reconstruction.coordinate_transform import transform_points
from reconstruction.pointcloud_fusion import FrameInput, fuse_frames, point_cloud, write_verified_ply
from reconstruction.synthetic import make_synthetic_frames
from reconstruction.validation import audit_sample, json_read
from reconstruction.visualization import cloud_plot, depth_plot, trajectory_plot

SESSION = "66ad9535-583f-4a25-b655-6b1697c83a40"


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def run_audit(output):
    report = audit_sample(ROOT/"data/input/samples"/SESSION, ROOT/"outputs/stage_a/sample_66ad")
    candidates = report.pop("candidate_poses")
    output.mkdir(parents=True, exist_ok=True)
    positions = np.asarray([row["world_from_optical_candidate_m"] for row in candidates])[:, :3, 3]
    colors = np.tile(np.array([[210, 55, 65]], dtype=np.uint8), (len(positions), 1))
    report["telemetry_position_ply"] = write_verified_ply(output/"camera_trajectory_only.ply", point_cloud(positions, colors),
        data_source="REAL TELEMETRY POSITIONS ONLY - NO DEPTH GEOMETRY - UE WORLD SCALE FROM METADATA")
    json_write(output/"camera_pose_candidates.json", {"status": "hypothesis_only_render_pose_unverified", "frames": candidates})
    trajectory_plot(output/"camera_trajectory.png", candidates)
    report["checked_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    json_write(output/"validation_report.json", report)
    lines = ["# Stage B — 真实样本前置验证", "", "状态：真实几何重建阻塞；没有输出真实场景单帧/融合点云。", "",
             f"重新检查 {report['frames_rechecked']} 帧相机、Parquet、Depth Metadata 和 6 组 raw code NPZ；没有新增样本下载。", "",
             "## 已确认", "", "- Session / decode_contract / world.json 均为 UE5、lh_z_up、world_to_meters=100。",
             "- 原始相机按 last_in_frame 与 Parquet 一致；世界尺度和深度单位是不同合同。",
             "- UE Rotator 及 optical→Blender 数学经过单元测试和独立 Blender 投影检查；这不证明 SDK 字段和 render pose 时刻。", "",
             "## 阻塞", ""]
    lines += [f"- `{item['id']}`：{item['reason']}" for item in report["blockers"]]
    lines += ["", "## 产物边界", "", "camera_trajectory_only.ply 仅含相机位置，不是场景点云。camera_pose_candidates.json / 方向箭头为 UE FRotator 解释下的候选，不是已验证渲染外参。",
              "", "另一套 OriginLab free-camera-v2 文档提供了 v3 逆公式线索，但未迁移到当前 Session，不假设深度单位、码值归一化或近裁面比例。",
              "", "完整 Stage B 验收需提供适用于此 Session 的深度解码/单位/饱和规格、像素中心规则和明确的 depth render-pose 对应关系；届时可导入已审查的 metric_frames/1 manifest。",
              "", "受限样本及派生遥测仅限本地内部评估；保留 Stage A 原删除期限 2026-12-24（Asia/Shanghai），不得再分发。"]
    (output/"validation_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str((output/"validation_report.md").resolve()), "blockers": [row["id"] for row in report["blockers"]]}, indent=2))
    return 2


def load_metric_manifest(path):
    manifest = json_read(path)
    if manifest.get("schema") != "stage_b.metric_frames/1" or manifest.get("coordinate_system") != "blender_rh_z_up" or manifest.get("pose_direction") != "world_from_optical_camera" or manifest.get("pose_translation_unit") != "m":
        raise ValueError("Explicit stage_b.metric_frames/1 Blender-world, optical camera-to-world, metre pose contract required")
    spec = DepthSpec(**manifest["depth_spec"])
    spec.validate()
    if not manifest.get("frames") or len(manifest["frames"]) > 720:
        raise ValueError("Manifest must contain 1..720 reviewed frame records")
    frames = []
    decoded_bytes = 0
    for row in manifest["frames"]:
        source = (path.parent/row["data_npz"]).resolve()
        # Bound decompression before NumPy materializes arrays, including zip bombs.
        with ZipFile(source) as archive:
            decoded_bytes += sum(info.file_size for info in archive.infolist())
        if decoded_bytes > 512*1024**2:
            raise ValueError("Reviewed NPZ input exceeds 512 MiB decoded-byte budget; select fewer frames")
        with np.load(source, allow_pickle=False) as data:
            frame = FrameInput(row["frame_index"], row["rgb_frame_index"], row["depth_frame_index"],
                row["original_tick"], row["timestamp_seconds"], data["depth"].copy(), data["valid_mask"].copy(),
                data["rgb"].copy(), data["K"].copy(), np.asarray(row["world_from_camera"], dtype=np.float64), spec,
                row["geometry_pose_verified"], row["sync_evidence"], row["source_observation_id"])
        frame.validate()
        frames.append(frame)
    return frames, {"data_source": manifest["data_source"], "input_manifest": str(path.resolve()),
                    "input_schema": manifest["schema"], "evidence_boundary": "caller-supplied reviewed depth/pose contract; not inferred by this loader"}


def reconstruct(frames, source_metadata, output, *, min_depth_m, max_depth_m, voxel_size_m, stride, truth=None):
    result = fuse_frames(frames, voxel_size_m=voxel_size_m, min_depth_m=min_depth_m,
                         max_depth_m=max_depth_m, stride=stride)
    output.mkdir(parents=True, exist_ok=True)
    receipts = [write_verified_ply(output/"single_frame_pointcloud.ply", result.first_frame_cloud, data_source=source_metadata["data_source"]),
                write_verified_ply(output/"scene_pointcloud.ply", result.cloud, data_source=source_metadata["data_source"])]
    np.savez_compressed(output/"point_provenance.npz", offsets=result.provenance_offsets,
                        frame_indices=result.provenance_frame_indices, timestamp_seconds=result.provenance_timestamps)
    first = result.first_backprojection
    np.savez_compressed(output/"single_frame_points.npz", points_camera_m=first.points_camera_m,
        points_world_m=np.asarray(result.first_frame_cloud.points), pixels_uv=first.pixels_uv, colors_rgb8=first.colors_rgb8,
        frame_index=np.int64(next(row["frame_index"] for row in result.frame_reports if row["status"] == "accepted")))
    accepted = [frame for frame, report in zip(frames, result.frame_reports) if report["status"] == "accepted"]
    cloud_plot(output/"pointcloud_views.png", result.first_frame_cloud, result.cloud,
               cameras=[frame.world_from_camera for frame in accepted], label=source_metadata["data_source"])
    if truth is not None:
        errors = []
        for frame in accepted:
            bp = backproject(frame.depth, frame.intrinsics, spec=frame.depth_spec, valid_mask=frame.valid_mask,
                             rgb=frame.rgb, min_depth_m=min_depth_m, max_depth_m=max_depth_m, stride=stride)
            expected = truth[frame.frame_index][bp.pixels_uv[:, 1], bp.pixels_uv[:, 0]]
            errors.extend(np.linalg.norm(transform_points(bp.points_camera_m, frame.world_from_camera)-expected, axis=1).tolist())
        source_metadata["analytic_backprojection_max_error_m"] = max(errors)
        source_metadata["analytic_backprojection_rms_error_m"] = float(np.sqrt(np.mean(np.square(errors))))
        if max(errors) > 1e-10:
            raise ValueError("Synthetic independent ray-caster geometry did not agree with back-projection")
        def surface_distance(points, box):
            low, high = np.asarray(box)
            outside = np.linalg.norm(np.maximum(np.maximum(low-points, points-high), 0), axis=1)
            inside = ((points >= low) & (points <= high)).all(axis=1)
            inner = np.min(np.minimum(points-low, high-points), axis=1)
            return np.where(inside, inner, outside)
        points = np.asarray(result.cloud.points)
        distance = np.minimum(surface_distance(points, source_metadata["room_aabb_blender_m"]),
                              surface_distance(points, source_metadata["block_aabb_blender_m"]))
        source_metadata["fused_analytic_surface_distance_m"] = {"max": float(distance.max()),
            "rms": float(np.sqrt(np.mean(distance**2))), "p95": float(np.percentile(distance, 95)),
            "bound": float(np.sqrt(3)*voxel_size_m), "scope": "distance to known synthetic room/box surfaces, not real sample geometry"}
        if distance.max() > np.sqrt(3)*voxel_size_m+1e-10:
            raise ValueError("Fused cloud escaped the known synthetic surfaces beyond one voxel diagonal")
        depth_plot(output/"backprojection_debug.png", accepted[0], first)
    metadata = {"schema": "stage_b.reconstruction/1", "status": "synthetic_demo_passed" if truth is not None else "reviewed_input_reconstructed",
                "real_sample_reconstruction_validated": False, "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "source": source_metadata, "coordinate_system": "blender_rh_z_up", "output_unit": "m",
                "pose_direction": "world_from_optical_camera", "matrix_storage": "row-major JSON, column-vector math",
                "first_frame_filtering": "finite/mask/metric range + pixel stride; SOR/voxel applied to fused cloud only",
                "depth_limits_m": [min_depth_m, max_depth_m], "fusion": result.stats,
                "frames": result.frame_reports, "ply_receipts": receipts,
                "provenance_file": "point_provenance.npz", "provenance_layout": "offsets[i]:offsets[i+1] slice gives accepted source frames/times for PLY vertex i; duplicates tracked as aliases in frames",
                "limitations": ["Static-scene accumulation only; dynamic objects may ghost.", "No ICP, meshing, TSDF, semantics, Blender scene generation or video rendering.", "Synthetic success is not a certificate for the raw Hugging Face sample."]}
    json_write(output/"reconstruction_metadata.json", metadata)
    print(json.dumps({"status": metadata["status"], "source": source_metadata["data_source"], "fusion": result.stats,
                      "output": str(output.resolve())}, ensure_ascii=False, indent=2))
    return 0


def save_synthetic_inputs(output, frames, metadata):
    folder = output/"inputs"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in frames:
        name = f"frame_{frame.frame_index:06d}.npz"
        np.savez_compressed(folder/name, depth=frame.depth, valid_mask=frame.valid_mask, rgb=frame.rgb, K=frame.intrinsics)
        row = {key: getattr(frame, key) for key in ("frame_index", "rgb_frame_index", "depth_frame_index", "original_tick", "timestamp_seconds",
            "geometry_pose_verified", "sync_evidence", "source_observation_id")}
        row.update(data_npz=f"inputs/{name}", world_from_camera=frame.world_from_camera.tolist())
        rows.append(row)
    json_write(output/"synthetic_input_manifest.json", {"schema": "stage_b.metric_frames/1", "data_source": metadata["data_source"],
        "coordinate_system": "blender_rh_z_up", "pose_direction": "world_from_optical_camera", "pose_translation_unit": "m",
        "depth_spec": asdict(frames[0].depth_spec), "frames": rows})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--synthetic", action="store_true")
    mode.add_argument("--input-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-depth-m", type=float)
    parser.add_argument("--max-depth-m", type=float)
    parser.add_argument("--voxel-size-m", type=float, default=0.08)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        if args.synthetic:
            output = args.output or ROOT/"outputs/stage_b/synthetic_demo"
            frames, truth, metadata = make_synthetic_frames()
            save_synthetic_inputs(output, frames, metadata)
            return reconstruct(frames, metadata, output, min_depth_m=0.1 if args.min_depth_m is None else args.min_depth_m,
                max_depth_m=40. if args.max_depth_m is None else args.max_depth_m,
                voxel_size_m=args.voxel_size_m, stride=args.stride, truth=truth)
        if args.input_manifest:
            if args.min_depth_m is None or args.max_depth_m is None:
                raise ValueError("Reviewed input requires explicit --min-depth-m and --max-depth-m, not assumed clip planes")
            frames, metadata = load_metric_manifest(args.input_manifest.resolve())
            return reconstruct(frames, metadata, args.output or ROOT/"outputs/stage_b/reviewed_input",
                min_depth_m=args.min_depth_m, max_depth_m=args.max_depth_m, voxel_size_m=args.voxel_size_m, stride=args.stride)
        return run_audit(args.output or ROOT/"outputs/stage_b/sample_66ad")
    except (ValueError, KeyError, OSError) as error:
        print(f"Stage B failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
