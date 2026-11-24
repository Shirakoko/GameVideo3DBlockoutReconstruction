"""Small real sample: 12s RGB, 720 poses, six raw Depth frames, honest checks."""
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage_a.depth import decode_v3_raw, relative_log_depth, validate_intrinsics, validate_near_plane_cm
from stage_a.media import ffmpeg_path, iter_frames, probe_local

ROOT = Path(__file__).resolve().parents[1]
SESSION = "66ad9535-583f-4a25-b655-6b1697c83a40"
SOURCE = ROOT / "data/input/samples" / SESSION
OUT = ROOT / "outputs/stage_a/sample_66ad"
COUNT = 720
KEYS = [0, 120, 240, 360, 600, 719]


def json_file(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prefix_records(path):
    # Explicitly discard the last partial line of a bounded HTTP prefix.
    return [json.loads(line) for line in path.read_bytes().split(b"\n")[:-1]]


def run_ffmpeg(args, timeout=60):
    result = subprocess.run([ffmpeg_path(), "-hide_banner", "-nostdin", *args], capture_output=True, timeout=timeout)
    if result.returncode:
        raise ValueError("FFmpeg sample operation failed: " + result.stderr.decode(errors="replace")[-2000:])
    return result


def hash_frames(path, depth=False):
    args = ["-loglevel", "error"]
    if depth:
        args += ["-r", "60", "-f", "hevc"]
    args += ["-i", str(path), "-map", "0:v:0", "-an", "-frames:v", str(COUNT), "-fps_mode", "passthrough"]
    if depth:
        args += ["-vf", "extractplanes=y", "-pix_fmt", "gray10le"]
    args += ["-f", "framemd5", "-"]
    result = run_ffmpeg(args)
    records = [line.split(",")[-1].strip() for line in result.stdout.decode().splitlines() if line and not line.startswith("#")]
    if len(records) != COUNT:
        raise ValueError(f"Decoded {len(records)} frames instead of {COUNT}")
    duplicates = [i for i in range(1, len(records)) if records[i] == records[i-1]]
    return records, duplicates


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in ("rgb", "depth", "camera", "metadata"):
        (OUT / folder).mkdir(exist_ok=True)
    session = json_file(SOURCE / "session.json")
    contract = json_file(SOURCE / "depth/decode_contract.json")
    world = json_file(SOURCE / "telemetry/world.json")
    conversion = json_file(SOURCE / "tables/conversion_manifest.json")
    if session["fps"] != 60 or contract["frame_accounting"]["alignment"]["prehud_latch_lag_ticks"] != 0:
        raise ValueError("Unsupported clock/latch contract")
    if conversion["rules"]["camera"].split(":", 1)[0] != "last_in_frame":
        raise ValueError("Unknown camera aggregation rule")
    meta = prefix_records(SOURCE / "depth/depth_meta.jsonl.prefix")
    meta_by_frame = {record["depth_frame"]: record for record in meta}
    cameras = prefix_records(SOURCE / "telemetry/camera.jsonl.prefix")
    grouped = defaultdict(list)
    for record in cameras:
        grouped[record["frame"]].append(record)
    if any(frame not in meta_by_frame or frame not in grouped for frame in range(COUNT)):
        raise ValueError("Clip has missing Depth Metadata or Camera frames")
    if any(first < COUNT for first, last in contract["frame_accounting"]["alignment"]["missing_tick_ranges"]):
        raise ValueError("Clip crosses original-to-delivery tick gap")
    intr = contract["intrinsics"]
    k = [[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]]
    validate_intrinsics(k, width=intr["width"], height=intr["height"])
    near = validate_near_plane_cm(contract["near_plane_cm"])
    for frame in range(COUNT):
        record = meta_by_frame[frame]
        if record["depth_transform"] != 3 or record["near_plane_cm"] != near or (record["width"], record["height"]) != (intr["width"], intr["height"]):
            raise ValueError("Per-frame Depth settings differ from reviewed contract")
    cols = ["frame_idx", "t_ms", "cam_valid", "cam_px", "cam_py", "cam_pz", "cam_pitch", "cam_yaw", "cam_roll"]
    table = pq.ParquetFile(SOURCE / "tables/frames.parquet")
    # Read only the first 1024-row batch and explicit columns, not all table data.
    first_batch = next(table.iter_batches(batch_size=1024, columns=cols)).to_pylist()
    by_frame = {row["frame_idx"]: row for row in first_batch}
    mismatches = []
    pose_rows = []
    epoch = contract["alignment"]["video0_qpc"]
    frequency = contract["alignment"]["qpc_frequency"]
    for frame in range(COUNT):
        selected = grouped[frame][-1]  # Verified provider rule, original file order.
        row = by_frame[frame]
        if not row["cam_valid"] or any(not np.isclose(selected[name], row["cam_"+name], atol=1e-4, rtol=0) for name in ("px", "py", "pz", "pitch", "yaw", "roll")):
            mismatches.append(frame)
        pose_rows.append({"clip_frame": frame, "rgb_frame_index": frame, "depth_frame_index": frame,
                          "original_tick": frame, "timestamp_seconds": frame/60,
                          "camera_rule": "last_in_frame:file_order", "camera_samples_count": len(grouped[frame]),
                          "camera_raw": selected, "camera_qpc_seconds_from_video0": (selected["qpc"]-epoch)/frequency,
                          "depth_referenced_camera_frame": meta_by_frame[frame]["camera_frame"],
                          "depth_referenced_camera_raw": grouped.get(meta_by_frame[frame]["camera_frame"], [None])[-1]})
    if mismatches:
        raise ValueError(f"Camera JSONL / frame table mismatch at {mismatches[:10]}")
    (OUT / "camera/poses.json").write_text(json.dumps({"coordinate_system": world["coordinate_system"], "world_to_meters": world["world_to_meters"], "rotation_representation": "source pitch/yaw/roll; no coordinate conversion", "frames": pose_rows}, indent=2), encoding="utf-8")
    source_rgb = SOURCE / "video/prehud_prefix.partial.mp4"
    source_depth = SOURCE / "depth/depth_prefix.partial.hevc"
    original = OUT / "original_video.mp4"
    run_ffmpeg(["-loglevel", "error", "-y", "-i", str(source_rgb), "-map", "0:v:0", "-frames:v", str(COUNT), "-c:v", "copy", "-an", "-movflags", "+faststart", str(original)])
    rgb_spec = probe_local(original)
    depth_spec = probe_local(source_depth, elementary_hevc=True)
    if (rgb_spec.width, rgb_spec.height) != (1920, 1080) or (depth_spec.width, depth_spec.height) != (1920, 1080):
        raise ValueError("Actual raster differs from selected Session")
    print("Validating 720 decoded RGB frames and timestamps", flush=True)
    source_hashes, rgb_duplicates = hash_frames(source_rgb)
    clip_hashes, _ = hash_frames(original)
    if clip_hashes != source_hashes:
        raise ValueError("Re-muxed clip pixels differ from source window")
    timestamp_result = run_ffmpeg(["-loglevel", "info", "-i", str(original), "-map", "0:v:0", "-frames:v", str(COUNT), "-an", "-vf", "showinfo", "-f", "null", "-"])
    timestamp_log = timestamp_result.stderr.decode(errors="replace")
    (OUT / "metadata/rgb_timestamp_decode.log").write_text(timestamp_log, encoding="utf-8")
    pts = [int(value) for value in re.findall(r"\bn:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time", timestamp_log)]
    time_base = re.search(r"config in time_base:\s*(\d+)/(\d+)", timestamp_log)
    if len(pts) != COUNT or not time_base:
        raise ValueError("Cannot verify all RGB PTS")
    times = np.array(pts)*int(time_base[1])/int(time_base[2])
    expected_times = np.arange(COUNT)/60
    depth_times = np.array([meta_by_frame[frame]["timestamp"] for frame in range(COUNT)])
    timestamp_error = float(np.max(np.abs(times-depth_times)))
    if not np.allclose(times, expected_times, atol=1e-9, rtol=0) or timestamp_error > 1e-9:
        raise ValueError("RGB / Depth Metadata timestamps do not match the frame grid")
    print("Validating 720 decoded native Depth frames", flush=True)
    depth_hashes, depth_duplicates = hash_frames(source_depth, depth=True)
    (OUT / "metadata/decode_hashes.json").write_text(json.dumps({"rgb_hashes": clip_hashes, "depth_luma_hashes": depth_hashes}), encoding="utf-8")
    depth_stats = []
    rgb_images = {}
    depth_images = {}
    for frame in KEYS:
        _, rgb = next(iter_frames(original, start=frame, count=1))
        rgb_image = Image.fromarray(rgb)
        rgb_image.save(OUT / f"rgb/frame_{frame:06d}.png")
        rgb_images[frame] = rgb_image
        _, raw = next(iter_frames(source_depth, start=frame, count=1, depth=True))
        codes, mask = decode_v3_raw(raw, contract)
        np.savez_compressed(OUT / f"depth/frame_{frame:06d}.npz", codes10=codes, valid_code_mask=mask,
                            frame_index=np.int64(frame), timestamp_seconds=np.float64(frame/60))
        depth_stats.append({"frame_index": frame, "code_min": int(codes.min()), "code_max": int(codes.max()),
                            "zero_code_fraction": float((codes == 0).mean()), "sentinel_fraction": float((codes == 1023).mean()),
                            "reserved_or_sentinel_fraction": float((~mask).mean()),
                            "valid_code_percentiles": np.percentile(codes[mask], [1, 50, 99]).tolist() if mask.any() else []})
        vis = np.repeat((codes.astype(np.float32)/1023*255).astype(np.uint8)[..., None], 3, axis=2)
        vis[~mask] = [255, 0, 255]
        depth_images[frame] = Image.fromarray(vis)
    sheet = Image.new("RGB", (960, 860), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((15, 10), "REAL SAMPLE / RGB + RAW 10-bit DEPTH CODES (NOT METRIC)", fill="black")
    for index, frame in enumerate(KEYS):
        x, y = (index%3)*320, 40+(index//3)*410
        sheet.paste(rgb_images[frame].resize((320, 180)), (x, y))
        sheet.paste(depth_images[frame].resize((320, 180)), (x, y+185))
        draw.text((x+8, y+370), f"Frame {frame} / {frame/60:.3f}s; magenta=reserved", fill="black")
    sheet.save(OUT / "contact_sheet.jpg", quality=95)
    file_mismatches = []
    for name in ("depth/decode_contract.json", "tables/conversion_manifest.json", "telemetry/world.json", "tables/frames.parquet"):
        raw = (SOURCE / name).read_bytes()
        claimed = session["files"][name]
        digest = hashlib.sha256(raw).hexdigest()
        if len(raw) != claimed["size_bytes"] or digest != claimed["sha256"]:
            file_mismatches.append({"path": name, "actual_bytes": len(raw), "claimed_bytes": claimed["size_bytes"],
                                    "actual_sha256": digest, "claimed_sha256": claimed["sha256"]})
    far = contract["transform_constants"]["far_plane_encode"]
    # Demonstrate why the old README formula is NOT a metric v3 decoder.
    old_at_half = float(relative_log_depth(np.array([0.5]), log_constant=4000)[0])*near/100
    report = {"status": "small_raw_sample_usable_metric_depth_unverified", "session_id": SESSION,
              "duration_seconds": 12, "fps": 60, "decoded_rgb_frames": len(clip_hashes), "decoded_depth_frames": len(depth_hashes),
              "exported_keyframes": KEYS, "exported_camera_poses": COUNT,
              "native_rgb_raster": [rgb_spec.width, rgb_spec.height], "native_depth_raster": [depth_spec.width, depth_spec.height],
              "rgb_clip_lossless_remux_verified": True, "audio_included": False,
              "timestamp_max_rgb_depth_meta_error_seconds": timestamp_error,
              "camera_table_pose_mismatch_frames": mismatches, "camera_aggregation_rule": conversion["rules"]["camera"],
              "camera_sample_count_histogram": dict(Counter(len(grouped[frame]) for frame in range(COUNT))),
              "depth_camera_reference_lag_frames_histogram": dict(Counter(frame-meta_by_frame[frame]["camera_frame"] for frame in range(COUNT))),
              "note_depth_camera_reference": "Depth Metadata.camera_frame is preserved separately; not substituted for RGB frame index. Snapshot-latch semantics are not yet fully verified.",
              "depth_transform": contract["depth_transform"], "transform_constants": contract["transform_constants"],
              "native_code_range": [0, 1023], "vui_full_range": contract["vui_full_range"], "near_plane_cm": near,
              "intrinsics": intr, "fov_fx_check_abs_error": abs(intr["width"]/(2*np.tan(np.deg2rad(contract["fov_horizontal_deg"])/2))-intr["fx"]),
              "world": world, "coordinate_conversion_applied": False,
              "depth_keyframe_stats": depth_stats,
              "rgb_exact_adjacent_duplicate_count": len(rgb_duplicates), "depth_exact_adjacent_duplicate_count": len(depth_duplicates),
              "duplicates_definition": "decoded RGB native YUV / decoded Depth native 10-bit Y equality; source dup_count is a separate session-level statistic",
              "session_declared_depth_dup_count": contract["frame_accounting"]["depth"]["dup_count"],
              "sample_first_frame_predecessor_checked": False,
              "metric_depth_validated": False,
              "metric_formula_blocker": "Actual transform=3 log_sentinel_v3. Official loader covers transform=2; contract constants do not define inverse equation, code normalization or far_plane_encode units. Do not use old formula as metric.",
              "legacy_formula_diagnostic": {"q": 0.5, "legacy_depth_m": old_at_half, "near_clip_m": near/100,
                                            "issue": "relative_depth*near_plane produces depths BELOW near clip for q<1; not a valid planar-z metric inverse"},
              "source_manifest_hash_or_size_mismatches": file_mismatches,
              "session_declared_global_count_conflict": {"alignment_frames": session["alignment"]["rgb"]["frames"], "signals_depth_frames": contract["signals"]["depth"]["frame_count"], "frame_table_rows": table.metadata.num_rows, "not_full_session_decoded": True},
              "frame_index_scope": "delivery frames 0..719 equal original ticks before first hole 97344; no assumption after missing-tick ranges",
              "stage_a_complete": False, "blender_used": False,
              "outputs": {"rgb_video": "original_video.mp4", "rgb_frames": "rgb/*.png", "depth_raw_frames": "depth/*.npz", "camera": "camera/poses.json", "preview": "contact_sheet.jpg"}}
    (OUT / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    text = f"""# 小规模真实样本报告

已输出 **12 秒 / 60 FPS / 720 帧** RGB 视频、720 帧相机记录、6 组 RGB/原生 Depth 关键帧。未下载完整 Session，未进入 Blender。

Session：`{SESSION}`，UE5 / lh_z_up，world_to_meters=100。选取 0–12 秒，不跨越该 Session 已知丢帧区间。

## 已通过

- RGB 与 Depth 均实际解码 720 帧；分辨率均为 1920×1080。
- 输出视频为原 RGB 码流无损 remux，720 个解码帧 hash 与源窗口一致。音频未导出。
- RGB PTS 与 Depth Metadata 时间戳最大误差：{timestamp_error:.3g} 秒。
- 相机采用官方 `last_in_frame` 文件顺序规则；720 帧全部有记录，且与 frames.parquet 六个 pose 字段交叉一致。
- near_plane_cm={near}，K=(fx={intr['fx']}, fy={intr['fy']}, cx={intr['cx']}, cy={intr['cy']})；与原生分辨率和水平 FOV 一致。
- 相邻精确重复：RGB {len(rgb_duplicates)} 帧，Depth luma {len(depth_duplicates)} 帧。源 Session dup_count={contract['frame_accounting']['depth']['dup_count']}，不可用整 Session 计数替代片段计数。

## 必须保留的限制

1. **Depth 尚不是已验证的米制深度。** 实际为 `log_sentinel_v3` / transform=3，K=4000、far_plane_encode={far}、sentinel=1023、max_valid=1020。官方 README/loader 的旧 transform=2 示例不能直接使用；`relative_depth × near_plane` 会落到 near clip 以内。缺少 v3 逆公式、码值分母及 far_plane_encode 单位的明确规格。
2. NPZ 中 `codes10` 是 uint16 原生 0–1023 luma；`valid_code_mask` 仅排除 1021–1023 保留/哨兵码，不代表已确认每个有效码的物理意义。没有伪造 metric_depth。
3. Depth Metadata 的 `camera_frame` 经常滞后于 `depth_frame`；两个字段和对应相机记录均保留，不把它作为 RGB Frame Index。这一 snapshot latch 的物理时刻仍需确认。
4. Session 级 manifest 的部分文件 size/hash 与实际固定版本文件不一致：{len(file_mismatches)} 项，详见 JSON。全局 RGB、Depth/帧表数量也相差 1；只确认本窗口，未声称整 Session 完整。
5. 相机旋转保持原始 pitch/yaw/roll，未生成外参矩阵或转换到 Blender。

## 产物

- `original_video.mp4`：真实 12 秒 RGB。
- `rgb/`：6 张原分辨率无损 PNG。
- `depth/`：6 个原生码值/掩码 NPZ。
- `camera/poses.json`：720 帧、原始 QPC、关联 Frame Index、Depth 相机引用。
- `contact_sheet.jpg`：RGB + raw-code 调试图（不是米制热力图）。
- `validation_report.json`：逐项统计与异常证据。

受限数据仅用于本地内部评估，不分发、不上传第三方。最早下载的删除期限以 `../metadata_download_receipt.json` 为准（2026-12-24，Asia/Shanghai）。
"""
    (OUT / "validation_report.md").write_text(text, encoding="utf-8")
    print(json.dumps({"output": str(OUT), "duration_seconds": 12, "rgb_frames": COUNT, "depth_decoded_frames": COUNT,
                      "keyframes": KEYS, "camera_poses": COUNT, "metric_depth_validated": False,
                      "source_manifest_mismatches": len(file_mismatches)}, indent=2))


if __name__ == "__main__":
    main()
