"""Read-only sample audit; distinguish metadata/math checks from physical proof."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from reconstruction.coordinate_transform import (blender_camera_from_optical_pose,
    inverse_rigid, world_from_optical_ue_pose)


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def complete_prefix(path):
    # Stage A's bounded downloads can end mid-record. Do not parse that tail.
    return [json.loads(line) for line in Path(path).read_bytes().split(b"\n")[:-1] if line.strip()]


def audit_sample(source: Path, prepared: Path) -> dict:
    session = json_read(source/"session.json")
    contract = json_read(source/"depth/decode_contract.json")
    world = json_read(source/"telemetry/world.json")
    poses = json_read(prepared/"camera/poses.json")
    stage_a = json_read(prepared/"validation_report.json")
    manifest = json_read(source/"tables/conversion_manifest.json")
    consistency = all(all(record.get(key) == world.get(key) for key in ("engine", "coordinate_system", "world_to_meters"))
                      for record in (session["world"], contract["world"]))
    if not consistency or world["engine"] != "ue5" or world["coordinate_system"] != "lh_z_up" or world["world_to_meters"] <= 0:
        raise ValueError("Conflicting/unsupported Session world contract; refuse coordinate conversion")
    if poses["coordinate_system"] != world["coordinate_system"] or poses["world_to_meters"] != world["world_to_meters"]:
        raise ValueError("Prepared pose world contract differs from source")
    if not manifest["rules"]["camera"].startswith("last_in_frame:"):
        raise ValueError("Camera aggregation rule not reviewed")
    groups = defaultdict(list)
    for record in complete_prefix(source/"telemetry/camera.jsonl.prefix"):
        groups[record["frame"]].append(record)
    rows = poses["frames"]
    depth_meta = {row["depth_frame"]: row for row in complete_prefix(source/"depth/depth_meta.jsonl.prefix")}
    columns = ["frame_idx", "cam_valid", "cam_px", "cam_py", "cam_pz", "cam_pitch", "cam_yaw", "cam_roll"]
    table_rows = next(pq.ParquetFile(source/"tables/frames.parquet").iter_batches(batch_size=1024, columns=columns)).to_pylist()
    table = {row["frame_idx"]: row for row in table_rows}
    if len(rows) != 720 or len({row["rgb_frame_index"] for row in rows}) != 720:
        raise ValueError("Expected already validated 720-frame window; other windows need their own adapter")
    candidates, mismatches = [], []
    for row in rows:
        index = row["rgb_frame_index"]
        raw = row["camera_raw"]
        if raw != groups[index][-1] or not table[index]["cam_valid"] or any(
                not np.isclose(raw[name], table[index]["cam_"+name], atol=1e-4, rtol=0)
                for name in ("px", "py", "pz", "pitch", "yaw", "roll")):
            mismatches.append(index)
        if row["depth_frame_index"] != index or row["original_tick"] != index or raw["frame"] != index:
            raise ValueError("Frame mapping differs from reviewed pre-hole window")
        if not np.isclose(row["timestamp_seconds"], depth_meta[index]["timestamp"], atol=1e-9, rtol=0):
            raise ValueError("Prepared timestamps differ from depth metadata")
        if row["depth_referenced_camera_frame"] != depth_meta[index]["camera_frame"]:
            raise ValueError("Prepared depth camera reference changed")
        t = world_from_optical_ue_pose([raw["px"], raw["py"], raw["pz"]],
            pitch_deg=raw["pitch"], yaw_deg=raw["yaw"], roll_deg=raw["roll"], world_units_per_meter=world["world_to_meters"])
        candidates.append({"frame_index": index, "original_tick": row["original_tick"],
                           "timestamp_seconds": row["timestamp_seconds"], "qpc_raw": raw["qpc"],
                           "depth_referenced_camera_frame": row["depth_referenced_camera_frame"],
                           "world_from_optical_candidate_m": t.tolist(),
                           "optical_from_world_candidate_m": inverse_rigid(t).tolist(),
                           "blender_camera_matrix_world_candidate_m": blender_camera_from_optical_pose(t).tolist()})
    if mismatches:
        raise ValueError(f"Source/Stage A/frame table poses disagree: {mismatches[:10]}")
    lag_counts = Counter(row["depth_frame_index"]-row["depth_referenced_camera_frame"] for row in rows)
    keyframes = []
    maximum = contract["transform_constants"]["max_valid_code10"]
    for path in sorted((prepared/"depth").glob("frame_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            codes, mask = data["codes10"], data["valid_code_mask"]
            if codes.dtype != np.uint16 or codes.shape != (contract["intrinsics"]["height"], contract["intrinsics"]["width"]) or codes.max() > 1023:
                raise ValueError("Raw depth NPZ does not match native code contract")
            if not np.array_equal(mask, codes <= maximum):
                raise ValueError("Stage A reserved-code mask changed")
            keyframes.append({"frame_index": int(data["frame_index"]),
                              "timestamp_seconds": float(data["timestamp_seconds"]),
                              "raw_code_min": int(codes.min()), "raw_code_max": int(codes.max()),
                              "code_1020_pixels": int((codes == maximum).sum()),
                              "zero_code_pixels": int((codes == 0).sum()), "metric_depth_present": "depth_m" in data})
    if len(keyframes) != 6:
        raise ValueError("Expected six already prepared keyframes")
    source_hashes = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (source/"session.json", source/"telemetry/world.json", source/"depth/decode_contract.json", source/"tables/conversion_manifest.json")}
    blockers = [
        {"id": "depth_metric_contract", "reason": "Current log_sentinel_v3 contract names constants but supplies no inverse/code-bin rule or explicit physical depth unit. World scale is NOT depth scale."},
        {"id": "render_pose_binding", "reason": "Depth camera_frame lags the delivery frame by 0..13 ticks; last_in_frame agrees with Parquet but does not prove the pose used for rendering depth."},
        {"id": "rotation_source_binding", "reason": "UE5 metadata and pitch/yaw/roll fields support an FRotator adapter, but this Session does not explicitly declare Euler units/signs or camera-local axes. Engine math is tested; SDK field binding remains unverified."},
        {"id": "pixel_sampling_and_saturation", "reason": "Centered K is declared, but pixel-centre offset and saturation/zero-code semantics are not explicit in this Session contract. Another OriginLab release masks code >= max_valid, unlike Stage A's <= max_valid raw-code mask."}]
    return {"status": "real_reconstruction_blocked", "session_id": session["session_uuid"],
            "scope": "local existing sample only; no additional dataset bytes downloaded", "stage_a_complete": stage_a["stage_a_complete"],
            "world_metadata_consistent": consistency, "world": world, "intrinsics": contract["intrinsics"],
            "frames_rechecked": len(rows), "pose_parquet_mismatch_frames": mismatches,
            "depth_camera_reference_lag_histogram": dict(sorted(lag_counts.items())),
            "depth_transform": contract["depth_transform"], "transform_constants": contract["transform_constants"],
            "metric_depth_validated": False, "render_pose_validated": False,
            "engine_rotation_math": "UE FRotator degrees, intrinsic yaw/pitch/roll; mathematical convention independently tested, not a capture-binding certificate",
            "coordinate_choice": "Blender world (X,-Y,Z); optical camera right/down/forward; column-vector camera-to-world matrix",
            "blockers": blockers, "keyframes": keyframes, "source_sha256": source_hashes,
            "cross_dataset_lead": {"url": "https://huggingface.co/datasets/originlab/game-recordings-free-camera-v2/blob/8731e47/README.md",
                "finding": "Other release documents log_sentinel_v3/v4 using a 16-bit code-bin midpoint and far*near_cm scaling; same transform NAME alone does not certify this Session or its near-plane scale.",
                "applied_to_this_session": False},
            "candidate_poses": candidates}
