"""Read source records and join only on explicit, validated Frame Index mappings."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path


class SyncError(ValueError):
    pass


def read_jsonl(path: Path, *, max_records=100000, max_line_bytes=1024*1024):
    """Streaming reader: preserves original keys/units and fails on malformed data."""
    with path.open("rb") as stream:
        for line_number in range(1, max_records + 2):
            line = stream.readline(max_line_bytes + 1)
            if not line:
                return
            if line_number > max_records or len(line) > max_line_bytes:
                raise SyncError("JSONL record/line budget exceeded")
            try:
                record = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            except (ValueError, UnicodeError) as error:
                raise SyncError(f"Malformed JSONL at line {line_number}") from error
            if not isinstance(record, dict):
                raise SyncError(f"JSONL line {line_number} is not an object")
            yield record


def read_frame_table(path: Path, *, columns: list[str], max_rows=100000):
    """Optional PyArrow, explicit columns, metadata row cap before allocating."""
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SyncError("Install pyarrow to inspect an authorized frames.parquet") from error
    table = pq.ParquetFile(path)
    if table.metadata.num_rows > max_rows:
        raise SyncError("Frame table exceeds row budget; use a reviewed row-group adapter")
    if any(name not in table.schema.names for name in columns):
        raise SyncError("Requested frame-table fields are absent")
    for batch in table.iter_batches(batch_size=1024, columns=columns):
        yield from batch.to_pylist()


def frame_index(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SyncError("Frame Index must be a nonnegative integer, not a row number or rounded timestamp")
    return value


def join_frames(rgb_indices, depth_indices, camera_records, *, camera_frame_key: str):
    """Return all pose samples for each frame. Never silently choose first/last.

    Callers MUST supply the source Frame Index mapping for each decoded ordinal,
    reviewed against session alignment and frame-table conversion manifest.
    Missing pose is not interpolated; a multiple-sample frame requires a later
    verified QPC binning/aggregation rule before a single pose can be exported.
    """
    rgb = [frame_index(value) for value in rgb_indices]
    depth = [frame_index(value) for value in depth_indices]
    if not rgb or not depth:
        raise SyncError("RGB and Depth frame mappings must be nonempty")
    for label, indices in (("RGB", rgb), ("Depth", depth)):
        if any(b <= a for a, b in zip(indices, indices[1:])):
            raise SyncError(f"{label} mapping must be unique and strictly increasing")
    poses = defaultdict(list)
    for record in camera_records:
        if camera_frame_key not in record:
            raise SyncError("Camera Frame Index field absent; timestamp fallback is forbidden")
        poses[frame_index(record[camera_frame_key])].append(dict(record))
    depth_ordinals = {index: ordinal for ordinal, index in enumerate(depth)}
    rows = [{"frame_index": index, "rgb_clip_ordinal": ordinal,
             "depth_clip_ordinal": depth_ordinals.get(index), "camera_samples": poses.get(index, [])}
            for ordinal, index in enumerate(rgb)]
    gaps = [[a, b] for a, b in zip(rgb, rgb[1:]) if b != a+1]
    report = {
        "rgb_frames": len(rgb), "depth_frames": len(depth),
        "missing_depth_frame_indices": sorted(set(rgb)-set(depth)),
        "extra_depth_frame_indices": sorted(set(depth)-set(rgb)),
        "missing_camera_frame_indices": [row["frame_index"] for row in rows if not row["camera_samples"]],
        "multiple_camera_sample_frame_indices": [row["frame_index"] for row in rows if len(row["camera_samples"]) > 1],
        "rgb_index_gap_edges": gaps,
        "camera_samples_per_frame_histogram": dict(Counter(len(row["camera_samples"]) for row in rows)),
        "timestamp_sync_verified": False,
        "single_pose_rule_verified": False,
        "mapping_provenance_verified": False,
    }
    report["index_join_complete"] = not any(report[name] for name in
        ("missing_depth_frame_indices", "extra_depth_frame_indices", "missing_camera_frame_indices"))
    return rows, report


def compare_timestamps(rgb_seconds, depth_seconds, *, tolerance_seconds: float):
    if not math.isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise SyncError("Explicit finite timestamp tolerance required")
    if len(rgb_seconds) != len(depth_seconds) or not rgb_seconds:
        raise SyncError("Equal, nonempty timestamp vectors required")
    for values in (rgb_seconds, depth_seconds):
        if not all(math.isfinite(value) for value in values) or any(b <= a for a, b in zip(values, values[1:])):
            raise SyncError("Timestamps must be finite, unique and strictly increasing")
    errors = [abs(a-b) for a, b in zip(rgb_seconds, depth_seconds)]
    return {"max_error_seconds": max(errors), "tolerance_seconds": tolerance_seconds,
            "within_tolerance": max(errors) <= tolerance_seconds,
            "scope": "supplied same-clock timestamps only; raw QPC units are not seconds"}
