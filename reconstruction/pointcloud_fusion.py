"""Open3D world alignment, per-observation voxel fusion, SOR, and provenance."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from reconstruction.backprojection import Backprojection, DepthSpec, backproject
from reconstruction.coordinate_transform import validate_rigid
from stage_a.depth import UnverifiedContractError


@dataclass
class FrameInput:
    frame_index: int
    rgb_frame_index: int
    depth_frame_index: int
    original_tick: int
    timestamp_seconds: float
    depth: np.ndarray
    valid_mask: np.ndarray
    rgb: np.ndarray
    intrinsics: np.ndarray
    world_from_camera: np.ndarray
    depth_spec: DepthSpec
    geometry_pose_verified: bool
    sync_evidence: str
    source_observation_id: str

    def validate(self):
        self.depth_spec.validate()
        for value in (self.frame_index, self.rgb_frame_index, self.depth_frame_index, self.original_tick):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Explicit nonnegative integer source indices required")
        if self.rgb_frame_index != self.depth_frame_index or self.frame_index != self.depth_frame_index:
            raise ValueError("RGB/depth/pose input must share the reviewed delivery frame index")
        if not np.isfinite(self.timestamp_seconds) or self.timestamp_seconds < 0:
            raise ValueError("Finite source timestamp in seconds required")
        if self.geometry_pose_verified is not True or not isinstance(self.sync_evidence, str) or not self.sync_evidence.strip():
            raise UnverifiedContractError("Render-pose binding, not just matching frame labels, must be verified before geometry")
        if not isinstance(self.source_observation_id, str) or not self.source_observation_id.strip():
            raise ValueError("Explicit source observation identity required; session dup_count is insufficient")
        validate_rigid(self.world_from_camera)


@dataclass
class FusionResult:
    cloud: o3d.geometry.PointCloud
    first_frame_cloud: o3d.geometry.PointCloud
    first_backprojection: Backprojection
    provenance_offsets: np.ndarray
    provenance_frame_indices: np.ndarray
    provenance_timestamps: np.ndarray
    frame_reports: list[dict]
    stats: dict


def point_cloud(points, colors_rgb8):
    p = np.asarray(points, dtype=np.float64)
    c = np.asarray(colors_rgb8)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or c.shape != p.shape or c.dtype != np.uint8:
        raise ValueError("Finite Nx3 points and Nx3 RGB uint8 colors required")
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(p)
    result.colors = o3d.utility.Vector3dVector(c.astype(np.float64)/255.)
    return result


def input_digest(frame):
    """Same depth under a MOVING pose is not a geometric duplicate.

    Exact equality is a conservative geometric-input deduplication, not proof
    of an encoder duplicate. Source observation identities are separate.
    """
    digest = hashlib.sha256()
    for array in (frame.depth, frame.valid_mask, frame.rgb, frame.intrinsics, frame.world_from_camera):
        a = np.ascontiguousarray(array)
        digest.update(str((a.shape, a.dtype.str)).encode("ascii"))
        digest.update(a.tobytes())
    digest.update(repr(asdict(frame.depth_spec)).encode("utf-8"))
    return digest.hexdigest()


def voxel_trace(cloud, voxel_size, origin, upper):
    reduced, _, trace = cloud.voxel_down_sample_and_trace(voxel_size, origin, upper, False)
    return reduced, [np.asarray(indices, dtype=np.int64) for indices in trace]


def fuse_frames(frames: list[FrameInput], *, voxel_size_m: float, min_depth_m: float,
                max_depth_m: float, stride: int = 1, sor_neighbors: int = 20,
                sor_std_ratio: float = 2., max_input_points: int = 2_000_000) -> FusionResult:
    """Each independent observation contributes ONE mean per world voxel.

    Dense nearby frames and exact duplicate deliveries cannot overwhelm a
    voxel by pixel count. This is static-scene point fusion, not TSDF, motion
    segmentation, hole filling, or ICP optimization of ground-truth poses.
    """
    if not frames or not np.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise ValueError("Nonempty frames and positive finite metric voxel size required")
    if isinstance(sor_neighbors, bool) or not isinstance(sor_neighbors, int) or sor_neighbors < 2 or not np.isfinite(sor_std_ratio) or sor_std_ratio <= 0:
        raise ValueError("SOR requires >=2 neighbours and positive finite std ratio")
    if isinstance(max_input_points, bool) or not isinstance(max_input_points, int) or max_input_points < 1:
        raise ValueError("Positive input point budget required")
    # Validate the WHOLE batch before producing geometry.
    for frame in frames:
        frame.validate()
    if any(b.frame_index <= a.frame_index or b.timestamp_seconds <= a.timestamp_seconds for a, b in zip(frames, frames[1:])):
        raise ValueError("Frame indices and timestamps must be strictly increasing; never sorted silently")
    identities, hashes = {}, {}
    clouds, backprojections, accepted, reports = [], [], [], []
    point_count = 0
    for frame in frames:
        digest = input_digest(frame)
        record = {"frame_index": frame.frame_index, "rgb_frame_index": frame.rgb_frame_index,
                  "depth_frame_index": frame.depth_frame_index, "original_tick": frame.original_tick,
                  "timestamp_seconds": frame.timestamp_seconds, "source_observation_id": frame.source_observation_id,
                  "world_from_optical_camera_m": frame.world_from_camera.tolist(),
                  "K": frame.intrinsics.tolist(), "depth_spec": asdict(frame.depth_spec),
                  "sync_evidence": frame.sync_evidence, "input_sha256": digest}
        if frame.source_observation_id in identities:
            previous_digest, previous_index = identities[frame.source_observation_id]
            if previous_digest != digest:
                raise ValueError("Repeated source observation has conflicting pixels/calibration/capture pose")
            record.update(status="skipped_source_duplicate", duplicate_of_frame_index=previous_index)
        elif digest in hashes:
            record.update(status="skipped_exact_geometric_input", duplicate_of_frame_index=hashes[digest])
        else:
            sampled_upper_bound = ((frame.depth.shape[0]+stride-1)//stride)*((frame.depth.shape[1]+stride-1)//stride) if isinstance(stride, int) and not isinstance(stride, bool) and stride > 0 and frame.depth.ndim == 2 else 0
            if sampled_upper_bound > max_input_points:
                raise ValueError("Single-frame point budget exceeded before allocation; increase stride")
            bp = backproject(frame.depth, frame.intrinsics, spec=frame.depth_spec,
                             valid_mask=frame.valid_mask, rgb=frame.rgb,
                             min_depth_m=min_depth_m, max_depth_m=max_depth_m, stride=stride)
            point_count += len(bp.points_camera_m)
            if point_count > max_input_points:
                raise ValueError("Input point budget exceeded; increase stride or bound the frame selection")
            if not len(bp.points_camera_m):
                record.update(status="skipped_no_valid_depth", backprojection=bp.stats)
            else:
                cloud = point_cloud(bp.points_camera_m, bp.colors_rgb8)
                cloud.transform(validate_rigid(frame.world_from_camera))
                clouds.append(cloud)
                backprojections.append(bp)
                accepted.append(frame)
                hashes[digest] = frame.frame_index
                record.update(status="accepted", backprojection=bp.stats)
        identities[frame.source_observation_id] = (digest, frame.frame_index if digest not in hashes else hashes[digest])
        reports.append(record)
    if not clouds:
        raise ValueError("No valid depth remains after contract and depth-range filtering")
    lower = np.min([cloud.get_min_bound() for cloud in clouds], axis=0)
    upper = np.max([cloud.get_max_bound() for cloud in clouds], axis=0)
    origin = np.floor(lower/voxel_size_m)*voxel_size_m - voxel_size_m
    upper = np.ceil(upper/voxel_size_m)*voxel_size_m + voxel_size_m
    combined = o3d.geometry.PointCloud()
    labels = []
    for frame, cloud in zip(accepted, clouds):
        reduced, _ = voxel_trace(cloud, voxel_size_m, origin, upper)
        combined += reduced
        labels.extend([frame.frame_index]*len(reduced.points))
        next(record for record in reports if record["frame_index"] == frame.frame_index)["per_frame_voxels"] = len(reduced.points)
    fused, trace = voxel_trace(combined, voxel_size_m, origin, upper)
    before_sor = len(fused.points)
    if before_sor <= sor_neighbors:
        retained = np.arange(before_sor, dtype=np.int64)
        sor_status = "not_applied_too_few_points"
    else:
        fused, retained_list = fused.remove_statistical_outlier(nb_neighbors=sor_neighbors, std_ratio=sor_std_ratio)
        retained = np.asarray(retained_list, dtype=np.int64)
        sor_status = "applied"
    if not len(fused.points):
        raise ValueError("Statistical outlier removal removed every point")
    offsets, members = [0], []
    label_array = np.asarray(labels, dtype=np.int64)
    for index in retained:
        source_frames = np.unique(label_array[trace[index]])
        members.extend(source_frames.tolist())
        offsets.append(len(members))
    timestamps = {frame.frame_index: frame.timestamp_seconds for frame in accepted}
    return FusionResult(fused, clouds[0], backprojections[0], np.asarray(offsets, dtype=np.int64),
        np.asarray(members, dtype=np.int64), np.asarray([timestamps[index] for index in members]), reports,
        {"input_frames": len(frames), "accepted_observations": len(clouds),
         "skipped_duplicate_frames": sum(record["status"].startswith("skipped_source") or record["status"] == "skipped_exact_geometric_input" for record in reports),
         "raw_valid_input_points": point_count, "per_frame_voxel_points": len(labels),
         "voxels_before_sor": before_sor, "outliers_removed": before_sor-len(fused.points),
         "output_points": len(fused.points), "voxel_size_m": voxel_size_m,
         "sor_neighbors": sor_neighbors, "sor_std_ratio": sor_std_ratio, "sor_status": sor_status,
         "voxel_origin_m": origin.tolist(), "weighting": "one centroid per independent observation per world voxel",
         "duplicate_policy": "verified source identity OR exactly identical depth+mask+RGB+K+capture pose+depth spec",
         "provenance": "CSR per output vertex; skipped deliveries remain aliases in frame_reports"})


def write_verified_ply(path: Path, cloud, *, data_source: str) -> dict:
    """Export binary PLY with units/evidence comments; reload and verify bytes."""
    if not len(cloud.points) or not np.isfinite(np.asarray(cloud.points)).all():
        raise ValueError("Cannot export empty/nonfinite geometry")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False):
        raise OSError(f"PLY write failed: {path}")
    payload = path.read_bytes()
    label = data_source.replace("\r", " ").replace("\n", " ").encode("ascii", errors="replace")
    prefix = b"ply\n"
    if not payload.startswith(prefix):
        raise OSError("Unexpected Open3D PLY header")
    comments = b"comment units meter\ncomment coordinates Blender_RH_Z_up\ncomment data_source " + label + b"\n"
    # PLY requires the format line immediately after "ply"; comments follow it.
    insert_at = payload.index(b"\n", len(prefix)) + 1
    path.write_bytes(payload[:insert_at] + comments + payload[insert_at:])
    loaded = o3d.io.read_point_cloud(str(path))
    expected, actual = np.asarray(cloud.points), np.asarray(loaded.points)
    if actual.shape != expected.shape or not np.allclose(actual, expected, atol=1e-12, rtol=0):
        raise OSError("PLY coordinate roundtrip failed")
    if not np.allclose(np.asarray(loaded.colors), np.asarray(cloud.colors), atol=0.5/255+1e-10, rtol=0):
        raise OSError("PLY RGB quantization roundtrip failed")
    return {"path": str(path.resolve()), "vertices": len(actual), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "roundtrip_verified": True,
            "data_source": data_source, "coordinate_system": "Blender RH Z-up", "unit": "m"}
