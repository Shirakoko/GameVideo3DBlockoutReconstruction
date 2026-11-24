"""Object measurements use masked metric points, never VLM positions."""
from __future__ import annotations

import numpy as np
import open3d as o3d

from agents.object_detection import confidence, unknown
from reconstruction.backprojection import backproject
from reconstruction.coordinate_transform import transform_points


def frame_points(frame, min_depth_m, max_depth_m):
    frame.validate()
    bp = backproject(frame.depth, frame.intrinsics, spec=frame.depth_spec, valid_mask=frame.valid_mask,
        rgb=frame.rgb, min_depth_m=min_depth_m, max_depth_m=max_depth_m)
    return transform_points(bp.points_camera_m, frame.world_from_camera), bp.pixels_uv


def load_pixel_cloud(path, frame, min_depth_m, max_depth_m):
    """Optional Stage B pixel-indexed cloud. Reject stale/wrong-pose geometry.

    A fused PLY has no native mask correspondence and is intentionally NOT
    projected wholesale into moving-object masks (would reintroduce ghosts).
    """
    with np.load(path, allow_pickle=False) as data:
        if int(data["frame_index"]) != frame.frame_index:
            raise ValueError("Point cloud source frame mismatch")
        p, uv = data["points_world_m"].copy(), data["pixels_uv"].copy()
    if p.ndim != 2 or p.shape[1] != 3 or uv.shape != (len(p), 2) or not np.issubdtype(uv.dtype, np.integer) or not np.isfinite(p).all():
        raise ValueError("Pixel cloud requires finite Nx3 world metres and integer Nx2 pixels")
    h, w = frame.depth.shape
    if (uv < 0).any() or (uv[:, 0] >= w).any() or (uv[:, 1] >= h).any() or len(np.unique(uv, axis=0)) != len(uv):
        raise ValueError("Pixel cloud has invalid or duplicate pixels")
    expected, all_uv = frame_points(frame, min_depth_m, max_depth_m)
    lookup = np.full((h, w), -1, dtype=int)
    lookup[all_uv[:, 1], all_uv[:, 0]] = np.arange(len(all_uv))
    indices = lookup[uv[:, 1], uv[:, 0]]
    if (indices < 0).any() or not np.allclose(p, expected[indices], atol=1e-6, rtol=0):
        raise ValueError("Point cloud contradicts reviewed depth/mask/camera contract")
    return p, uv


def measure(detection, points, pixels, *, min_points=12):
    if points is None or detection.mask_kind == "bbox_proxy":
        return None, unknown("Verified metric geometry and an object silhouette mask are both required")
    selected = points[detection.mask[pixels[:, 1], pixels[:, 0]]]
    if len(selected) < min_points:
        return None, unknown(f"Fewer than {min_points} valid masked depth points")
    # SOR rejects isolated depth spikes; a broad surface is not clipped to a
    # semantic size prior. Bounds describe observed surfaces, not hidden volume.
    if len(selected) > 30:
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(selected))
        _, keep = cloud.remove_statistical_outlier(nb_neighbors=min(12, len(selected)-1), std_ratio=3.)
        selected = selected[np.asarray(keep, dtype=int)]
    if len(selected) < min_points:
        return None, unknown("Too few points after statistical outlier rejection")
    low, high = np.quantile(selected, [0.005, 0.995], axis=0)
    dims = np.maximum(high-low, 0.02)  # explicit inferred plane thickness
    center = (low+high)/2
    valid_fraction = len(selected)/int(detection.mask.sum())
    score = min(0.85, detection.mask_confidence["score"] * min(1., valid_fraction/0.6))
    c = confidence(score, "inferred", "masked_world_points_robust_aabb",
        "Metric depth+verified capture pose+object mask; 0.5/99.5% observed-surface AABB. Hidden extent/orientation unobserved; minimum thickness 0.02m")
    return dict(position=center, dimensions=dims, points=selected, low=low, high=high), c


def static_bounds(measurements, voxel_size=0.04):
    """One centroid per independent observation per voxel, static tracks only."""
    means = []
    for m in measurements:
        p = m["points"]
        keys = np.floor(p/voxel_size).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        means.append(np.column_stack([np.bincount(inverse, weights=p[:, axis])/counts for axis in range(3)]))
    p = np.concatenate(means)
    low, high = np.quantile(p, [0.005, 0.995], axis=0)
    return (low+high)/2, np.maximum(high-low, 0.02), p


def object_transform(position):
    # Axis-aligned blockout, NOT an estimated physical orientation.
    return dict(position=np.asarray(position).tolist(), rotation=[0., 0., 0., 1.], scale=[1., 1., 1.])
