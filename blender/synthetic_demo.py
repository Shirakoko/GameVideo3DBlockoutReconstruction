"""Known-metric independent ray/AABB original, ONLY for Stage E/F QA."""
from __future__ import annotations

import numpy as np

from agents.object_detection import SyntheticColorProvider
from reconstruction.backprojection import DepthSpec
from reconstruction.coordinate_transform import world_from_optical_ue_pose
from reconstruction.pointcloud_fusion import FrameInput
from reconstruction.synthetic import independent_ue_basis, ray_box


def demo_frame(index, fps=60):
    time = index/fps
    w, h, focal = 240, 140, 135.
    k = np.array([[focal, 0., w/2], [0., focal, h/2], [0., 0., 1.]])
    v, u = np.mgrid[:h, :w]
    rays = np.stack([np.ones((h, w)), (u+.5-w/2)/focal, -(v+.5-h/2)/focal], axis=-1)
    origin_ue = np.array([time/30, -.2+time*2/75, 2.2])
    pitch, yaw = -8., 1.-time/10
    directions = rays @ independent_ue_basis(pitch, yaw, 0).T
    directions[..., 1] *= -1
    origin = origin_ue*[1, -1, 1]
    boxes = {"person": ([4.+time*.15, -.4, 0.], [4.6+time*.15, .2, 1.8]),
             "tree": ([7.5, -3.4, 0.], [8.5, -2.4, 4.2]), "rock": ([5.5, 2., 0.], [7., 3.4, 1.5]),
             "barrier": ([8.5, .8, 0.], [8.8, 2., 1.1])}
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -origin[2]/directions[..., 2]
    depth = np.where((t > 0) & (t < 35), t, np.inf)
    labels = np.where(np.isfinite(depth), "ground", "").astype("U16")
    for category, (low, high) in boxes.items():
        hit = ray_box(origin, directions, np.array(low), np.array(high))
        labels[hit < depth] = category
        depth = np.minimum(depth, hit)
    rgb = np.full((h, w, 3), [75, 110, 180], dtype=np.uint8)
    for category, color in SyntheticColorProvider.palette.items():
        rgb[labels == category] = color
    valid = np.isfinite(depth) & (depth < 35)
    depth[~valid] = np.nan
    spec = DepthSpec("m", 1., "planar_z", .5, True, "SYNTHETIC independent ray/AABB renderer; metre planar depth, half pixel centres")
    source_index = 7000+index
    return FrameInput(source_index, source_index, source_index, 17000+index, time, depth, valid, rgb, k,
        world_from_optical_ue_pose(origin_ue*100, pitch_deg=pitch, yaw_deg=yaw, roll_deg=0, world_units_per_meter=100),
        spec, True, "SYNTHETIC same-call RGB/depth/camera capture", f"synthetic-ef-{source_index}")
