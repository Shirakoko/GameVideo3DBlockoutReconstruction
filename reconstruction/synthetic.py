"""Independent analytical room/box ray caster with declared centimetre depth."""
from dataclasses import replace

import numpy as np

from reconstruction.backprojection import DepthSpec
from reconstruction.coordinate_transform import world_from_optical_ue_pose
from reconstruction.pointcloud_fusion import FrameInput


def independent_ue_basis(pitch, yaw, roll):
    p, y, r = np.deg2rad([pitch, yaw, roll])
    sp, sy, sr = np.sin([p, y, r])
    cp, cy, cr = np.cos([p, y, r])
    return np.column_stack(([cp*cy, cp*sy, sp],
        [sr*sp*cy-cr*sy, sr*sp*sy+cr*cy, -sr*cp],
        [-cr*sp*cy-sr*sy, -cr*sp*sy+sr*cy, cr*cp]))


def ray_box(origin, directions, low, high):
    with np.errstate(divide="ignore", invalid="ignore"):
        first = (low-origin)/directions
        second = (high-origin)/directions
    near = np.max(np.minimum(first, second), axis=-1)
    far = np.min(np.maximum(first, second), axis=-1)
    hit = (far >= np.maximum(near, 1e-8))
    return np.where(hit, np.where(near > 1e-8, near, far), np.inf)


def make_synthetic_frames():
    width, height, focal = 160, 90, 80.
    k = np.array([[focal, 0., width/2], [0., focal, height/2], [0., 0., 1.]])
    vv, uu = np.mgrid[:height, :width]
    # Independent ray construction, not inv(K) or the transform under test.
    right = (uu+0.5-width/2)/focal
    down = (vv+0.5-height/2)/focal
    ue_rays = np.stack([np.ones_like(right), right, -down], axis=-1)
    spec = DepthSpec("cm", 0.01, "planar_z", 0.5, True,
                     "SYNTHETIC analytical ray parameter = optical planar Z; explicitly stored in cm; half-pixel centres")
    frames, ground_truth = [], {}
    raw_poses = []
    for ordinal, clip_frame in enumerate([0, 120, 240, 360, 480, 600, 719]):
        position_ue_m = np.array([0.25+ordinal*0.38, -1.05+ordinal*0.35, 1.7])
        pitch, yaw, roll = -7., 14.-ordinal*4., (ordinal-3)*0.3
        rotation = independent_ue_basis(pitch, yaw, roll)
        directions = ue_rays @ rotation.T
        # Blender world choice: reflect world Y, keep Z up.
        directions[..., 1] *= -1
        origin = position_ue_m * [1., -1., 1.]
        room = ray_box(origin, directions, np.array([-2., -4., 0.]), np.array([9., 4., 5.]))
        block = ray_box(origin, directions, np.array([4., -1.3, 0.]), np.array([5.5, 0.7, 2.4]))
        distance = np.minimum(room, block)
        truth = origin + directions*distance[..., None]
        color = np.empty((height, width, 3), dtype=np.uint8)
        color[:] = [96, 150, 218]
        color[np.abs(truth[..., 2]) < 1e-7] = [165, 165, 170]
        color[np.abs(truth[..., 2]-5.) < 1e-7] = [216, 221, 230]
        color[block < room] = [236, 137, 55]
        checker = (np.floor(truth[..., 0]*2)+np.floor(truth[..., 1]*2)+np.floor(truth[..., 2]*2)).astype(np.int64)%2
        color = np.clip(color.astype(float)*(0.82+0.18*checker[..., None]), 0, 255).astype(np.uint8)
        depth_cm = distance*100.
        valid = np.isfinite(distance)
        valid[:3, :3] = False  # synthetic sky/invalid region, explicitly masked.
        depth_cm[3, 0], depth_cm[3, 1], depth_cm[3, 2] = 0., np.nan, np.inf
        depth_cm[3, 3] = 10000.  # beyond explicitly selected 40 m reconstruction range.
        frame_index = 1000+clip_frame
        frame = FrameInput(frame_index, frame_index, frame_index, 5000+clip_frame, clip_frame/60.,
            depth_cm, valid, color, k.copy(), world_from_optical_ue_pose(position_ue_m*100,
                pitch_deg=pitch, yaw_deg=yaw, roll_deg=roll, world_units_per_meter=100.),
            spec, True, "SYNTHETIC same-call ray caster and camera pose; known delivery/tick mapping", f"synthetic-capture-{frame_index}")
        frames.append(frame)
        ground_truth[frame_index] = truth
        raw_poses.append({"frame_index": frame_index, "position_ue_cm": (position_ue_m*100).tolist(),
                          "pitch_deg": pitch, "yaw_deg": yaw, "roll_deg": roll})
    duplicate = replace(frames[0], frame_index=1030, rgb_frame_index=1030, depth_frame_index=1030,
                        original_tick=5030, timestamp_seconds=0.5)
    frames.insert(1, duplicate)
    return frames, ground_truth, {"data_source": "SYNTHETIC ONLY — NOT HUGGING FACE",
        "raster": [width, height], "depth_unit": "cm", "world_unit": "cm", "depth_type": "planar_z",
        "pixel_center_offset": 0.5, "room_aabb_blender_m": [[-2, -4, 0], [9, 4, 5]],
        "block_aabb_blender_m": [[4, -1.3, 0], [5.5, 0.7, 2.4]], "raw_poses": raw_poses,
        "duplicate_delivery_frame": 1030, "duplicate_original_observation": 1000}
