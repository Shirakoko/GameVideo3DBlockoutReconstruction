"""Independent raycast five-category fixture with moving person/camera."""
from dataclasses import replace

import numpy as np

from agents.object_detection import SyntheticColorProvider
from reconstruction.backprojection import DepthSpec
from reconstruction.coordinate_transform import world_from_optical_ue_pose
from reconstruction.pointcloud_fusion import FrameInput
from reconstruction.synthetic import ray_box, independent_ue_basis


def make_scene_fixture():
    w, h, focal = 240, 140, 135.
    k = np.array([[focal, 0., w/2], [0., focal, h/2], [0., 0., 1.]])
    v, u = np.mgrid[:h, :w]
    rays = np.stack([np.ones((h, w)), (u+0.5-w/2)/focal, -(v+0.5-h/2)/focal], axis=-1)
    spec = DepthSpec("m", 1., "planar_z", 0.5, True, "SYNTHETIC independent ray/AABB renderer, metre planar ray parameter, half pixel centres")
    frames, truth = [], {}
    for ordinal, clip_index in enumerate([0, 60, 120, 180, 240]):
        origin_ue = np.array([0.1*ordinal, -0.2+ordinal*0.08, 2.2])
        pitch, yaw = -8., 1.-ordinal*0.3
        directions = rays @ independent_ue_basis(pitch, yaw, 0).T
        directions[..., 1] *= -1
        origin = origin_ue*[1, -1, 1]
        boxes = {
            "person": ([4.+ordinal*0.45, -0.4, 0.], [4.6+ordinal*0.45, 0.2, 1.8]),
            "tree": ([7.5, -3.4, 0.], [8.5, -2.4, 4.2]),
            "rock": ([5.5, 2., 0.], [7., 3.4, 1.5]),
            "barrier": ([8.5, 0.8, 0.], [8.8, 2., 1.1])
        }
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -origin[2]/directions[..., 2]
        t = np.where((t > 0) & (t < 35), t, np.inf)
        depth = t.copy()
        labels = np.where(np.isfinite(depth), "ground", "").astype("U16")
        for category, (low, high) in boxes.items():
            hit = ray_box(origin, directions, np.array(low), np.array(high))
            nearer = hit < depth
            depth = np.minimum(depth, hit)
            labels[nearer] = category
        rgb = np.full((h, w, 3), [75, 110, 180], dtype=np.uint8)
        for category, color in SyntheticColorProvider.palette.items():
            rgb[labels == category] = color
        valid = np.isfinite(depth) & (depth < 35)
        depth[~valid] = np.nan
        index = 2000+clip_index
        frame = FrameInput(index, index, index, 8000+clip_index, clip_index/60., depth, valid, rgb, k.copy(),
            world_from_optical_ue_pose(origin_ue*100, pitch_deg=pitch, yaw_deg=yaw, roll_deg=0, world_units_per_meter=100),
            spec, True, "SYNTHETIC raycaster uses same-call capture pose", f"synthetic-capture-{index}")
        frames.append(frame)
        truth[index] = dict(labels=labels, boxes=boxes, points=origin+directions*depth[..., None])
    # A duplicate delivery must not add motion evidence or geometric weight.
    frames.insert(1, replace(frames[0], frame_index=2030, rgb_frame_index=2030, depth_frame_index=2030, original_tick=8030, timestamp_seconds=0.5))
    truth[2030] = truth[2000]
    return frames, truth


def check_fixture(scene, frames, truth):
    """Persist measured, explicitly synthetic acceptance criteria, no GT claims
    about the real RGB detector. Fail the demo when a required class is lost.
    """
    from agents.object_detection import mask_iou
    from reconstruction.object_geometry import frame_points
    provider = SyntheticColorProvider()
    categories = set(provider.palette)
    by_category = {c: [o for o in scene["objects"] if o["category"] == c] for c in categories}
    if any(len(v) != 1 for v in by_category.values()):
        raise ValueError("Synthetic acceptance failed: all five categories must keep exactly one ID")
    delivered = {f.frame_index for f in frames}
    if any(set(v[0]["source_frames"]) != delivered for v in by_category.values()):
        raise ValueError("Synthetic acceptance failed: a required category dropped a keyframe")
    if any(v[0]["is_static"] != (c != "person") for c, v in by_category.items()):
        raise ValueError("Synthetic acceptance failed: wrong static/dynamic classification")
    min_iou, max_error, center_error = 1., 0., []
    for frame in frames:
        for d in provider.detect(frame.rgb, frame.frame_index):
            min_iou = min(min_iou, mask_iou(d.mask, truth[frame.frame_index]["labels"] == d.category))
        p, uv = frame_points(frame, 0.1, 35)
        expected = truth[frame.frame_index]["points"][uv[:, 1], uv[:, 0]]
        max_error = max(max_error, float(np.linalg.norm(p-expected, axis=1).max()))
    for r in by_category["person"][0]["observations"]:
        low, high = np.array(truth[r["frame_index"]]["boxes"]["person"])
        center_error.append(float(np.linalg.norm(np.array(r["position"])-(low+high)/2)))
    if min_iou != 1. or max_error > 1e-10 or max(center_error) > 0.35:
        raise ValueError("Synthetic acceptance failed: segmentation/geometry tolerance exceeded")
    return dict(status="passed", scope="SYNTHETIC palette/raycast fixture ONLY; NOT real vision accuracy",
        category_count=5, persistent_ids_per_category={c: len(v) for c, v in sorted(by_category.items())},
        id_switches=0, static_dynamic_accuracy=1., mask_iou_min=min_iou,
        backprojection_max_error_m=max_error, person_surface_center_vs_volume_center_max_error_m=max(center_error),
        independent_captures=5, delivered_keyframes=len(frames), duplicate_deliveries=1,
        tolerance_person_center_m=0.35, hidden_volume_recovered=False)
