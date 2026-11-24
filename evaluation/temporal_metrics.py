"""Frame-index joins, SE(3) pose/trajectory errors, explicit seconds clocks."""
from __future__ import annotations

import numpy as np

from reconstruction.coordinate_transform import inverse_rigid, validate_rigid


def indexed_rows(rows, *, cameras=False):
    result, last = {}, {}
    for row in rows:
        index, timestamp = row["frame_index"], row["timestamp_seconds"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("Explicit nonnegative integer Frame Index required")
        if isinstance(timestamp, bool) or not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Finite nonnegative timestamp in seconds required; QPC ticks are not seconds")
        camera = row.get("camera_id", "camera_main") if cameras else "stream"
        if not isinstance(camera, str) or not camera:
            raise ValueError("Nonempty camera identity required")
        key = (camera, index) if cameras else index
        if key in result or (camera in last and (index <= last[camera][0] or timestamp <= last[camera][1])):
            raise ValueError("Duplicate/non-monotonic source indices or timestamps")
        result[key], last[camera] = row, (index, timestamp)
    return result


def statistics(values, prefix):
    a = np.asarray(values, dtype=float)
    return {prefix+"_"+name: float(fn(a)) if len(a) else None for name, fn in {
        "mean": np.mean, "rmse": lambda x: np.sqrt(np.mean(x*x)), "median": np.median,
        "p95": lambda x: np.percentile(x, 95), "max": np.max}.items()}


def rotation_error_deg(rotation):
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1))))


def _se3_alignment(pred, truth):
    """Rigid least-squares alignment; scale is never fitted. Reflections refused."""
    if len(pred) < 3 or np.linalg.matrix_rank(pred-pred.mean(axis=0)) < 2:
        raise ValueError("SE3 alignment needs >=3 non-collinear matched positions")
    u, _, vt = np.linalg.svd((pred-pred.mean(axis=0)).T @ (truth-truth.mean(axis=0)))
    sign = np.diag([1., 1., np.linalg.det(vt.T @ u.T)])
    r = vt.T @ sign @ u.T
    return pred @ r.T + truth.mean(axis=0)-pred.mean(axis=0) @ r.T


def camera_metrics(predicted, ground_truth, *, alignment="none", rpe_delta_frames=1):
    """Absolute pose is always unaligned. ATE optionally reports SE3 alignment.

    RPE pairs GT source indices separated by delta, not successive available
    predictions. Missing frames cannot silently shorten the intended interval.
    Multiple cameras are joined by (camera_id, frame_index), never mixed.
    """
    if alignment not in ("none", "se3"):
        raise ValueError("Trajectory alignment must be none or se3; no implicit scale fitting")
    if isinstance(rpe_delta_frames, bool) or not isinstance(rpe_delta_frames, int) or rpe_delta_frames < 1:
        raise ValueError("RPE source-frame delta must be a positive integer")
    pred, gt = indexed_rows(predicted, cameras=True), indexed_rows(ground_truth, cameras=True)
    if not gt:
        raise ValueError("Ground Truth camera track is empty")
    for rows in (pred, gt):
        for row in rows.values():
            validate_rigid(row["world_from_camera"])
    keys = sorted(set(pred) & set(gt))
    translation, rotation, signed_time, per_frame, ate = [], [], [], [], []
    for key in keys:
        p, g = np.asarray(pred[key]["world_from_camera"]), np.asarray(gt[key]["world_from_camera"])
        t = float(np.linalg.norm(p[:3, 3]-g[:3, 3]))
        r = rotation_error_deg(g[:3, :3].T @ p[:3, :3])
        delta = float(pred[key]["timestamp_seconds"]-gt[key]["timestamp_seconds"])
        translation.append(t); rotation.append(r); signed_time.append(delta)
        per_frame.append(dict(camera_id=key[0], frame_index=key[1], translation_m=t, rotation_deg=r, timestamp_error_s=delta))
    for camera in sorted({key[0] for key in keys}):
        subset = [key for key in keys if key[0] == camera]
        pp = np.array([pred[key]["world_from_camera"] for key in subset])[:, :3, 3]
        gg = np.array([gt[key]["world_from_camera"] for key in subset])[:, :3, 3]
        if alignment == "se3":
            pp = _se3_alignment(pp, gg)
        ate.extend(np.linalg.norm(pp-gg, axis=1).tolist())
    relative_t, relative_r, expected_pairs = [], [], 0
    for a in sorted(gt):
        b = (a[0], a[1]+rpe_delta_frames)
        if b not in gt:
            continue
        expected_pairs += 1
        if a not in pred or b not in pred:
            continue
        delta_gt = inverse_rigid(gt[a]["world_from_camera"]) @ np.asarray(gt[b]["world_from_camera"])
        delta_pred = inverse_rigid(pred[a]["world_from_camera"]) @ np.asarray(pred[b]["world_from_camera"])
        error = inverse_rigid(delta_gt) @ delta_pred
        relative_t.append(float(np.linalg.norm(error[:3, 3])))
        relative_r.append(rotation_error_deg(error[:3, :3]))
    values = dict(camera_pose_match_count=len(keys), camera_pose_gt_count=len(gt), camera_pose_coverage=len(keys)/len(gt),
                  camera_pose_missing_frames=len(set(gt)-set(pred)), camera_pose_extra_frames=len(set(pred)-set(gt)),
                  rpe_pair_count=len(relative_t), rpe_expected_pair_count=expected_pairs)
    for a, prefix in [(translation, "camera_translation_error_m"), (rotation, "camera_rotation_error_deg"),
                      (ate, "trajectory_ate_m"), (relative_t, "trajectory_rpe_translation_m"),
                      (relative_r, "trajectory_rpe_rotation_deg"), (np.abs(signed_time), "camera_timestamp_error_s")]:
        values.update(statistics(a, prefix))
    return dict(values=values, details=dict(alignment=alignment, rpe_delta_frames=rpe_delta_frames, per_frame=per_frame),
                definition="Same source-frame camera-to-world matrices in metres; unaligned absolute pose, rigid/no-scale ATE, SE3 relative pose error")


def synchronization_metrics(streams, *, reference="rgb"):
    """Caller must supply reviewed common-origin seconds, not raw clock ticks."""
    if reference not in streams:
        raise ValueError("Synchronization reference stream missing")
    indexed = {name: indexed_rows(rows) for name, rows in streams.items()}
    truth = indexed[reference]
    if not truth:
        raise ValueError("Synchronization reference stream empty")
    results = {}
    for name, rows in indexed.items():
        if name == reference:
            continue
        keys = sorted(set(rows) & set(truth))
        signed = np.array([rows[k]["timestamp_seconds"]-truth[k]["timestamp_seconds"] for k in keys])
        times = np.array([truth[k]["timestamp_seconds"] for k in keys])
        drift = float(np.polyfit(times-times[0], signed, 1)[0]*1e6) if len(keys) > 1 else None
        values = dict(sync_match_count=len(keys), sync_coverage=len(keys)/len(truth), sync_missing_frames=len(set(truth)-set(rows)),
                      sync_extra_frames=len(set(rows)-set(truth)), sync_signed_offset_s=float(signed.mean()) if len(keys) else None,
                      sync_clock_drift_ppm=drift)
        values.update(statistics(np.abs(signed), "sync_error_s"))
        results[name] = dict(values=values, details=dict(reference=reference,
            per_frame=[dict(frame_index=k, signed_error_s=float(e)) for k, e in zip(keys, signed)]),
            definition="Timestamp difference at equal reviewed source Frame Index; no nearest-time matching, offset correction or index guessing")
    return results

