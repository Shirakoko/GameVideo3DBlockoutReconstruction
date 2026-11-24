"""Empirical geometry route. Never constructs a verified DepthSpec.

All estimates are conditional on telemetry scale, selected camera convention,
static feature correspondences and observed depth codes. RGB holdout frames
are excluded from convention selection and curve fitting.
"""
from __future__ import annotations

import numpy as np
from functools import lru_cache
from scipy.interpolate import PchipInterpolator
from scipy.optimize import isotonic_regression
from scipy.optimize import least_squares

from reconstruction.coordinate_transform import inverse_rigid, transform_points, world_from_optical_ue_pose

HOLDOUT = (60, 180, 300, 420, 540, 660)


def selected_frames(count, step=12):
    if count < 2 or step < 1:
        raise ValueError("At least two frames and a positive sampling step required")
    return sorted(set(range(0, count, step)) | {count-1})


def pose_track(rows, world_units_per_meter, *, yaw_sign=1, pitch_sign=1, angular_unit="degrees", field="camera_raw", frame_offset=0):
    factor = 180/np.pi if angular_unit == "radians" else 1.
    result = []
    for index in range(len(rows)):
        row = rows[min(max(index+frame_offset, 0), len(rows)-1)]
        p = row[field]
        if p is None:
            raise ValueError("Missing camera snapshot: "+field)
        result.append(world_from_optical_ue_pose([p[k] for k in ("px", "py", "pz")],
            pitch_deg=p["pitch"]*pitch_sign*factor, yaw_deg=p["yaw"]*yaw_sign*factor,
            roll_deg=p["roll"]*factor, world_units_per_meter=world_units_per_meter))
    return result


def features(rgb, static_mask, max_features=8000):
    import cv2
    sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=.025)
    keypoints, descriptors = sift.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
        static_mask.astype(np.uint8)*255)
    return np.array([k.pt for k in keypoints], dtype=np.float64).reshape(-1, 2), descriptors


def match_features(a, b):
    import cv2
    if a[1] is None or b[1] is None or min(len(a[0]), len(b[0])) < 2:
        return np.empty((0, 2)), np.empty((0, 2))
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(a[1], b[1], k=2)
    reverse = matcher.knnMatch(b[1], a[1], k=2)
    back = {m.queryIdx: m.trainIdx for pair in reverse if len(pair) == 2
            for m, n in [pair] if m.distance < .75*n.distance}
    good = [m for pair in forward if len(pair) == 2 for m, n in [pair]
            if m.distance < .75*n.distance and back.get(m.trainIdx) == m.queryIdx]
    return a[0][[m.queryIdx for m in good]], b[0][[m.trainIdx for m in good]]


def epipolar_errors(uv_a, uv_b, pose_a, pose_b, k):
    relative = inverse_rigid(pose_b) @ pose_a
    t = relative[:3, 3]
    skew = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    inv_k = np.linalg.inv(k)
    f = inv_k.T @ skew @ relative[:3, :3] @ inv_k
    a = np.column_stack((uv_a, np.ones(len(uv_a))))
    b = np.column_stack((uv_b, np.ones(len(uv_b))))
    lb, la = a @ f.T, b @ f
    residual = np.abs(np.sum(b*lb, axis=1))
    denominator = np.sqrt(lb[:, 0]**2+lb[:, 1]**2)+np.sqrt(la[:, 0]**2+la[:, 1]**2)
    return np.divide(2*residual, denominator, out=np.full(len(a), np.inf), where=denominator > 1e-12)


def project(points, pose, k):
    local = transform_points(points, inverse_rigid(pose))
    projected = local @ np.asarray(k).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = projected[:, :2]/projected[:, 2, None]
    return uv, local[:, 2]


def triangulate(uv_a, uv_b, pose_a, pose_b, k, *, min_angle_deg=1.):
    import cv2
    if not len(uv_a):
        return np.empty((0, 3)), np.zeros(0, dtype=bool)
    pa, pb = np.asarray(k) @ inverse_rigid(pose_a)[:3], np.asarray(k) @ inverse_rigid(pose_b)[:3]
    h = cv2.triangulatePoints(pa, pb, uv_a.T, uv_b.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        points = h[:, :3]/h[:, 3, None]
    finite = np.isfinite(points).all(axis=1)
    safe = np.where(finite[:, None], points, 0.)
    ua, za = project(safe, pose_a, k)
    ub, zb = project(safe, pose_b, k)
    ra, rb = safe-pose_a[:3, 3], safe-pose_b[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.sum(ra*rb, axis=1)/(np.linalg.norm(ra, axis=1)*np.linalg.norm(rb, axis=1))
    angle = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
    keep = (finite & (za > .1) & (zb > .1) & (za < 200) & (zb < 200)
            & (angle >= min_angle_deg) & (np.linalg.norm(ua-uv_a, axis=1) <= 3)
            & (np.linalg.norm(ub-uv_b, axis=1) <= 3))
    return safe[keep], keep


def fit_curve(codes, depths, bins=32):
    codes, depths = np.asarray(codes, float), np.asarray(depths, float)
    keep = np.isfinite(codes) & np.isfinite(depths) & (codes > 0) & (codes < 1020) & (depths > .1) & (depths < 200)
    codes, depths = codes[keep], depths[keep]
    if len(codes) < 100:
        raise ValueError("Insufficient static triangulated samples for empirical depth calibration")
    edges = np.unique(np.quantile(codes, np.linspace(0, 1, bins+1)))
    x, y, weight = [], [], []
    for low, high in zip(edges, edges[1:]):
        selected = (codes >= low) & (codes <= high)
        if selected.sum() >= 5:
            x.append(float(np.median(codes[selected])))
            y.append(float(np.median(np.log(depths[selected]))))
            weight.append(int(selected.sum()))
    # Quantile bins can share a code; combine them before monotonic fitting.
    unique = sorted(set(x))
    yy = [np.average([v for xx, v in zip(x, y) if xx == code], weights=[w for xx, w in zip(x, weight) if xx == code]) for code in unique]
    ww = [sum(w for xx, w in zip(x, weight) if xx == code) for code in unique]
    if len(unique) < 4 or unique[-1]-unique[0] < 10:
        raise ValueError("Depth code support too narrow; extrapolation refused")
    fitted = isotonic_regression(yy, weights=ww, increasing=True).x
    return dict(code_knots=unique, log_depth_knots=fitted.tolist(), sample_count=int(len(codes)),
                method="weighted monotonic log-depth PCHIP from static RGB triangulation",
                provenance="inferred", physical_contract_verified=False, extrapolation=False)


def decode_empirical(codes, curve):
    if curve.get("model") == "robust_log1p":
        a, b = curve["parameters"]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            values = a+np.log(np.expm1(np.asarray(codes, float)*np.exp(b)))
        values = np.where((np.asarray(codes) >= curve["code_knots"][0]) &
            (np.asarray(codes) <= curve["code_knots"][-1]), values, np.nan)
    else:
        values = PchipInterpolator(curve["code_knots"], curve["log_depth_knots"], extrapolate=False)(codes)
    with np.errstate(over="ignore", invalid="ignore"):
        depth = np.exp(values).astype(np.float32)
    valid = np.isfinite(depth) & (codes > 0) & (codes < 1020) & (depth >= .1) & (depth <= 80)
    depth[~valid] = 0.
    return depth, valid


def fit_smooth_curve(codes, depths):
    """A two-parameter empirical family, NOT an SDK inverse formula.

    Robust fit suppresses invalid silhouette correspondences; code support
    is restricted to observed inliers and extrapolation is still forbidden.
    """
    codes, depths = np.asarray(codes, float), np.asarray(depths, float)
    keep = (codes > 0) & (codes < 1020) & (depths > .1) & (depths < 200) & np.isfinite(depths)
    codes, logz = codes[keep], np.log(depths[keep])
    if len(codes) < 100:
        raise ValueError("Too few samples for smooth empirical curve")
    def residual(parameters):
        a, b = parameters
        return a+np.log(np.expm1(codes*np.exp(b)))-logz
    fitted = least_squares(residual, [0., np.log(.01)], bounds=([-10., np.log(.0001)], [10., np.log(.05)]),
        loss="soft_l1", f_scale=.1)
    error = residual(fitted.x)
    inliers = np.abs(error) < max(.15, 3*float(np.median(np.abs(error))))
    if inliers.sum() < 100:
        raise ValueError("Smooth empirical curve lacks support")
    low, high = float(codes[inliers].min()), float(codes[inliers].max())
    knots = np.linspace(low, high, 32)
    a, b = fitted.x
    return dict(model="robust_log1p", parameters=fitted.x.tolist(), code_knots=knots.tolist(),
        log_depth_knots=(a+np.log(np.expm1(knots*np.exp(b)))).tolist(), sample_count=int(len(codes)),
        inlier_count=int(inliers.sum()), method="robust two-parameter log1p regression from RGB triangulation",
        provenance="inferred", physical_contract_verified=False, extrapolation=False)


def unproject_inferred(depth, k, valid, *, pixel_offset=0., stride=1):
    """Numerical projection only; caller carries empirical provenance."""
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    keep = valid[v, u] & np.isfinite(depth[v, u]) & (depth[v, u] > 0)
    uv = np.column_stack((u[keep], v[keep]))
    rays = np.column_stack((uv+pixel_offset, np.ones(len(uv)))) @ np.linalg.inv(k).T
    return rays*depth[v[keep], u[keep], None], uv


def align_depth(depth, valid, depth_pose, rgb_pose, k):
    """Depth-camera -> RGB raster with nearest-surface z-buffer, no hole fill."""
    if np.allclose(depth_pose, rgb_pose, atol=1e-12):
        return depth.copy(), valid.copy()
    local, _ = unproject_inferred(depth, k, valid)
    world = transform_points(local, depth_pose)
    uv, z = project(world, rgb_pose, k)
    h, w = depth.shape
    finite = np.isfinite(uv).all(1) & (z > .1) & (z <= 80)
    xy = np.rint(np.where(finite[:, None], uv, -1)).astype(np.int64)
    keep = finite & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    raster = np.full(h*w, np.inf, np.float32)
    np.minimum.at(raster, xy[keep, 1]*w+xy[keep, 0], z[keep].astype(np.float32))
    raster = raster.reshape(h, w)
    mask = np.isfinite(raster)
    raster[~mask] = 0
    return raster, mask


def sample_depth_curve(points, raw, depth_pose, k):
    uv, z = project(points, depth_pose, k)
    h, w = raw.shape
    finite = np.isfinite(uv).all(1) & (z > .1)
    xy = np.rint(np.where(finite[:, None], uv, -1)).astype(int)
    keep = finite & (xy[:, 0] >= 1) & (xy[:, 0] < w-1) & (xy[:, 1] >= 1) & (xy[:, 1] < h-1)
    x, y = xy[keep, 0], xy[keep, 1]
    # Depth discontinuities are not reliable RGB/depth correspondences.
    neighborhood = np.stack([raw[y+dy, x+dx] for dy, dx in ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0))]).astype(float)
    stable = np.ptp(neighborhood, axis=0) <= 8
    return raw[y[stable], x[stable]], z[keep][stable]


def validate_predictions(pairs, curve, poses, k, depth_loader, depth_poses=None):
    errors, summaries = [], []
    for a, b, ua, ub in pairs:
        codes = depth_loader(a)
        x = np.clip(np.rint(ua[:, 0]).astype(int), 0, codes.shape[1]-1)
        y = np.clip(np.rint(ua[:, 1]).astype(int), 0, codes.shape[0]-1)
        if depth_poses is None or np.allclose(depth_poses[a], poses[a], atol=1e-12):
            z, valid = decode_empirical(codes[y, x], curve)
        else:
            raster, mask = decode_empirical(codes, curve)
            raster, mask = align_depth(raster, mask, depth_poses[a], poses[a], k)
            z, valid = raster[y, x], mask[y, x]
        rays = np.column_stack((ua, np.ones(len(ua)))) @ np.linalg.inv(k).T
        points = transform_points(rays[valid]*z[valid, None], poses[a])
        target, target_z = project(points, poses[b], k)
        e = np.linalg.norm(target-ub[valid], axis=1)
        e = e[(target_z > 0) & np.isfinite(e)]
        errors.extend(e.tolist())
        summaries.append(dict(source_frame=a, target_frame=b, samples=len(e),
                              median_pixels=float(np.median(e)) if len(e) else None))
    med, p90 = (float(np.median(errors)), float(np.percentile(errors, 90))) if errors else (None, None)
    passed = len(errors) >= 100 and all(p["samples"] >= 10 for p in summaries) and med <= 3 and p90 <= 10
    return dict(status="passed" if passed else "failed", sample_count=len(errors), pairs=summaries,
                median_pixels=med, p90_pixels=p90, thresholds=dict(median_pixels=3, p90_pixels=10, minimum_samples=100))


def calibrate(rows, world_scale, k, frame_indices, feature_loader, depth_loader):
    source_k = np.asarray(k, float).copy()
    depth_loader = lru_cache(maxsize=len(frame_indices))(depth_loader)
    training = [i for i in frame_indices if i not in HOLDOUT]
    def pairs_for(indices):
        pairs = []
        for offset in (1, 2, 3, 4):
            for a, b in zip(indices, indices[offset:]):
                ua, ub = match_features(feature_loader(a), feature_loader(b))
                if len(ua) >= 30:
                    pairs.append((a, b, ua, ub))
        return pairs
    train_pairs = pairs_for(training)
    if len(train_pairs) < 3:
        raise ValueError("Too few static feature pairs; calibration is unobservable")
    candidates = []
    for unit in ("degrees", "radians"):
        for yaw_sign in (1, -1):
            for pitch_sign in (1, -1):
                fields = ("camera_raw", "depth_referenced_camera_raw") if all(r.get("depth_referenced_camera_raw") for r in rows) else ("camera_raw",)
                for field in fields:
                    poses = pose_track(rows, world_scale, yaw_sign=yaw_sign, pitch_sign=pitch_sign, angular_unit=unit, field=field)
                    errors = np.concatenate([epipolar_errors(ua, ub, poses[a], poses[b], k) for a, b, ua, ub in train_pairs])
                    candidates.append(dict(angular_unit=unit, yaw_sign=yaw_sign, pitch_sign=pitch_sign, field=field,
                                           median_epipolar_pixels=float(np.median(errors))))
    choice = min(candidates, key=lambda c: c["median_epipolar_pixels"])
    # SDK metadata is still unverified. Compare a bounded one-parameter focal
    # scale and tick binding on training RGB only; never mutate source bytes.
    convention = {key: choice[key] for key in ("angular_unit", "yaw_sign", "pitch_sign")}
    binding_candidates = []
    fields = ("camera_raw", "depth_referenced_camera_raw") if all(r.get("depth_referenced_camera_raw") for r in rows) else ("camera_raw",)
    for field in fields:
        for offset in (0, -1, 1, -2, 2):
            track = pose_track(rows, world_scale, field=field, frame_offset=offset, **convention)
            usable = [p for p in train_pairs if 0 <= p[0]+offset < len(rows) and 0 <= p[1]+offset < len(rows)]
            for scale in sorted(set([1.] + np.arange(.75, 1.351, .05).round(8).tolist())):
                intr = source_k.copy()
                intr[0, 0] *= scale; intr[1, 1] *= scale
                errors = np.concatenate([epipolar_errors(ua, ub, track[a], track[b], intr) for a, b, ua, ub in usable])
                binding_candidates.append(dict(**convention, field=field, frame_offset=offset, focal_scale=scale,
                    median_epipolar_pixels=float(np.median(errors))))
    binding = min(binding_candidates, key=lambda c: c["median_epipolar_pixels"])
    # A fine local focal search, still entirely frozen before holdout access.
    for scale in np.arange(max(.75, binding["focal_scale"]-.05), min(1.35, binding["focal_scale"]+.05)+.0001, .005):
        track = pose_track(rows, world_scale, field=binding["field"], frame_offset=binding["frame_offset"], **convention)
        intr = source_k.copy(); intr[0, 0] *= scale; intr[1, 1] *= scale
        offset = binding["frame_offset"]
        errors = np.concatenate([epipolar_errors(ua, ub, track[a], track[b], intr) for a, b, ua, ub in train_pairs
            if 0 <= a+offset < len(rows) and 0 <= b+offset < len(rows)])
        binding_candidates.append(dict(**convention, field=binding["field"], frame_offset=offset,
            focal_scale=float(scale), median_epipolar_pixels=float(np.median(errors))))
    choice = min(binding_candidates, key=lambda c: c["median_epipolar_pixels"])
    k = source_k.copy(); k[0, 0] *= choice["focal_scale"]; k[1, 1] *= choice["focal_scale"]
    poses = pose_track(rows, world_scale, **{key: choice[key] for key in ("angular_unit", "yaw_sign", "pitch_sign", "field", "frame_offset")})
    print(f"Training camera binding: {choice}", flush=True)
    triangulated = []
    for a, b, ua, ub in train_pairs:
        epi = epipolar_errors(ua, ub, poses[a], poses[b], k) <= 3
        ua, ub = ua[epi], ub[epi]
        points, keep = triangulate(ua, ub, poses[a], poses[b], k)
        ua, ub = ua[keep], ub[keep]
        triangulated.append((a, b, ua, ub, points))
    convention = {key: choice[key] for key in ("angular_unit", "yaw_sign", "pitch_sign")}
    modes = {"rgb_same_frame": poses}
    if all(row.get("depth_referenced_camera_raw") is not None for row in rows):
        modes["depth_referenced_snapshot"] = pose_track(rows, world_scale, field="depth_referenced_camera_raw", **convention)
    # Association selection uses an inner training split, never final holdouts.
    selection_frames = set(training[4::5])
    fitting = [r for r in triangulated if r[0] not in selection_frames and r[1] not in selection_frames]
    selecting = [(a, b, ua, ub) for a, b, ua, ub, _ in triangulated if a in selection_frames]
    if not fitting or not selecting:
        raise ValueError("Insufficient independent training split for depth association")
    depth_candidates = []
    def fit_for(records, depth_poses, smooth=False):
        codes, depths = [], []
        for a, b, _, _, points in records:
            for index in (a, b):
                cc, zz = sample_depth_curve(points, depth_loader(index), depth_poses[index], k)
                codes.extend(cc.tolist())
                depths.extend(zz.tolist())
        return fit_smooth_curve(codes, depths) if smooth else fit_curve(codes, depths)
    for mode, depth_poses in modes.items():
        for smooth in (False, True):
            provisional = fit_for(fitting, depth_poses, smooth)
            selection = validate_predictions(selecting, provisional, poses, k, depth_loader, depth_poses)
            depth_candidates.append(dict(mode=mode, smooth=smooth, selection=selection))
            print(f"Depth association {mode}, smooth={smooth}: median={selection['median_pixels']} p90={selection['p90_pixels']}", flush=True)
    depth_choice = min(depth_candidates, key=lambda c: c["selection"]["p90_pixels"] if c["selection"]["p90_pixels"] is not None else float("inf"))
    depth_poses = modes[depth_choice["mode"]]
    curve = fit_for(triangulated, depth_poses, depth_choice["smooth"])
    # Holdouts are first touched only after convention and curve are frozen.
    hold_pairs = []
    for h in HOLDOUT:
        if h >= len(rows):
            continue
        # Prefer a nearby pair with measurable parallax; pure turns cannot
        # independently validate metric depth. Selection does not use the curve.
        choices = []
        for neighbor in sorted(training, key=lambda i: abs(i-h))[:6]:
            ua, ub = match_features(feature_loader(h), feature_loader(neighbor))
            epi = epipolar_errors(ua, ub, poses[h], poses[neighbor], k) <= 3
            ua, ub = ua[epi], ub[epi]
            _, observable = triangulate(ua, ub, poses[h], poses[neighbor], k)
            choices.append((int(observable.sum()), neighbor, ua[observable], ub[observable]))
        _, neighbor, ua, ub = max(choices, key=lambda r: r[0])
        hold_pairs.append((h, neighbor, ua, ub))
    validation = validate_predictions(hold_pairs, curve, poses, k, depth_loader, depth_poses)
    report = dict(schema="refined.calibration/1", status=validation["status"], provenance="inferred",
        physical_contract_verified=False, camera_convention=choice, camera_candidates=candidates,
        camera_binding_candidates=binding_candidates, source_intrinsics=source_k.tolist(), intrinsics=k.tolist(),
        depth_curve=curve, holdout_frames=list(HOLDOUT), training_frames=training, validation=validation,
        depth_pose_mode=depth_choice["mode"], depth_pose_candidates=depth_candidates,
        selection_frames=sorted(selection_frames),
        limitations=["Absolute scale is conditional on telemetry world units.",
                     "Focal scale and camera tick binding are empirical; boundary ticks use the nearest available snapshot.",
                     "Cross-frame consistency is not proof of the physical depth inverse or SDK camera convention.",
                     "Depth values outside observed calibration support remain unknown."])
    return poses, curve, report
