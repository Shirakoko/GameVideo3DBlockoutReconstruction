"""Observed-surface metrics in metres, using bounded KD-tree nearest neighbours."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def points_array(points):
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        points = points.reshape(0, 3)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Finite Nx3 metric world points required")
    return points


def geometry_metrics(predicted, ground_truth, *, thresholds_m=(0.05, 0.1), max_points=200000, seed=0):
    """CD = (mean(pred->GT) + mean(GT->pred))/2, NOT squared or a sum.

    Coverage is the fraction of observed GT samples within each tolerance.
    Empty predictions have zero recall/F-score, undefined precision and CD.
    Empty GT is not an evaluable scene. Sampling is deterministic and disclosed.
    """
    pred, gt = points_array(predicted), points_array(ground_truth)
    if not len(gt):
        raise ValueError("Ground Truth point cloud is empty")
    if isinstance(max_points, bool) or not isinstance(max_points, int) or max_points < 1:
        raise ValueError("max_points must be a positive integer")
    if any(isinstance(value, (bool, np.bool_)) for value in np.asarray(thresholds_m, dtype=object).flat):
        raise ValueError("Geometry metre thresholds cannot be boolean")
    thresholds = np.asarray(thresholds_m, dtype=float)
    if thresholds.ndim != 1 or not len(thresholds) or not np.isfinite(thresholds).all() or (thresholds <= 0).any():
        raise ValueError("Nonempty finite positive metre thresholds required")
    if len(np.unique(thresholds)) != len(thresholds):
        raise ValueError("Duplicate geometry thresholds")
    counts = dict(predicted_points=len(pred), ground_truth_points=len(gt))
    rng = np.random.default_rng(seed)
    pred = pred[rng.choice(len(pred), max_points, replace=False)] if len(pred) > max_points else pred
    gt = gt[rng.choice(len(gt), max_points, replace=False)] if len(gt) > max_points else gt
    counts.update(evaluated_predicted_points=len(pred), evaluated_ground_truth_points=len(gt), sampling_seed=seed)
    if len(pred):
        forward = cKDTree(gt).query(pred, workers=1)[0]
        backward = cKDTree(pred).query(gt, workers=1)[0]
    else:
        forward, backward = np.array([]), np.full(len(gt), np.inf)
    values = dict(chamfer_distance_m=float((forward.mean()+backward.mean())/2) if len(pred) else None,
                  chamfer_distance_squared_m2=float((np.mean(forward**2)+np.mean(backward**2))/2) if len(pred) else None)
    for threshold in thresholds:
        precision = float(np.mean(forward <= threshold)) if len(pred) else None
        recall = float(np.mean(backward <= threshold))
        score = 2*precision*recall/(precision+recall) if precision is not None and precision+recall else 0.
        suffix = f"at_{threshold:g}m"
        values.update({f"point_precision_{suffix}": precision, f"point_cloud_coverage_{suffix}": recall,
                       f"f_score_{suffix}": float(score)})
    return dict(values=values, details=counts, definition="Symmetric mean Euclidean nearest-sample distance; observed surfaces only, not hidden volume")

