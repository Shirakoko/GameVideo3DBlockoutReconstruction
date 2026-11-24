"""One-to-one instance detection, category/classification and persistent ID QA."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _validate(rows):
    seen = set()
    for row in rows:
        frame, identity = row["frame_index"], row["object_id"]
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 or not isinstance(identity, str) or not identity:
            raise ValueError("Semantic source frame and nonempty object identity required")
        if (frame, identity) in seen:
            raise ValueError("Duplicate object identity within a frame")
        seen.add((frame, identity))
        box = np.asarray(row["bbox_xyxy"], dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or (box[:2] < 0).any() or (box[2:] <= box[:2]).any():
            raise ValueError("Finite valid exclusive-upper bbox_xyxy required")
        if "raster_size" in row:
            size = row["raster_size"]
            if len(size) != 2 or any(isinstance(s, bool) or not isinstance(s, int) or s < 1 for s in size) or (box[2:] > size).any():
                raise ValueError("Semantic bbox must fit its explicit native raster")
        if not isinstance(row.get("category"), str) or not row["category"]:
            raise ValueError("Explicit category required")
        if row.get("is_static") is not None and not isinstance(row["is_static"], bool):
            raise ValueError("Static/dynamic label must be boolean or null")


def box_iou(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    size = np.maximum(0., np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(size))
    return intersection/(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection)


def _matches(pred, gt, threshold, *, category):
    if not pred or not gt:
        return []
    iou = np.array([[box_iou(p["bbox_xyxy"], g["bbox_xyxy"]) for g in gt] for p in pred])
    allowed = iou >= threshold
    if category:
        allowed &= np.array([[p["category"] == g["category"] for g in gt] for p in pred])
    # Maximum cardinality first, IoU second. A forbidden edge must not steal
    # the only valid partner (a common Hungarian + post-filtering bug).
    weights = np.where(allowed, min(len(pred), len(gt))+1+iou, 0.)
    pids, gids = linear_sum_assignment(weights, maximize=True)
    return [(int(p), int(g), float(iou[p, g])) for p, g in zip(pids, gids) if allowed[p, g]]


def semantic_metrics(predicted, ground_truth, *, evaluated_frames, iou_threshold=0.5, max_id_gap_frames=None):
    """Annotations must be exhaustive on evaluated_frames (including empty frames).

    Detection P/R is category-aware. Category and motion accuracy use separate
    class-agnostic localization matches; wrong classes do not disappear from
    their denominator. ID switches retain the last match across occlusions,
    optionally bounded by an explicit source-frame gap.
    """
    _validate(predicted); _validate(ground_truth)
    if isinstance(iou_threshold, bool) or not np.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0,1]")
    frames = sorted(set(evaluated_frames))
    if not frames or len(frames) != len(evaluated_frames) or any(isinstance(f, bool) or not isinstance(f, int) or f < 0 for f in frames):
        raise ValueError("Explicit unique evaluated source frames required")
    if max_id_gap_frames is not None and (isinstance(max_id_gap_frames, bool) or not isinstance(max_id_gap_frames, int) or max_id_gap_frames < 1):
        raise ValueError("ID gap must be null or a positive source-frame interval")
    if any(r["frame_index"] not in frames for r in ground_truth):
        raise ValueError("GT object outside exhaustive annotation frame set")
    pred = [r for r in predicted if r["frame_index"] in frames]
    matched, localized, categories_correct, static_correct, static_count, switches, fragments = 0, 0, 0, 0, 0, 0, 0
    previous, missed, records, confusion = {}, set(), [], {}
    for frame in frames:
        pp, gg = [r for r in pred if r["frame_index"] == frame], [r for r in ground_truth if r["frame_index"] == frame]
        matched += len(_matches(pp, gg, iou_threshold, category=True))
        pairs = _matches(pp, gg, iou_threshold, category=False)
        localized += len(pairs)
        observed = set()
        for p, g, iou in pairs:
            prediction, truth = pp[p], gg[g]
            identity = truth["object_id"]
            observed.add(identity)
            categories_correct += int(prediction["category"] == truth["category"])
            confusion.setdefault(truth["category"], {}).setdefault(prediction["category"], 0)
            confusion[truth["category"]][prediction["category"]] += 1
            if truth.get("is_static") is not None:
                static_count += 1
                # Unknown prediction counts as incorrect, not as excluded.
                static_correct += int(prediction.get("is_static") == truth["is_static"])
            last = previous.get(identity)
            eligible = last is not None and (max_id_gap_frames is None or frame-last[1] <= max_id_gap_frames)
            switched = eligible and last[0] != prediction["object_id"]
            switches += int(switched)
            fragments += int(identity in missed and last is not None)
            previous[identity] = (prediction["object_id"], frame)
            missed.discard(identity)
            records.append(dict(frame_index=frame, predicted_id=prediction["object_id"], ground_truth_id=identity, iou=iou, id_switch=bool(switched)))
        missed.update(r["object_id"] for r in gg if r["object_id"] not in observed and r["object_id"] in previous)
    precision = matched/len(pred) if pred else None
    recall = matched/len(ground_truth) if ground_truth else None
    fscore = 2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else (0. if ground_truth else None)
    return dict(values=dict(detection_precision=precision, detection_recall=recall, detection_f1=fscore,
        detection_true_positives=matched, detection_false_positives=len(pred)-matched, detection_false_negatives=len(ground_truth)-matched,
        category_accuracy=categories_correct/localized if localized else None, category_match_count=localized,
        static_dynamic_accuracy=static_correct/static_count if static_count else None, static_dynamic_match_count=static_count,
        object_tracking_id_switches=switches if ground_truth else None, tracking_fragmentations=fragments if ground_truth else None,
        tracking_match_coverage=localized/len(ground_truth) if ground_truth else None),
        details=dict(evaluated_frames=frames, ignored_prediction_count=len(predicted)-len(pred), iou_threshold=iou_threshold,
                     max_id_gap_frames=max_id_gap_frames, confusion_matrix=confusion, matches=records),
        definition="Exhaustive frame annotations, maximum-cardinality one-to-one bbox IoU; category-aware detection, class-agnostic accuracy and ID association")

