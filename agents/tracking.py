"""Deterministic scene-scoped persistent IDs and conservative association."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np
from scipy.optimize import linear_sum_assignment

from agents.object_detection import confidence, unknown


def appearance(rgb, mask):
    pixels = rgb[mask].astype(float)/255
    return np.concatenate([np.histogram(pixels[:, channel], bins=8, range=(0, 1))[0]/len(pixels) for channel in range(3)])


def bbox_iou(a, b):
    low = np.maximum(a[:2], b[:2])
    high = np.minimum(a[2:], b[2:])
    overlap = np.prod(np.maximum(high-low, 0))
    return float(overlap/(np.prod(np.subtract(a[2:], a[:2]))+np.prod(np.subtract(b[2:], b[:2]))-overlap))


@dataclass
class Observation:
    frame_index: int
    timestamp_seconds: float
    source_observation_id: str
    detection: object
    measurement: dict | None
    geometry_confidence: dict
    appearance: np.ndarray
    mask_path: str
    camera_id: str = "camera_main"


@dataclass
class Track:
    object_id: str
    category: str
    hint: str | None
    observations: list = field(default_factory=list)
    association_method: str = "first_observation"


class PersistentTracker:
    def __init__(self, scene_id, max_gap_seconds=6., max_speed_mps=8., ambiguity_margin=0.08):
        if not scene_id or not np.isfinite([max_gap_seconds, max_speed_mps, ambiguity_margin]).all() or min(max_gap_seconds, max_speed_mps, ambiguity_margin) <= 0:
            raise ValueError("Tracker requires scene ID and positive finite gates")
        self.scene_id = scene_id
        self.max_gap = max_gap_seconds
        self.max_speed = max_speed_mps
        self.margin = ambiguity_margin
        self.tracks = []
        self.last_time = -1.

    def _cost(self, track, obs):
        previous = track.observations[-1]
        dt = obs.timestamp_seconds-previous.timestamp_seconds
        if track.category != obs.detection.category or dt <= 0 or dt > self.max_gap:
            return 1e6
        if track.hint is not None or obs.detection.identity_hint is not None:
            return 1e6  # hints are matched explicitly, never merged with inferred IDs
        color = np.mean(np.abs(previous.appearance-obs.appearance))*4
        a, b = previous.measurement, obs.measurement
        if a is not None and b is not None:
            if color > 0.65:
                return 1e6
            history = independent_observations(track)
            prediction = a["position"].copy()
            if len(history) >= 2 and history[-2].measurement is not None:
                older = history[-2]
                velocity = (a["position"]-older.measurement["position"])/(previous.timestamp_seconds-older.timestamp_seconds)
                if np.linalg.norm(velocity) <= self.max_speed:
                    prediction += velocity*dt
            distance = np.linalg.norm(prediction-b["position"])
            uncertainty = max(0.35, 0.25*np.linalg.norm(a["dimensions"]))
            gate = uncertainty + self.max_speed*dt
            if distance > gate:
                return 1e6
            size = np.mean(np.abs(np.log(np.maximum(a["dimensions"], 0.1)/np.maximum(b["dimensions"], 0.1))))
            if size > 1.5:
                return 1e6
            return min(0.8, distance/gate)*0.65 + color*0.25 + min(1., size)*0.1
        # Image motion alone cannot separate camera motion from object motion.
        # Cross-camera matching requires world geometry or reviewed identity.
        if previous.camera_id != obs.camera_id:
            return 1e6
        overlap = bbox_iou(previous.detection.bbox, obs.detection.bbox)
        if overlap < 0.15 or color > 0.3:
            return 1e6
        return (1-overlap)*0.7 + color*0.3

    def update(self, observations, timestamp_seconds):
        if timestamp_seconds <= self.last_time:
            raise ValueError("Tracker batches must have strictly increasing timestamps")
        self.last_time = timestamp_seconds
        hints = [o.detection.identity_hint for o in observations if o.detection.identity_hint]
        if len(hints) != len(set(hints)):
            raise ValueError("One identity_hint cannot occur twice in the same batch")
        keys = [o.detection.key for o in observations]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate detection key within one frame")
        matched, used = {}, set()
        for index, obs in enumerate(observations):
            hint = obs.detection.identity_hint
            if hint:
                existing = next((t for t in self.tracks if t.hint == hint), None)
                if existing:
                    if existing.category != obs.detection.category:
                        raise ValueError("Persistent identity hint changed category")
                    matched[index] = existing
                    used.add(existing.object_id)
                    existing.association_method = "assisted_identity_hint"
        active = [t for t in self.tracks if t.object_id not in used and timestamp_seconds-t.observations[-1].timestamp_seconds <= self.max_gap]
        remaining = [i for i in range(len(observations)) if i not in matched and not observations[i].detection.identity_hint]
        if active and remaining:
            costs = np.array([[self._cost(t, observations[i]) for i in remaining] for t in active])
            # Dummy columns allow abstention instead of forcing a bad match.
            rows, columns = linear_sum_assignment(np.concatenate([costs, np.full((len(active), len(active)), 0.92)], axis=1))
            for row, col in zip(rows, columns):
                if col >= len(remaining) or costs[row, col] >= 0.92:
                    continue
                competitors = np.delete(costs[:, col], row)
                alternatives = np.delete(costs[row], col)
                if (len(competitors) and competitors.min()-costs[row, col] < self.margin) or (len(alternatives) and alternatives.min()-costs[row, col] < self.margin):
                    continue  # ambiguous crossings: new ID, never fabricate certainty
                t = active[row]
                matched[remaining[col]] = t
                t.association_method = "world_motion_appearance_assignment" if observations[remaining[col]].measurement is not None else "image_iou_appearance_assignment"
        for index, obs in enumerate(observations):
            t = matched.get(index)
            if t is None:
                identity = obs.detection.identity_hint or f"{obs.camera_id}:{obs.frame_index}:{obs.detection.key}"
                oid = "obj_" + hashlib.sha256(f"{self.scene_id}:{identity}".encode()).hexdigest()[:16]
                if any(t.object_id == oid for t in self.tracks):
                    raise ValueError("Persistent ID collision or conflicting observation identity")
                t = Track(oid, obs.detection.category, obs.detection.identity_hint)
                self.tracks.append(t)
            t.observations.append(obs)
        return self.tracks


def independent_observations(track):
    seen, result = set(), []
    for obs in track.observations:
        if obs.source_observation_id not in seen:
            result.append(obs)
            seen.add(obs.source_observation_id)
    return result


def classify_motion(track, static_tolerance_m=0.30, dynamic_displacement_m=0.55):
    independent = independent_observations(track)
    measured = [o for o in independent if o.measurement is not None]
    hints = {o.detection.static_hint for o in independent if o.detection.static_hint is not None}
    if len(hints) > 1:
        raise ValueError("Conflicting static/dynamic review hints")
    if len(measured) >= 3 and measured[-1].timestamp_seconds-measured[0].timestamp_seconds >= 1.:
        p = np.array([o.measurement["position"] for o in measured])
        spread = float(np.max(np.linalg.norm(p-p[0], axis=1)))
        # Wide static ground changes visible AABB as the camera moves. Test
        # world surface support instead of mistaking that for object motion.
        from scipy.spatial import cKDTree
        reference = cKDTree(measured[0].measurement["points"])
        support_error = max(float(np.quantile(reference.query(o.measurement["points"])[0], 0.80)) for o in measured[1:])
        if spread <= static_tolerance_m or support_error <= 0.12:
            status = True
        elif spread >= dynamic_displacement_m:
            status = False
        else:
            status = None
        if status is not None:
            if hints and status != next(iter(hints)):
                return None, unknown("World displacement disagrees with review hint; inspect mask/viewpoint bias")
            return status, confidence(0.8, "inferred", "independent_world_displacement_surface_support",
                f"{len(measured)} independent observations; max displacement {spread:.4f}m; max 80% surface residual {support_error:.4f}m; static <=0.30m displacement or <=0.12m surface residual, dynamic >=0.55m; partial visibility can bias classification")
    if hints:
        return next(iter(hints)), confidence(0.65, "inferred", "assisted_rgb_motion_review", "Explicit RGB review hint, not measured world velocity")
    return None, unknown("Insufficient independent geometry/time span or ambiguous motion; category is not proof of staticness")
