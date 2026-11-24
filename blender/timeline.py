"""Deterministic sampling in source seconds; no QPC or row-number guesses."""
from __future__ import annotations

from bisect import bisect_right
from fractions import Fraction
import math

import numpy as np

from reconstruction.coordinate_transform import validate_rigid


def fps_fraction(value):
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError("Explicit positive finite FPS required")
    return Fraction(str(value)).limit_denominator(1_000_000)


def slerp(a, b, alpha):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    dot = min(1., max(-1., dot))
    if dot > 0.9995:
        result = a+(b-a)*alpha
        return result/np.linalg.norm(result)
    angle = math.acos(dot)
    return (math.sin((1-alpha)*angle)*a+math.sin(alpha*angle)*b)/math.sin(angle)


def matrix_quaternion(matrix):
    """Proper rotation matrix -> unit xyzw, including 180-degree rotations."""
    r = validate_rigid(matrix)[:3, :3]
    # Eigenvector avoids division by a near-zero trace at half turns.
    xx, xy, xz = r[0]
    yx, yy, yz = r[1]
    zx, zy, zz = r[2]
    k = np.array([[xx-yy-zz, xy+yx, xz+zx, zy-yz],
                  [xy+yx, yy-xx-zz, yz+zy, xz-zx],
                  [xz+zx, yz+zy, zz-xx-yy, yx-xy],
                  [zy-yz, xz-zx, yx-xy, xx+yy+zz]])/3
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1]
    return q if q[3] >= 0 else -q


def bracket(rows, time, *, hold=False):
    times = [r["timestamp_seconds"] for r in rows]
    if not rows or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Nonempty strictly increasing track required")
    if not hold and not times[0]-1e-8 <= time <= times[-1]+1e-8:
        raise ValueError(f"Camera track does not cover source time {time}; extrapolation refused")
    i = max(0, min(len(rows)-1, bisect_right(times, time)-1))
    j = min(i+1, len(rows)-1)
    alpha = min(1., max(0., (time-times[i])/(times[j]-times[i]))) if i != j else 0.
    return rows[i], rows[j], alpha


def sample_transform(rows, time):
    a, b, alpha = bracket(rows, time, hold=True)
    a, b = a["transform"], b["transform"]
    return dict(position=((1-alpha)*np.array(a["position"])+alpha*np.array(b["position"])).tolist(),
                rotation=slerp(a["rotation"], b["rotation"], alpha).tolist(),
                scale=((1-alpha)*np.array(a["scale"])+alpha*np.array(b["scale"])).tolist())


def sample_camera(rows, time):
    a, b, alpha = bracket(rows, time)
    ta, tb = np.asarray(a["world_from_camera"]), np.asarray(b["world_from_camera"])
    return dict(position=((1-alpha)*ta[:3, 3]+alpha*tb[:3, 3]).tolist(),
                optical_rotation=slerp(matrix_quaternion(ta), matrix_quaternion(tb), alpha).tolist(),
                intrinsics=((1-alpha)*np.asarray(a["intrinsics"])+alpha*np.asarray(b["intrinsics"])).tolist())


def frame_mapping(timeline, video, camera_id):
    fps = fps_fraction(timeline["fps"])
    if abs(float(fps)-float(Fraction(video["fps"]))) > 1e-8:
        raise ValueError(f"Video FPS {video['fps']} != Scene FPS {timeline['fps']}; no resampling")
    source = [r for r in timeline["frames"] if r["camera_id"] == camera_id]
    first = source[0]
    start = timeline["start_seconds"]
    origin = first["frame_index"]-round((first["timestamp_seconds"]-start)*float(fps))
    ticks = {r["frame_index"]: r["original_tick"] for r in source}
    for row in source:
        ordinal = (row["timestamp_seconds"]-start)*float(fps)
        if abs(ordinal-round(ordinal)) > 1e-6 or row["frame_index"] != origin+round(ordinal):
            raise ValueError("Scene Frame Index/time mapping is not a CFR source lattice")
    last = start+(video["frame_count"]-1)/float(fps)
    if last > timeline["end_seconds"]+1e-8:
        raise ValueError("Scene timeline does not cover all original video frames")
    # Reject accidental truncation of a longer scene, not just video overrun.
    if abs(timeline["end_seconds"]-last) > 1e-8:
        raise ValueError("Original video is shorter than the declared Scene timeline (end_seconds is last frame PTS)")
    return [dict(blender_frame=i+1, video_frame_index=video["start_frame"]+i,
                 source_frame_index=origin+i, timestamp_seconds=start+i/float(fps),
                 video_pts_seconds=video["first_pts_seconds"]+i/float(fps),
                 original_tick=ticks.get(origin+i)) for i in range(video["frame_count"])]
