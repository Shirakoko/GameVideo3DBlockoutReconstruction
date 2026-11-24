"""Draft 2020-12 validation plus references, units, and time invariants."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator, FormatChecker

from reconstruction.coordinate_transform import validate_rigid

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "config/scene_schema.json"
REFINED_SCHEMA_PATH = SCHEMA_PATH.with_name("scene_schema_refined.json")


class SceneValidationError(ValueError):
    pass


def validate_scene(scene: dict) -> dict:
    """Reject unsupported versions, NaN, dangling IDs, inconsistent tracks.

    No implicit migration. Validation checks a contract, not factual accuracy.
    Paths are relative to the scene JSON, and are not opened by this validator.
    """
    issues = []
    try:
        json.dumps(scene, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise SceneValidationError(f"Non-JSON/nonfinite value: {error}") from error
    refined = scene.get("schema_version") == "1.1.0"
    schema = json.loads((REFINED_SCHEMA_PATH if refined else SCHEMA_PATH).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(scene), key=lambda e: str(list(e.absolute_path)))
    if errors:
        raise SceneValidationError("\n".join(f"/{'/'.join(map(str, e.absolute_path))}: {e.message}" for e in errors[:30]))

    def check(condition, message):
        if not condition:
            issues.append(message)

    def unique(rows, key, label):
        values = [r[key] for r in rows]
        check(len(values) == len(set(values)), f"{label}: duplicate {key}")

    def confidence(c, observed):
        check(observed == (c["provenance"] != "unobserved"), "Observed value contradicts confidence provenance")
        if not observed:
            check(c["score"] == 0, "Unobserved confidence must be zero")

    def transform(t):
        if t is not None:
            check(np.isclose(np.linalg.norm(t["rotation"]), 1, atol=1e-6), "Quaternion xyzw must be unit length")

    cameras = scene["cameras"]
    objects = scene["objects"]
    unique(cameras, "camera_id", "cameras")
    unique(objects, "object_id", "objects")
    camera_ids = {c["camera_id"] for c in cameras}
    by_id = {o["object_id"]: o for o in objects}
    timeline = scene["timeline"]
    # jsonschema's RFC3339 format checker is optional on some installations.
    # Always enforce an explicit timezone using stdlib too.
    try:
        date = datetime.fromisoformat(scene["created_at"])
        check(date.utcoffset() is not None and "T" in scene["created_at"], "created_at requires ISO/RFC3339 timestamp with timezone")
    except ValueError:
        issues.append("Invalid created_at timestamp")
    check(timeline["end_seconds"] >= timeline["start_seconds"], "Timeline end precedes start")
    refs = {}
    for f in timeline["frames"]:
        key = (f["camera_id"], f["frame_index"])
        check(key not in refs, "Duplicate timeline camera/frame")
        check(f["camera_id"] in camera_ids, "Unknown timeline camera")
        check(timeline["start_seconds"] <= f["timestamp_seconds"] <= timeline["end_seconds"], "Frame outside timeline")
        refs[key] = f

    def timed(rows, camera_id, label):
        previous = (-1, -1.)
        for r in rows:
            ci = r.get("camera_id", camera_id)
            source = refs.get((ci, r["frame_index"]))
            check(source is not None, f"{label}: unknown camera/frame")
            if source:
                check(abs(source["timestamp_seconds"] - r["timestamp_seconds"]) <= 1e-9, f"{label}: timestamp mismatch")
            check(r["frame_index"] > previous[0] and r["timestamp_seconds"] > previous[1], f"{label}: samples must be strictly increasing")
            previous = (r["frame_index"], r["timestamp_seconds"])

    for c in cameras:
        timed([f for f in timeline["frames"] if f["camera_id"] == c["camera_id"]], c["camera_id"], "timeline")
        timed(c["track"], c["camera_id"], "camera track")
        check({r["frame_index"] for r in c["track"]} == {i for ci, i in refs if ci == c["camera_id"]}, "Camera track must cover timeline samples")
        for r in c["track"]:
            confidence(r["confidence"], r["world_from_camera"] is not None)
            try:
                if r["world_from_camera"] is not None:
                    validate_rigid(r["world_from_camera"])
                if r["intrinsics"] is not None:
                    k = np.asarray(r["intrinsics"])
                    check(k[0, 0] > 0 and k[1, 1] > 0 and np.allclose(k[2], [0, 0, 1]) and k[1, 0] == 0, "Invalid camera intrinsics")
            except ValueError as error:
                issues.append(f"Invalid camera calibration: {error}")

    for o in objects:
        transform(o["transform"])
        known = o["transform"] is not None
        check(known == (o["dimensions"] is not None), "Object transform/dimensions must both be known or null")
        check(known == (o["geometry_source"]["kind"] != "unobserved"), "Geometry source contradicts transform")
        check(known == (o["geometry_source"]["bounds_type"] != "unobserved"), "Bounds type contradicts transform")
        check(known == (o["geometry_source"]["point_cloud_path"] is not None), "Point cloud source contradicts transform")
        confidence(o["geometry_source"]["confidence"], known)
        confidence(o["classification_confidence"], o["is_static"] is not None)
        if not known:
            check(not o["motion_track"], "Unobserved geometry cannot have world motion")
        if o["is_static"] is True:
            check(not o["motion_track"], "Static objects cannot have a motion track")
        if o["is_static"] is False and known:
            check(bool(o["motion_track"]), "Known dynamic object requires a motion track")
        parent = o["parent_id"]
        check(parent is None or parent in by_id, "Dangling parent_id")
        seen = {o["object_id"]}
        while parent in by_id:
            if parent in seen:
                issues.append("Parent hierarchy cycle")
                break
            seen.add(parent)
            parent = by_id[parent]["parent_id"]
        # Multiple cameras can observe one object at the same timestamp.
        for ci in camera_ids:
            timed([r for r in o["observations"] if r["camera_id"] == ci], ci, "observations")
            timed([r for r in o["visibility"] if r["camera_id"] == ci], ci, "visibility")
        for r in o["observations"] + o["visibility"]:
            check(r["camera_id"] in camera_ids, "Unknown object observation/visibility camera")
        observed_indices = {r["frame_index"] for r in o["observations"]}
        check(set(o["source_frames"]) == observed_indices, "source_frames must equal observation frame indices")
        for r in o["observations"]:
            x0, y0, x1, y1 = r["bbox_xyxy"]
            w, h = r["raster_size"]
            check(0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, "Invalid native-raster bbox")
            confidence(r["geometry_confidence"], r["position"] is not None)
            check((r["position"] is None) == (r["dimensions"] is None), "Observation position/dimensions inconsistent")
            check((r["position"] is not None) == (r["point_count"] > 0), "Observation point_count contradicts geometry")
            if r["mask"]["kind"] == "bbox_proxy":
                check(r["position"] is None, "BBox proxy cannot certify object geometry")
        visible = {(r["camera_id"], r["frame_index"]): r["state"] for r in o["visibility"]}
        for r in o["observations"]:
            check(visible.get((r["camera_id"], r["frame_index"])) == "visible", "Observed object must have visible state")
        obs_by_frame = {r["frame_index"]: r for r in o["observations"]}
        previous = -1.
        for r in o["motion_track"]:
            check(r["timestamp_seconds"] > previous, "Motion timestamps must strictly increase")
            previous = r["timestamp_seconds"]
            obs = obs_by_frame.get(r["frame_index"])
            check(obs is not None and obs["position"] is not None, "Motion sample requires an observed world position")
            if obs:
                check(abs(obs["timestamp_seconds"] - r["timestamp_seconds"]) < 1e-9, "Motion timestamp mismatch")
            confidence(r["confidence"], True)
            transform(r["transform"])

    for r in scene["relationships"]:
        check(r["subject_id"] in by_id and r["object_id"] in by_id and r["subject_id"] != r["object_id"], "Invalid relationship IDs")
        for oid in (r["subject_id"], r["object_id"]):
            if oid in by_id:
                check(any(o["frame_index"] == r["frame_index"] and abs(o["timestamp_seconds"]-r["timestamp_seconds"]) < 1e-9 and o["position"] is not None for o in by_id[oid]["observations"]), "Relationship requires simultaneous geometric observations")
    meta = scene["reconstruction_metadata"]
    inferred = refined and meta["reconstruction_mode"] == "inferred"
    if inferred:
        check(meta["geometry_contract_verified"] is False, "Inferred reconstruction must not claim a verified physical contract")
        check(meta["status"] == "complete_inferred", "Inferred scene requires complete_inferred status")
        check(bool(objects) and all(o["transform"] is not None for o in objects), "Inferred scene requires nonempty known estimates")
        for obj in objects:
            check(obj["geometry_source"]["confidence"]["provenance"] == "inferred", "Empirical geometry must be labelled inferred")
        for camera in cameras:
            check(all(r["confidence"]["provenance"] == "inferred" for r in camera["track"]), "Empirical Camera Pose must be labelled inferred")
        heuristic = any(o["geometry_source"]["kind"] == "heuristic_voxel" for o in objects)
        check(heuristic == ("heuristic_proxy" in meta), "Heuristic geometry requires its distinct proxy receipt")
        if heuristic:
            check(all(o["geometry_source"]["kind"] == "heuristic_voxel" and o["geometry_source"]["bounds_type"] == "inferred_proxy_aabb" for o in objects), "Proxy geometry must not masquerade as observed bounds")
    if not meta["geometry_contract_verified"] and not inferred:
        check(not any(o["transform"] is not None for o in objects), "Unverified geometry contract cannot emit world transforms")
        check(not any(r["world_from_camera"] is not None for c in cameras for r in c["track"]), "Unverified geometry contract cannot emit camera transforms")
    if meta["status"] == "complete":
        check(meta["geometry_contract_verified"] and bool(objects) and all(o["transform"] is not None for o in objects), "Complete status requires verified, nonempty geometry")
    if issues:
        raise SceneValidationError("\n".join(issues[:30]))
    return {"valid": True, "schema_version": scene["schema_version"], "objects": len(objects), "cameras": len(cameras), "timeline_samples": len(refs)}
