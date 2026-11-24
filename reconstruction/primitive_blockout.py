"""Primitive blockout from original video observations, not voxel completion.

Back-projects cached RGB / depth codes / labels with the frozen camera track,
fits editable boxes / cylinders / spheres to those points, and writes a new
output directory. Existing voxel and clay-from-voxel results stay untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree

from agents.object_detection import confidence
from blender.pipeline import ROOT, blender_path, json_write, run_pipeline, sha256
from reconstruction.heuristic_proxy import AUTHORIZATION, proxy_lut
from reconstruction.refine import refinement_lock, render_previews, verified_frame_cache
from reconstruction.refined_calibration import HOLDOUT
from reconstruction.refined_segmentation import CATEGORIES
from scene_schema import validate_scene

BOX_CATEGORIES = {"environment", "rock", "barrier", "building", "wall", "road"}
VERTICAL_CATEGORIES = {"tree", "grass", "shrub"}
PERSON = CATEGORIES.index("person")
GROUND = CATEGORIES.index("ground")


def merge_boxes(cells):
    """Greedy solid rectangles. Every occupied cell is in exactly one box."""
    cells = np.unique(np.asarray(cells, np.int64), axis=0)
    if len(cells) == 0:
        return []
    lines = {}
    for x, y, z in cells.tolist():
        lines.setdefault((int(y), int(z)), []).append(int(x))
    runs = {}
    for (y, z), xs in lines.items():
        xs = sorted(set(xs))
        start = prev = xs[0]
        for x in xs[1:]:
            if x == prev + 1:
                prev = x
            else:
                runs.setdefault((start, prev, z), []).append(y)
                start = prev = x
        runs.setdefault((start, prev, z), []).append(y)
    rects = {}
    for (x0, x1, z), ys in runs.items():
        ys = sorted(set(ys))
        start = prev = ys[0]
        for y in ys[1:]:
            if y == prev + 1:
                prev = y
            else:
                rects.setdefault((x0, x1, start, prev), []).append(z)
                start = prev = y
        rects.setdefault((x0, x1, start, prev), []).append(z)
    boxes = []
    for (x0, x1, y0, y1), zs in rects.items():
        zs = sorted(set(zs))
        start = prev = zs[0]
        for z in zs[1:]:
            if z == prev + 1:
                prev = z
            else:
                boxes.append((x0, x1, y0, y1, start, prev))
                start = prev = z
        boxes.append((x0, x1, y0, y1, start, prev))
    return boxes


def box_bounds(box, size, factor):
    x0, x1, y0, y1, z0, z1 = box
    low = np.array([x0, y0, z0], float) * factor * size
    high = (np.array([x1, y1, z1], float) + 1) * factor * size
    return low, high


def camera_near_aabb(low, high, poses, clearance=1.2):
    if len(poses) == 0:
        return False
    points = np.asarray(poses, float)[:, :3, 3]
    closest = np.minimum(np.maximum(points, low), high)
    return bool(np.any(np.linalg.norm(points - closest, axis=1) < clearance))


def boxes_from_cells(cells, size, poses, factor):
    cells = np.unique(np.asarray(cells, np.int64), axis=0)
    if len(cells) == 0:
        return []
    if factor <= 1:
        return [box_bounds(box, size, 1) for box in merge_boxes(cells)]
    coarse = np.unique(np.floor(cells / factor).astype(np.int64), axis=0)
    safe, unsafe = [], []
    for box in merge_boxes(coarse):
        low, high = box_bounds(box, size, factor)
        if camera_near_aabb(low, high, poses):
            unsafe.append(box)
        else:
            safe.append((low, high))
    if unsafe:
        quant = np.floor(cells / factor).astype(np.int64)
        keep = np.zeros(len(cells), dtype=bool)
        for x0, x1, y0, y1, z0, z1 in unsafe:
            keep |= ((quant[:, 0] >= x0) & (quant[:, 0] <= x1) & (quant[:, 1] >= y0)
                     & (quant[:, 1] <= y1) & (quant[:, 2] >= z0) & (quant[:, 2] <= z1))
        safe.extend(boxes_from_cells(cells[keep], size, poses, max(1, factor // 2)))
    return safe


def _spec(name, category, primitive, position, dimensions, parent=None):
    dimensions = [float(v) for v in dimensions]
    if min(dimensions) <= 1e-4:
        return None
    return dict(name=name, category=category, primitive_type=primitive,
                position=[float(v) for v in position], dimensions=dimensions, parent=parent)


def contains_camera(spec, poses, margin=0.05):
    center = np.asarray(spec["position"], float)
    dimensions = np.asarray(spec["dimensions"], float)
    points = np.asarray(poses, float)[:, :3, 3]
    kind = spec["primitive_type"]
    if kind in {"box", "plane"}:
        low, high = center - dimensions / 2, center + dimensions / 2
        return bool(np.any(np.all(points > low + margin, axis=1) & np.all(points < high - margin, axis=1)))
    if kind == "sphere":
        return bool(np.any(np.linalg.norm(points - center, axis=1) < dimensions[0] / 2 - margin))
    if kind == "cylinder":
        radial = np.linalg.norm(points[:, :2] - center[:2], axis=1)
        vertical = np.abs(points[:, 2] - center[2]) < dimensions[2] / 2 - margin
        return bool(np.any((radial < dimensions[0] / 2 - margin) & vertical))
    raise ValueError("Unsupported primitive: " + kind)


def surface_clearance(spec, poses):
    center = np.asarray(spec["position"], float)
    dimensions = np.asarray(spec["dimensions"], float)
    points = np.asarray(poses, float)[:, :3, 3]
    kind = spec["primitive_type"]
    if kind in {"box", "plane"}:
        low, high = center - dimensions / 2, center + dimensions / 2
        closest = np.minimum(np.maximum(points, low), high)
        return float(np.linalg.norm(points - closest, axis=1).min())
    if kind == "sphere":
        return float(np.linalg.norm(points - center, axis=1).min() - dimensions[0] / 2)
    radial = np.linalg.norm(points[:, :2] - center[:2], axis=1) - dimensions[0] / 2
    vertical = np.abs(points[:, 2] - center[2]) - dimensions[2] / 2
    outside = np.maximum(radial, vertical)
    return float(outside.min())


def _clear_or_drop(spec, poses, clearance=1.2):
    if contains_camera(spec, poses) or surface_clearance(spec, poses) < clearance:
        if spec["primitive_type"] == "box":
            return None
        center = np.asarray(spec["position"], float)
        dimensions = list(spec["dimensions"])
        if spec["primitive_type"] == "sphere":
            distance = float(np.linalg.norm(np.asarray(poses)[:, :3, 3] - center, axis=1).min())
            radius = distance - clearance
            if radius < 0.12:
                return None
            return _spec(spec["name"], spec["category"], "sphere", center, [radius * 2] * 3, spec.get("parent"))
        radial = np.linalg.norm(np.asarray(poses)[:, :3, 3][:, :2] - center[:2], axis=1)
        height = dimensions[2]
        inside = np.abs(np.asarray(poses)[:, :3, 3][:, 2] - center[2]) < height / 2
        hit = inside & (radial < dimensions[0] / 2 + clearance)
        if not np.any(hit):
            return spec
        radius = float(radial[hit].min()) - clearance
        if radius < 0.12:
            return None
        return _spec(spec["name"], spec["category"], "cylinder", center, [radius * 2, radius * 2, height], spec.get("parent"))
    return spec


def observed_points(raw, training, poses, k, *, stride=8, min_votes=2):
    """Back-project sampled video pixels. No column fill and no floor completion."""
    poses = np.asarray(poses, float)
    k = np.asarray(k, float)
    ground = float(np.min(poses[:, 2, 3]) - 1.85)
    anchors, distances = [], []
    for index in training:
        rgb, codes, labels = raw(index)
        h, w = codes.shape
        v, u = np.mgrid[0:h:24, 0:w:24]
        rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(k).T @ poses[index][:3, :3].T
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (ground - poses[index][2, 3]) / rays[:, :, 2]
        use = ((labels[v, u] == GROUND) | (labels[v, u] == CATEGORIES.index("grass"))) & (rays[:, :, 2] < -.06) & (z >= 2) & (z <= 70)
        anchors.append(codes[v[use], u[use]])
        distances.append(z[use])
    lut, curve = proxy_lut(np.concatenate(anchors), np.concatenate(distances))
    records = []
    for index in training:
        rgb, codes, labels = raw(index)
        h, w = codes.shape
        v, u = np.mgrid[0:h:stride, 0:w:stride]
        code, cat = codes[v, u], labels[v, u]
        depth = lut[code]
        rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(k).T
        points = (rays * depth[:, :, None]) @ poses[index][:3, :3].T + poses[index][:3, 3]
        use = (code > 0) & (code < 800) & (cat != PERSON) & np.isfinite(points).all(2)
        use &= (points[:, :, 2] > ground - 2) & (points[:, :, 2] < ground + 25)
        cells = np.floor(points[use] / 0.5).astype(np.int64)
        owners = cat[use].astype(np.int64)
        keys = np.column_stack((cells, owners))
        unique, counts = np.unique(keys, axis=0, return_counts=True)
        records.append(np.column_stack((unique, counts)))
        print(f"Observed layout cues {index}: {len(unique)} cells", flush=True)
    keys = np.concatenate(records)
    unique, inverse = np.unique(keys[:, :4], axis=0, return_inverse=True)
    votes = np.zeros(len(unique), np.int64)
    np.add.at(votes, inverse, keys[:, 4])
    keep = votes >= min_votes
    cells, categories = unique[keep, :3], unique[keep, 3]
    points = (cells.astype(float) + 0.5) * 0.5
    report = dict(method="video_depth_code_backprojection", stride=stride, min_votes=min_votes,
                  observed_cells=int(len(points)), curve=curve, ground_plane_height=ground,
                  column_fill=False, floor_completion=False, physical_geometry_verified=False, inferred=True)
    return points, categories, report


def _connected_xy(points, gap_m):
    if len(points) == 0:
        return []
    cells = np.unique(np.floor(points[:, :2] / gap_m).astype(np.int64), axis=0)
    parent = {tuple(c): tuple(c) for c in cells.tolist()}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    cell_set = set(parent)
    for x, y in list(cell_set):
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            other = (x + dx, y + dy)
            if other in cell_set:
                union((x, y), other)
    groups = {}
    for key in cell_set:
        groups.setdefault(find(key), []).append(key)
    assigned = []
    tree = cKDTree(points[:, :2])
    for members in groups.values():
        centers = (np.asarray(members, float) + 0.5) * gap_m
        keep = set()
        for center in centers:
            keep.update(tree.query_ball_point(center, gap_m * 1.5))
        if keep:
            assigned.append(points[sorted(keep)])
    return assigned


def _fit_ground_points(points, poses, names, tile=4.0):
    specs = []
    if len(points) == 0:
        return specs
    tiles = np.floor(points[:, :2] / tile).astype(np.int64)
    for key in np.unique(tiles, axis=0):
        part = points[np.all(tiles == key, axis=1)]
        if len(part) < 3:
            continue
        low = part.min(0)
        high = part.max(0)
        height = max(0.25, float(np.percentile(part[:, 2], 75) - np.percentile(part[:, 2], 25)) + 0.2)
        center_z = float(np.median(part[:, 2]))
        center = [(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, center_z]
        dims = [max(0.5, high[0] - low[0] + 0.5), max(0.5, high[1] - low[1] + 0.5), height]
        name = f"ground_{names['ground']:04d}"
        names["ground"] += 1
        spec = _spec(name, "ground", "box", center, dims)
        if spec and not contains_camera(spec, poses):
            specs.append(spec)
    return specs


def _fit_tree_points(points, poses, names, category):
    specs = []
    for cluster in _connected_xy(points, 1.0 if category == "tree" else 0.75):
        if len(cluster) < 4:
            continue
        low, high = cluster.min(0), cluster.max(0)
        height = float(high[2] - low[2])
        span = float(max(high[0] - low[0], high[1] - low[1], 0.3))
        if category == "tree" and height > 1.0:
            split = low[2] + height * 0.45
            trunk_pts = cluster[cluster[:, 2] <= split]
            canopy_pts = cluster[cluster[:, 2] >= split - 0.2]
            if len(trunk_pts) < 2:
                trunk_pts = cluster[: max(2, len(cluster) // 3)]
            if len(canopy_pts) < 2:
                canopy_pts = cluster[-(max(2, len(cluster) // 2)):]
            trunk_r = max(0.12, float(np.percentile(np.linalg.norm(trunk_pts[:, :2] - trunk_pts[:, :2].mean(0), axis=1), 80)))
            trunk_h = max(0.4, float(trunk_pts[:, 2].max() - trunk_pts[:, 2].min()) + 0.2)
            trunk_c = [float(trunk_pts[:, 0].mean()), float(trunk_pts[:, 1].mean()),
                       float((trunk_pts[:, 2].min() + trunk_pts[:, 2].max()) / 2)]
            trunk = _spec(f"tree_{names['tree']:04d}_trunk", "tree", "cylinder", trunk_c, [trunk_r * 2, trunk_r * 2, trunk_h])
            names["tree"] += 1
            canopy_low, canopy_high = canopy_pts.min(0), canopy_pts.max(0)
            canopy_c = ((canopy_low + canopy_high) / 2).tolist()
            canopy_d = np.maximum(canopy_high - canopy_low, 0.4).tolist()
            parent = trunk["name"] if trunk else None
            canopy = _spec(f"tree_{names['tree']:04d}_canopy", "tree", "sphere", canopy_c, canopy_d, parent)
            names["tree"] += 1
            for item in (trunk, canopy):
                cleared = _clear_or_drop(item, poses) if item else None
                if cleared:
                    specs.append(cleared)
        else:
            radius = max(0.12, span / 2)
            center = [float(cluster[:, 0].mean()), float(cluster[:, 1].mean()), float((low[2] + high[2]) / 2)]
            name = f"{category}_{names[category]:04d}"
            names[category] += 1
            spec = _spec(name, category, "cylinder", center, [radius * 2, radius * 2, max(0.3, height)])
            cleared = _clear_or_drop(spec, poses) if spec else None
            if cleared:
                specs.append(cleared)
    return specs


def _solid_boxes_from_points(points, category, poses, names, cell=0.5):
    """Exact occupied rectangles only; holes are never filled."""
    if len(points) == 0:
        return []
    cells = np.unique(np.floor(points / cell).astype(np.int64), axis=0)
    specs = []
    for low, high in boxes_from_cells(cells, cell, poses, 1):
        center = ((np.asarray(low) + np.asarray(high)) / 2).tolist()
        dims = (np.asarray(high) - np.asarray(low)).tolist()
        name = f"{category}_{names[category]:04d}"
        names[category] += 1
        spec = _spec(name, category, "box", center, dims)
        cleared = _clear_or_drop(spec, poses) if spec else None
        if cleared:
            specs.append(cleared)
    return specs


def fit_primitives_from_points(points, categories, poses, *, max_objects=8000):
    """Fit director primitives to multi-view observed points only."""
    points = np.asarray(points, float)
    categories = np.asarray(categories, np.int64)
    poses = np.asarray(poses, float)
    if len(points) == 0 or len(points) != len(categories):
        raise ValueError("Observed points/categories required")
    names = {name: 1 for name in CATEGORIES}
    specs = []
    for index, name in enumerate(CATEGORIES):
        part = points[categories == index]
        if len(part) == 0 or name == "person":
            continue
        if name == "ground":
            specs.extend(_fit_ground_points(part, poses, names))
        elif name in VERTICAL_CATEGORIES:
            specs.extend(_fit_tree_points(part, poses, names, name))
        else:
            cell = 1.0 if name in {"environment", "building", "rock"} else 0.5
            for cluster in _connected_xy(part, max(1.5, cell * 2)):
                specs.extend(_solid_boxes_from_points(cluster, name, poses, names, cell=cell))
    specs = [item for item in specs if item]
    safe = []
    for item in specs:
        if item["category"] == "ground":
            if not contains_camera(item, poses):
                safe.append(item)
            continue
        cleared = _clear_or_drop(item, poses)
        if cleared and not contains_camera(cleared, poses) and surface_clearance(cleared, poses) >= 1.2 - 1e-6:
            safe.append(cleared)
    specs = safe
    if len(specs) > max_objects:
        order = sorted(range(len(specs)), key=lambda i: -np.prod(specs[i]["dimensions"]))
        specs = [specs[i] for i in order[:max_objects]]
        print(f"Primitive budget trimmed to {len(specs)} largest solids", flush=True)
    if not specs:
        raise ValueError("Primitive fit produced no objects")
    counts = {}
    for item in specs:
        counts[item["primitive_type"]] = counts.get(item["primitive_type"], 0) + 1
    return specs, counts


def export_scene(bundle, specs, base_scene, proxy_receipt, fit_receipt):
    assets = bundle / "assets"
    assets.mkdir()
    centers = np.array([item["position"] for item in specs], float)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centers))
    cloud_path = assets / "primitive_centers.ply"
    if not o3d.io.write_point_cloud(str(cloud_path), cloud):
        raise ValueError("Primitive reference export failed")
    conf = confidence(.4, "inferred", "heuristic_primitive_from_video",
                      "Fit to multi-view video observations; not measured geometry")
    relative_cloud = str(cloud_path.relative_to(bundle))
    by_name = {}
    objects = []
    for item in specs:
        obj = dict(object_id=item["name"], category=item["category"], primitive_type=item["primitive_type"],
                   transform=dict(position=item["position"], rotation=[0, 0, 0, 1], scale=[1, 1, 1]),
                   dimensions=item["dimensions"], parent_id=item["parent"], is_static=True,
                   classification_confidence=conf, confidence=conf, tracking_confidence=conf,
                   geometry_source=dict(kind="heuristic_primitive", point_cloud_path=relative_cloud,
                                        bounds_type="inferred_proxy_aabb", confidence=conf),
                   source_frames=[], observations=[], visibility=[], motion_track=[])
        by_name[item["name"]] = obj
        objects.append(obj)
    for obj in objects:
        if obj["parent_id"] is not None and obj["parent_id"] not in by_name:
            obj["parent_id"] = None
    scene = json.loads(json.dumps(base_scene))
    scene["objects"] = objects
    scene["scene_id"] = "user-approved-primitive-from-video-66ad"
    scene["created_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    scene["appearance"] = dict(texture_mode="category_primitive", background_color=[0.55, 0.58, 0.62])
    json_write(bundle / "proxy_validation.json", proxy_receipt)
    json_write(bundle / "primitive_fit.json", fit_receipt)
    calibration = bundle / "camera_calibration.json"
    if not calibration.is_file():
        raise ValueError("Primitive bundle is missing the copied camera calibration")
    scene["reconstruction_metadata"].update(
        provider="local_heuristic_RGB_video_primitive_blockout",
        data_source="REAL LOCAL VIDEO / USER-APPROVED HEURISTIC PRIMITIVE FROM VIDEO / NOT GEOMETRY RECOVERY",
        empirical_validation=dict(path="camera_calibration.json", sha256=sha256(calibration)),
        heuristic_proxy=dict(path="proxy_validation.json", sha256=sha256(bundle / "proxy_validation.json")),
        primitive_fit=dict(path="primitive_fit.json", sha256=sha256(bundle / "primitive_fit.json")),
        limitations=["User approved completion beyond verified geometry on 2026-11-24.",
                     "Primitives are fit to observed video points; voxel completion is not an input.",
                     "Depth codes supply inferred scale only; physical inverse remains unverified.",
                     "No column fill and no unobserved floor completion.",
                     "Category materials are director colors, not recovered albedo.",
                     "Camera snapshot/focal binding remains inferred; original timeline, ticks and FPS are copied.",
                     "Characters, skeletal animation and unseen completion stay excluded.",
                     "Local internal only; delete with restricted source by 2026-12-24 Asia/Shanghai."])
    validate_scene(scene)
    json_write(bundle / "scene_schema.json", scene)
    return scene


def _load_video_sources():
    base = ROOT / "outputs/stage_ef/real_voxel_rgb"
    source = json.loads((base / "refinement_report.json").read_text(encoding="utf-8"))
    if source["calibration"]["status"] != "passed":
        raise ValueError("Frozen RGB camera calibration required")
    for name, digest in source["source_hashes"].items():
        if sha256(Path(name)) != digest:
            raise ValueError("Original inputs changed")
    seed = json.loads((ROOT / "outputs/stage_ef/real_refined/refinement_report.json").read_text(encoding="utf-8"))
    cache = Path(seed["frame_cache"])
    calibration = json.loads((base / "calibration_report.json").read_text(encoding="utf-8"))
    training = calibration["training_frames"]
    if not verified_frame_cache(cache, sorted(set(training) | set(HOLDOUT))):
        raise ValueError("Selected frame cache failed hashes")
    scene = json.loads((base / "scene_schema.json").read_text(encoding="utf-8"))
    poses = np.asarray([row["world_from_camera"] for row in scene["cameras"][0]["track"]], float)
    k = np.asarray(scene["cameras"][0]["track"][0]["intrinsics"], float)
    camera_hash = hashlib.sha256(json.dumps(scene["cameras"], sort_keys=True).encode()).hexdigest()

    @lru_cache(maxsize=8)
    def raw(index):
        with np.load(cache / f"frame_{index:06d}.npz", allow_pickle=False) as data:
            return data["rgb"].copy(), data["codes"].copy(), data["labels"].copy()

    return scene, poses, k, training, raw, cache, source["source_hashes"], calibration, camera_hash, base


def masked_luminance_mae(original, rendered, mask):
    if not mask.any():
        return None
    a = original.astype(np.float32).mean(2)
    b = rendered.astype(np.float32).mean(2)
    return float(np.abs(a[mask] - b[mask]).mean())


def render_accepted(output, *, blender=None):
    report = json.loads((output / "primitive_run.json").read_text(encoding="utf-8"))
    if report.get("schema") != "primitive.run/1" or report.get("preview_status") != "passed":
        raise ValueError("Primitive preview must pass before rendering")
    if report.get("quality_gate") != "passed":
        raise ValueError("Primitive quality gate must pass before rendering")
    for name, digest in report["source_hashes"].items():
        if sha256(Path(name)) != digest:
            raise ValueError("Primitive original source changed")
    for name, digest in report["preview_hashes"].items():
        if sha256(Path(name)) != digest:
            raise ValueError("Primitive preview changed since acceptance")
    result = run_pipeline(Path(report["source_bundle"]) / "scene_schema.json",
                          ROOT / "outputs/stage_a/sample_66ad/original_video.mp4", output, blender=blender,
                          engine="BLENDER_EEVEE", samples=16, allow_inferred=True, allow_heuristic=True)
    for name in ("blockout_video.mp4", "original_vs_blockout.mp4"):
        if result["media"][name]["frame_count"] != 720 or result["media"][name]["fps"] != "60":
            raise ValueError("Primitive MP4 original timeline mismatch")
    report.update(status="passed", full_video_rendered=True, artifact_hashes=result["artifact_hashes"])
    json_write(output / "primitive_run.json", report)
    return result


def run(args):
    if not args.allow_heuristic:
        raise ValueError("Explicit --allow-heuristic required for the user-approved video primitive path")
    output = args.output.resolve()
    protected = {ROOT / "outputs/stage_ef/heuristic_voxel_proxy",
                 ROOT / "outputs/stage_ef/heuristic_primitive_proxy"}
    if output in protected or ROOT / "data" in output.parents or ROOT / "outputs/stage_a" in output.parents:
        raise ValueError("Choose an independent output; existing voxel/clay directories stay untouched")
    if output == ROOT:
        raise ValueError("Independent primitive output required")
    if args.render_only:
        with refinement_lock(output):
            return render_accepted(output, blender=args.blender)
    if output.exists() and any(output.iterdir()) and not (output / "primitive_run.json").is_file():
        leftovers = {path.name for path in output.iterdir()}
        if leftovers - {".refined.lock", "refinement_report.json"}:
            raise ValueError("Unmanaged primitive output; choose an independent directory")
        for name in ("refinement_report.json", ".refined.lock"):
            path = output / name
            if path.exists():
                path.unlink()
    output.mkdir(parents=True, exist_ok=True)
    with refinement_lock(output):
        scene, poses, k, training, raw, cache, source_hashes, calibration, camera_hash, base = _load_video_sources()
        print("Back-projecting observed video points", flush=True)
        points, categories, layout = observed_points(raw, training, poses, k)
        print(f"Fitting primitives to {len(points)} observed cells", flush=True)
        specs, counts = fit_primitives_from_points(points, categories, poses)
        print(f"Primitive fit: {len(specs)} objects {counts}", flush=True)
        proxy_receipt = dict(schema="heuristic.proxy/1", status="passed", authorization=AUTHORIZATION,
                             geometry_accuracy_validated=False, camera_inside_frame_count=0, coverage_status="passed",
                             camera_sha256=camera_hash, layout=layout,
                             evidence_boundary="Observed video points without voxel column/floor completion")
        fit_receipt = dict(schema="heuristic.primitive/1", status="passed", authorization=AUTHORIZATION,
                           geometry_accuracy_validated=False, camera_inside_frame_count=0, camera_sha256=camera_hash,
                           source="video_observations", primitive_count=len(specs), counts_by_type=counts,
                           evidence_boundary="Boxes, cylinders and spheres fit to video points; not measured geometry")
        report = dict(schema="primitive.run/1", status="fit", authorization=AUTHORIZATION, source_hashes=source_hashes,
                      primitive_count=len(specs), counts_by_type=counts, geometry_accuracy_validated=False,
                      full_video_rendered=False, preview_status="pending", quality_gate="pending",
                      data_policy="Local internal only; delete by 2026-12-24 Asia/Shanghai")
        json_write(output / "primitive_run.json", report)
        bundle = Path(tempfile.mkdtemp(prefix="source_bundle_", dir=output))
        (bundle / "camera_calibration.json").write_bytes((base / "calibration_report.json").read_bytes())
        export_scene(bundle, specs, scene, proxy_receipt, fit_receipt)
        if args.fit_only:
            report.update(status="fit_only", source_bundle=str(bundle))
            json_write(output / "primitive_run.json", report)
            return report
        executable = blender_path(args.blender)
        run_pipeline(bundle / "scene_schema.json", ROOT / "outputs/stage_a/sample_66ad/original_video.mp4", output,
                     blender=executable, engine="BLENDER_EEVEE", samples=16, allow_inferred=True,
                     allow_heuristic=True, generate_only=True)
        frames = sorted(set(HOLDOUT) | {0, 360, 719})
        render_previews(output, frames, executable=executable)
        baseline = ROOT / "outputs/stage_ef/heuristic_primitive_proxy"
        rows = []
        for index in frames:
            rgb, codes, labels = raw(index)
            rendered = np.asarray(Image.open(output / f"frames/frame_{index+1:06d}.png").convert("RGB"))
            mask = (codes > 0) & (codes < 1020) & (labels != PERSON)
            candidate = masked_luminance_mae(rgb, rendered, mask)
            baseline_path = baseline / f"frames/frame_{index+1:06d}.png"
            baseline_mae = None
            if baseline_path.is_file():
                previous = np.asarray(Image.open(baseline_path).convert("RGB"))
                baseline_mae = masked_luminance_mae(rgb, previous, mask)
            rows.append(dict(frame_index=index, masked_luminance_mae=candidate, clay_baseline_mae=baseline_mae))
        candidate_scores = [row["masked_luminance_mae"] for row in rows if row["masked_luminance_mae"] is not None]
        baseline_scores = [row["clay_baseline_mae"] for row in rows if row["clay_baseline_mae"] is not None]
        if not candidate_scores:
            raise ValueError("Primitive previews produced no comparable pixels")
        mean_candidate = float(np.mean(candidate_scores))
        mean_baseline = float(np.mean(baseline_scores)) if baseline_scores else None
        improved = mean_baseline is None or mean_candidate < mean_baseline
        report.update(status="preview_passed" if improved else "quality_rejected",
                      preview_status="passed" if improved else "failed",
                      quality_gate="passed" if improved else "failed",
                      source_bundle=str(bundle), primitive_count=len(specs), preview_frames=rows,
                      mean_masked_luminance_mae=mean_candidate, clay_baseline_mean_mae=mean_baseline,
                      preview_hashes={str(output / f"frames/frame_{index+1:06d}.png"): sha256(output / f"frames/frame_{index+1:06d}.png")
                                      for index in frames})
        json_write(output / "primitive_run.json", report)
        if not improved:
            raise ValueError("Video primitives did not improve masked luminance agreement versus clay baseline")
        if not args.preview_only:
            return render_accepted(output, blender=executable)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-heuristic", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/stage_ef/heuristic_primitive_from_video")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--blender", type=Path)
    args = parser.parse_args(argv)
    if args.fit_only and args.render_only:
        parser.error("Choose either --fit-only or --render-only")
    try:
        run(args)
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        report_path = args.output.resolve() / "primitive_run.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("schema") == "primitive.run/1":
                report.update(status="failed" if report.get("status") != "quality_rejected" else "quality_rejected",
                              full_video_rendered=False, error_type=type(error).__name__)
                json_write(report_path, report)
        print(f"Primitive blockout stopped: {error}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
