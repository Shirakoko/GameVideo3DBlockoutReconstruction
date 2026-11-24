"""Clay primitive blockout beside the voxel proxy, not a replacement.

Boxes, vertical cylinders and spheres are fit to the already accepted voxel
occupancy. The voxel output stays where it is. Camera motion, source ticks
and FPS are copied, not redesigned. Nothing here is measured geometry.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

import numpy as np
import open3d as o3d
from PIL import Image

from agents.object_detection import confidence
from blender.pipeline import ROOT, blender_path, json_write, run_pipeline, sha256
from reconstruction.heuristic_proxy import AUTHORIZATION
from reconstruction.refine import refinement_lock, render_previews
from reconstruction.refined_calibration import HOLDOUT
from reconstruction.refined_segmentation import CATEGORIES
from scene_schema import validate_scene

BOX_CATEGORIES = {"environment", "rock", "barrier", "building", "wall", "road"}
VERTICAL_SPAN = {"tree": 6.0, "grass": 2.5, "shrub": 3.0}


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
            if x == prev+1:
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
            if y == prev+1:
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
            if z == prev+1:
                prev = z
            else:
                boxes.append((x0, x1, y0, y1, start, prev))
                start = prev = z
        boxes.append((x0, x1, y0, y1, start, prev))
    return boxes


def box_bounds(box, size, factor):
    x0, x1, y0, y1, z0, z1 = box
    low = np.array([x0, y0, z0], float)*factor*size
    high = (np.array([x1, y1, z1], float)+1)*factor*size
    return low, high


def camera_near_aabb(low, high, poses, clearance=1.2):
    """Coarse boxes swell into the free corridor; split those before export."""
    if len(poses) == 0:
        return False
    points = np.asarray(poses, float)[:, :3, 3]
    closest = np.minimum(np.maximum(points, low), high)
    return bool(np.any(np.linalg.norm(points-closest, axis=1) < clearance))


def boxes_from_cells(cells, size, poses, factor):
    cells = np.unique(np.asarray(cells, np.int64), axis=0)
    if len(cells) == 0:
        return []
    if factor <= 1:
        return [box_bounds(box, size, 1) for box in merge_boxes(cells)]
    coarse = np.unique(np.floor(cells/factor).astype(np.int64), axis=0)
    safe, unsafe = [], []
    for box in merge_boxes(coarse):
        low, high = box_bounds(box, size, factor)
        if camera_near_aabb(low, high, poses):
            unsafe.append(box)
        else:
            safe.append((low, high))
    if unsafe:
        quant = np.floor(cells/factor).astype(np.int64)
        keep = np.zeros(len(cells), dtype=bool)
        for x0, x1, y0, y1, z0, z1 in unsafe:
            keep |= ((quant[:, 0] >= x0) & (quant[:, 0] <= x1) & (quant[:, 1] >= y0)
                     & (quant[:, 1] <= y1) & (quant[:, 2] >= z0) & (quant[:, 2] <= z1))
        safe.extend(boxes_from_cells(cells[keep], size, poses, max(1, factor//2)))
    return safe


def cluster_columns(cells, gap):
    buckets = {}
    for x, y, z in np.asarray(cells, np.int64).tolist():
        buckets.setdefault((int(x), int(y)), []).append(int(z))
    parent = {key: key for key in buckets}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for x, y in buckets:
        for dx in range(0, gap+1):
            for dy in range(-gap, gap+1):
                if dx == 0 and dy <= 0:
                    continue
                other = (x+dx, y+dy)
                if other in buckets:
                    union((x, y), other)
    groups = {}
    for key in buckets:
        groups.setdefault(find(key), []).append(key)
    return buckets, groups


def _spec(name, category, primitive, position, dimensions, parent=None):
    dimensions = [float(v) for v in dimensions]
    if min(dimensions) <= 1e-4:
        return None
    return dict(name=name, category=category, primitive_type=primitive,
                position=[float(v) for v in position], dimensions=dimensions, parent=parent)


def _clear_round(position, dimensions, poses, *, sphere):
    center = np.asarray(position, float)
    radius, height = dimensions[0]/2, dimensions[2]
    points = np.asarray(poses, float)[:, :3, 3]
    if sphere:
        distance = np.linalg.norm(points-center, axis=1)
        if not np.any(distance < radius):
            return position, dimensions
        limit = float(distance.min())-0.3
    else:
        radial = np.linalg.norm(points[:, :2]-center[:2], axis=1)
        inside = np.abs(points[:, 2]-center[2]) < height/2
        hit = inside & (radial < radius)
        if not np.any(hit):
            return position, dimensions
        limit = float(radial[hit].min())-0.25
    if limit < 0.12:
        return None
    return position, [limit*2, limit*2, height if not sphere else limit*2]


def contains_camera(spec, poses, margin=0.05):
    center = np.asarray(spec["position"], float)
    dimensions = np.asarray(spec["dimensions"], float)
    points = np.asarray(poses, float)[:, :3, 3]
    kind = spec["primitive_type"]
    if kind in {"box", "plane"}:
        low, high = center-dimensions/2, center+dimensions/2
        return bool(np.any(np.all(points > low+margin, axis=1) & np.all(points < high-margin, axis=1)))
    if kind == "sphere":
        return bool(np.any(np.linalg.norm(points-center, axis=1) < dimensions[0]/2-margin))
    if kind == "cylinder":
        radial = np.linalg.norm(points[:, :2]-center[:2], axis=1)
        vertical = np.abs(points[:, 2]-center[2]) < dimensions[2]/2-margin
        return bool(np.any((radial < dimensions[0]/2-margin) & vertical))
    raise ValueError("Unsupported primitive: "+kind)


def _named_boxes(cells, size, poses, category, names, factor):
    specs = []
    for low, high in boxes_from_cells(cells, size, poses, factor):
        name = f"{category}_{names[category]:04d}"
        names[category] += 1
        specs.append(_spec(name, category, "box", (low+high)/2, high-low))
    return [item for item in specs if item]


def _fit_ground(cells, size, poses, names):
    values, counts = np.unique(cells[:, 2], return_counts=True)
    floor = int(values[int(np.argmax(counts))])
    layer = cells[cells[:, 2] == floor]
    rest = cells[cells[:, 2] != floor]
    low = np.array([layer[:, 0].min(), layer[:, 1].min(), floor], float)*size
    high = (np.array([layer[:, 0].max(), layer[:, 1].max(), floor], float)+1)*size
    specs = []
    if camera_near_aabb(low, high, poses):
        specs.extend(_named_boxes(layer, size, poses, "ground", names, 4))
    else:
        name = f"ground_{names['ground']:04d}"
        names["ground"] += 1
        specs.append(_spec(name, "ground", "box", (low+high)/2, high-low))
    if len(rest):
        specs.extend(_named_boxes(rest, size, poses, "ground", names, 4))
    return [item for item in specs if item]


def _fit_vertical(cells, size, poses, category, names, gap):
    buckets, groups = cluster_columns(cells, gap)
    specs, leftover = [], []
    ordered = sorted(groups.values(), key=lambda members: (min(item[0] for item in members), min(item[1] for item in members)))
    for members in ordered:
        xs = [item[0] for item in members]
        ys = [item[1] for item in members]
        span = max((max(xs)-min(xs)+1)*size, (max(ys)-min(ys)+1)*size)
        owned = np.array([(x, y, z) for x, y in members for z in buckets[(x, y)]], np.int64)
        if span > VERTICAL_SPAN[category]:
            leftover.append(owned)
            continue
        z0, z1 = int(owned[:, 2].min()), int(owned[:, 2].max())
        center_x = (min(xs)+max(xs)+1)*size/2
        center_y = (min(ys)+max(ys)+1)*size/2
        base, top = z0*size, (z1+1)*size
        height = top-base
        radius = max(span/2, size*0.5)
        if category == "tree" and height > size*1.5:
            trunk_r = float(np.clip(radius*0.22, 0.15, 0.55))
            trunk_h = max(size, min(height*0.45, height-size*0.5))
            trunk = _clear_round([center_x, center_y, base+trunk_h/2], [trunk_r*2, trunk_r*2, trunk_h], poses, sphere=False)
            canopy_r = max(radius, trunk_r*1.8)
            canopy = _clear_round([center_x, center_y, base+trunk_h+canopy_r*0.35], [canopy_r*2]*3, poses, sphere=True)
            trunk_name = None
            if trunk:
                trunk_name = f"tree_{names['tree']:04d}_trunk"
                names["tree"] += 1
                specs.append(_spec(trunk_name, "tree", "cylinder", trunk[0], trunk[1]))
            if canopy:
                canopy_name = f"tree_{names['tree']:04d}_canopy"
                names["tree"] += 1
                specs.append(_spec(canopy_name, "tree", "sphere", canopy[0], canopy[1], trunk_name))
        else:
            radius = float(np.clip(radius, 0.15, 0.9 if category != "tree" else radius))
            shaped = _clear_round([center_x, center_y, base+height/2], [radius*2, radius*2, height], poses, sphere=False)
            if shaped:
                name = f"{category}_{names[category]:04d}"
                names[category] += 1
                specs.append(_spec(name, category, "cylinder", shaped[0], shaped[1]))
    if leftover:
        specs.extend(_named_boxes(np.concatenate(leftover), size, poses, category, names, 4))
    return [item for item in specs if item]


def fit_primitives(cells, categories, size, poses, *, max_objects=2500):
    """Axis-aligned clay stand-ins. Large connected masses stay as boxes."""
    cells = np.asarray(cells, np.int64)
    categories = np.asarray(categories, np.int64)
    poses = np.asarray(poses, float)
    if cells.ndim != 2 or cells.shape[1] != 3 or len(cells) != len(categories):
        raise ValueError("Primitive fit requires paired occupancy cells and categories")
    if size <= 0:
        raise ValueError("Primitive block size must be positive")
    names = {name: 1 for name in CATEGORIES}
    ground, vertical, boxes = [], [], []
    box_parts = []
    for index, name in enumerate(CATEGORIES):
        part = cells[categories == index]
        if len(part) == 0 or name == "person":
            continue
        if name == "ground":
            ground = _fit_ground(part, size, poses, names)
        elif name in VERTICAL_SPAN:
            vertical.extend(_fit_vertical(part, size, poses, name, names, gap=1))
        elif name in BOX_CATEGORIES:
            box_parts.append((name, part))
        else:
            box_parts.append((name, part))
    chosen = None
    for factor in (2, 4, 8):
        built = []
        scratch = dict(names)
        for name, part in box_parts:
            built.extend(_named_boxes(part, size, poses, name, scratch, factor))
        chosen = [item for item in ground+vertical+built if item]
        names = scratch
        if len(chosen) <= max_objects:
            break
    blocked = [item["name"] for item in chosen if contains_camera(item, poses)]
    if blocked:
        raise ValueError("Primitive fit encloses a source camera: "+", ".join(blocked[:8]))
    if not chosen:
        raise ValueError("Primitive fit produced no objects")
    counts = {}
    for item in chosen:
        counts[item["primitive_type"]] = counts.get(item["primitive_type"], 0)+1
    return chosen, counts


def export_scene(bundle, specs, base_scene, proxy_receipt, fit_receipt):
    assets = bundle/"assets"
    assets.mkdir()
    centers = np.array([item["position"] for item in specs], float)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centers))
    cloud_path = assets/"primitive_centers.ply"
    if not o3d.io.write_point_cloud(str(cloud_path), cloud):
        raise ValueError("Primitive reference export failed")
    conf = confidence(.35, "inferred", "heuristic_primitive_fit", "Fit to accepted voxel occupancy; not measured geometry")
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
    scene["scene_id"] = "user-approved-primitive-proxy-66ad"
    scene["created_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    scene["appearance"] = dict(texture_mode="clay_primitive", background_color=[0.62, 0.63, 0.65])
    json_write(bundle/"proxy_validation.json", proxy_receipt)
    json_write(bundle/"primitive_fit.json", fit_receipt)
    calibration = bundle/"camera_calibration.json"
    if not calibration.is_file():
        raise ValueError("Primitive bundle is missing the copied camera calibration")
    scene["reconstruction_metadata"].update(
        provider="local_heuristic_RGB_layout_primitive_blockout",
        data_source="REAL LOCAL VIDEO / USER-APPROVED HEURISTIC PRIMITIVE BLOCKOUT / NOT GEOMETRY RECOVERY",
        empirical_validation=dict(path="camera_calibration.json", sha256=sha256(calibration)),
        heuristic_proxy=dict(path="proxy_validation.json", sha256=sha256(bundle/"proxy_validation.json")),
        primitive_fit=dict(path="primitive_fit.json", sha256=sha256(bundle/"primitive_fit.json")),
        limitations=["User approved completion beyond verified geometry on 2026-11-24.",
                     "Primitives are fit to the accepted voxel occupancy; the voxel output is unchanged.",
                     "Boxes, cylinders and spheres are editable stand-ins, NOT measured surfaces.",
                     "Uniform clay materials are a director-blockout look, not recovered albedo.",
                     "Camera snapshot/focal binding remains inferred; original timeline, ticks and FPS are copied.",
                     "Characters, skeletal animation and unseen completion stay excluded.",
                     "Local internal only; delete with restricted source by 2026-12-24 Asia/Shanghai."])
    validate_scene(scene)
    json_write(bundle/"scene_schema.json", scene)
    return scene


def _load_accepted_voxels(voxel_dir, voxel_path):
    run_receipt = json.loads((voxel_dir/"proxy_run.json").read_text(encoding="utf-8"))
    validation = json.loads((voxel_dir/"proxy_validation.json").read_text(encoding="utf-8"))
    if (validation.get("schema") != "heuristic.proxy/1" or validation.get("status") != "passed"
            or validation.get("camera_inside_frame_count") != 0 or validation.get("coverage_status") != "passed"
            or validation.get("authorization") != AUTHORIZATION):
        raise ValueError("Accepted voxel proxy receipt required; primitive fit will not invent a new layout")
    if sha256(voxel_path) != validation.get("voxel_sha256"):
        raise ValueError("Voxel occupancy does not match the accepted proxy receipt")
    for name, digest in run_receipt.get("source_hashes", {}).items():
        if sha256(Path(name)) != digest:
            raise ValueError("Original inputs changed since the voxel proxy")
    source = Path(run_receipt["source_bundle"])
    scene = json.loads((source/"scene_schema.json").read_text(encoding="utf-8"))
    camera_hash = hashlib.sha256(json.dumps(scene["cameras"], sort_keys=True).encode()).hexdigest()
    if validation.get("camera_sha256") != camera_hash:
        raise ValueError("Voxel scene cameras no longer match the clearance receipt")
    with np.load(voxel_path, allow_pickle=False) as data:
        cells, categories = data["cells"].copy(), data["categories"].copy()
    return scene, validation, source, cells, categories, camera_hash


def _structure(image):
    return float(image.astype(np.float32).mean(axis=2).std())


def render_accepted(output, *, blender=None):
    report = json.loads((output/"primitive_run.json").read_text(encoding="utf-8"))
    if report.get("schema") != "primitive.run/1" or report.get("preview_status") != "passed":
        raise ValueError("Primitive preview must pass before rendering")
    for name, digest in report["source_hashes"].items():
        if sha256(Path(name)) != digest:
            raise ValueError("Primitive original source changed")
    for name, digest in report["preview_hashes"].items():
        if sha256(Path(name)) != digest:
            raise ValueError("Primitive preview changed since acceptance")
    result = run_pipeline(Path(report["source_bundle"])/"scene_schema.json",
                          ROOT/"outputs/stage_a/sample_66ad/original_video.mp4", output, blender=blender,
                          engine="BLENDER_EEVEE", samples=16, allow_inferred=True, allow_heuristic=True)
    for name in ("blockout_video.mp4", "original_vs_blockout.mp4"):
        if result["media"][name]["frame_count"] != 720 or result["media"][name]["fps"] != "60":
            raise ValueError("Primitive MP4 original timeline mismatch")
    report.update(status="passed", full_video_rendered=True, artifact_hashes=result["artifact_hashes"])
    json_write(output/"primitive_run.json", report)
    return result


def run(args):
    if not args.allow_heuristic:
        raise ValueError("Explicit --allow-heuristic required; this path reuses the user-approved voxel layout")
    output, voxel_dir = args.output.resolve(), args.voxel_run.resolve()
    voxel_path = args.voxels.resolve()
    if output in {ROOT, voxel_dir} or ROOT/"data" in output.parents or ROOT/"outputs/stage_a" in output.parents or voxel_dir in output.parents:
        raise ValueError("Primitive output must be an independent directory, leaving the voxel proxy untouched")
    if args.render_only:
        with refinement_lock(output):
            return render_accepted(output, blender=args.blender)
    if output.exists() and any(output.iterdir()) and not (output/"primitive_run.json").is_file():
        raise ValueError("Unmanaged primitive output; choose an independent directory")
    output.mkdir(parents=True, exist_ok=True)
    with refinement_lock(output):
        scene, validation, source, cells, categories, camera_hash = _load_accepted_voxels(voxel_dir, voxel_path)
        size = float(validation["block_size_metres"])
        poses = np.asarray([row["world_from_camera"] for row in scene["cameras"][0]["track"]], float)
        print(f"Fitting clay primitives to {len(cells)} accepted voxels", flush=True)
        specs, counts = fit_primitives(cells, categories, size, poses)
        print(f"Primitive fit: {len(specs)} objects {counts}", flush=True)
        fit_receipt = dict(schema="heuristic.primitive/1", status="passed", authorization=AUTHORIZATION,
                           geometry_accuracy_validated=False, camera_inside_frame_count=0, camera_sha256=camera_hash,
                           source_voxel_sha256=validation["voxel_sha256"], block_size_metres=size,
                           primitive_count=len(specs), counts_by_type=counts,
                           evidence_boundary="Boxes, cylinders and spheres fit to accepted voxels; not measured geometry")
        report = dict(schema="primitive.run/1", status="fit", authorization=AUTHORIZATION,
                      source_hashes=json.loads((voxel_dir/"proxy_run.json").read_text(encoding="utf-8"))["source_hashes"],
                      voxel_sha256=validation["voxel_sha256"], primitive_count=len(specs), counts_by_type=counts,
                      geometry_accuracy_validated=False, full_video_rendered=False, preview_status="pending",
                      data_policy="Local internal only; delete by 2026-12-24 Asia/Shanghai")
        json_write(output/"primitive_run.json", report)
        if args.fit_only:
            bundle = Path(tempfile.mkdtemp(prefix="source_bundle_", dir=output))
            (bundle/"camera_calibration.json").write_bytes((source/"camera_calibration.json").read_bytes())
            export_scene(bundle, specs, scene, validation, fit_receipt)
            report.update(status="fit_only", source_bundle=str(bundle))
            json_write(output/"primitive_run.json", report)
            return report
        bundle = Path(tempfile.mkdtemp(prefix="source_bundle_", dir=output))
        (bundle/"camera_calibration.json").write_bytes((source/"camera_calibration.json").read_bytes())
        export_scene(bundle, specs, scene, validation, fit_receipt)
        executable = blender_path(args.blender)
        run_pipeline(bundle/"scene_schema.json", ROOT/"outputs/stage_a/sample_66ad/original_video.mp4", output,
                     blender=executable, engine="BLENDER_EEVEE", samples=16, allow_inferred=True,
                     allow_heuristic=True, generate_only=True)
        frames = sorted(set(HOLDOUT) | {0, 360, 719})
        render_previews(output, frames, executable=executable)
        rows = []
        for index in frames:
            rendered = np.asarray(Image.open(output/f"frames/frame_{index+1:06d}.png").convert("RGB"))
            rows.append(dict(frame_index=index, luminance_std=_structure(rendered)))
        detail = float(min(row["luminance_std"] for row in rows))
        report.update(status="preview_passed" if detail >= 4 else "quality_rejected",
                      preview_status="passed" if detail >= 4 else "failed", source_bundle=str(bundle),
                      primitive_count=len(specs), preview_frames=rows,
                      preview_hashes={str(output/f"frames/frame_{index+1:06d}.png"): sha256(output/f"frames/frame_{index+1:06d}.png") for index in frames})
        json_write(output/"primitive_run.json", report)
        if detail < 4:
            raise ValueError("Primitive previews have no readable form")
        if not args.preview_only:
            return render_accepted(output, blender=executable)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-heuristic", action="store_true")
    parser.add_argument("--voxel-run", type=Path, default=ROOT/"outputs/stage_ef/heuristic_voxel_proxy")
    parser.add_argument("--voxels", type=Path, default=ROOT/"outputs/stage_ef/heuristic_voxel_proxy/proxy_voxels.npz")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/stage_ef/heuristic_primitive_proxy")
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
        report_path = args.output.resolve()/"primitive_run.json"
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
