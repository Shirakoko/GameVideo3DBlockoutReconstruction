"""Local multi-view RGB voxel blockout, independent of the unknown depth inverse.

Camera hypotheses are frozen from training-only calibration. Static feature
tracks are triangulated only from training frames and tested on RGB holdouts.
Dense three-view LK tracks then provide the observed surface voxel geometry.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from blender.pipeline import ROOT, blender_path, json_write, run_pipeline, sha256
from blender.media import probe_video
from reconstruction.coordinate_transform import transform_points, inverse_rigid
from reconstruction.refined_calibration import HOLDOUT, pose_track, project, triangulate
from reconstruction.refined_segmentation import CATEGORIES
from reconstruction.refined_voxels import VoxelFusion
from reconstruction.refine import (make_scene, preview_quality, refinement_lock,
    render_previews, verified_frame_cache)


def match_indices(a, b):
    if min(len(a[0]), len(b[0])) < 2:
        return np.empty((0, 2), int)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(a[1], b[1], k=2)
    reverse = matcher.knnMatch(b[1], a[1], k=2)
    back = {m.queryIdx: m.trainIdx for pair in reverse if len(pair) == 2
        for m, n in [pair] if m.distance < .75*n.distance}
    good = [(m.queryIdx, m.trainIdx) for pair in forward if len(pair) == 2 for m, n in [pair]
        if m.distance < .75*n.distance and back.get(m.trainIdx) == m.queryIdx]
    return np.asarray(good, int).reshape(-1, 2)


def static_tracks(training, feature_loader, poses, k):
    """Union feature IDs; ambiguous same-frame tracks are never triangulated."""
    parent = {}
    def find(node):
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            node, parent[node] = parent[node], root
        return root
    for gap in (1, 2, 3, 4):
        for a, b in zip(training, training[gap:]):
            matches = match_indices(feature_loader(a), feature_loader(b))
            for ia, ib in matches:
                root_a, root_b = find((a, int(ia))), find((b, int(ib)))
                if root_a != root_b:
                    parent[root_b] = root_a
    groups = defaultdict(list)
    for node in parent:
        groups[find(node)].append(node)
    tracks, point_for_feature = [], {}
    for nodes in groups.values():
        if len(nodes) < 3 or len({n[0] for n in nodes}) != len(nodes):
            continue
        nodes.sort()
        a, b = max(((a, b) for j, a in enumerate(nodes) for b in nodes[j+1:]),
            key=lambda pair: np.linalg.norm(poses[pair[0][0]][:3,3]-poses[pair[1][0]][:3,3]))
        ua, ub = feature_loader(a[0])[0][a[1]], feature_loader(b[0])[0][b[1]]
        points, keep = triangulate(ua[None], ub[None], poses[a[0]], poses[b[0]], k)
        if not len(points):
            continue
        point = points[0]
        inliers = []
        for frame, idx in nodes:
            uv, z = project(point[None], poses[frame], k)
            if .1 < z[0] <= 80 and np.linalg.norm(uv[0]-feature_loader(frame)[0][idx]) <= 2.5:
                inliers.append((frame, idx))
        if len(inliers) < 3:
            continue
        identity = len(tracks)
        tracks.append((point, inliers))
        for node in inliers:
            point_for_feature[node] = identity
    if len(tracks) < 100:
        raise ValueError("Too few three-view static RGB tracks")
    return tracks, point_for_feature


def validate_tracks(training, feature_loader, tracks, point_for_feature, poses, k):
    errors, rows = [], []
    for h in HOLDOUT:
        candidates = defaultdict(list)
        for neighbor in sorted(training, key=lambda i: abs(i-h))[:6]:
            for ih, it in match_indices(feature_loader(h), feature_loader(neighbor)):
                identity = point_for_feature.get((neighbor, int(it)))
                if identity is not None:
                    candidates[int(ih)].append(identity)
        frame_errors = []
        for ih, identities in candidates.items():
            # Select by training evidence, never by the heldout reprojection error.
            identity = max(set(identities), key=lambda i: (identities.count(i), len(tracks[i][1])))
            point = tracks[identity][0]
            uv, z = project(point[None], poses[h], k)
            if .1 < z[0] <= 80 and np.isfinite(uv).all():
                frame_errors.append(float(np.linalg.norm(uv[0]-feature_loader(h)[0][ih])))
        errors.extend(frame_errors)
        rows.append(dict(frame_index=h, samples=len(frame_errors),
            median_pixels=float(np.median(frame_errors)) if frame_errors else None,
            p90_pixels=float(np.percentile(frame_errors,90)) if frame_errors else None))
    median = float(np.median(errors)) if errors else None
    p90 = float(np.percentile(errors,90)) if errors else None
    passed = len(errors) >= 100 and all(r["samples"] >= 10 for r in rows) and median <= 3 and p90 <= 10
    return dict(status="passed" if passed else "failed", sample_count=len(errors), pairs=rows,
        median_pixels=median, p90_pixels=p90,
        thresholds=dict(median_pixels=3, p90_pixels=10, minimum_samples=100, minimum_per_frame=10),
        evidence="Three-or-more training-view world points projected into excluded RGB holdouts; no heldout fitting")


def dense_triplet(a, b, c, raw, poses, k, stride=4):
    """Three-view static flow with cycle, depth, epipolar and reprojection gates."""
    rgb_a, codes_a, labels_a = raw(a)
    rgb_b, codes_b, labels_b = raw(b)
    rgb_c, codes_c, labels_c = raw(c)
    h, w = codes_a.shape
    v, u = np.mgrid[12:h-12:stride, 12:w-12:stride]
    static = (codes_a[v,u] > 0) & (codes_a[v,u] < 1020) & (labels_a[v,u] != CATEGORIES.index("person"))
    uv_a = np.column_stack((u[static],v[static])).astype(np.float32).reshape(-1,1,2)
    if not len(uv_a):
        return np.empty((0,3)), [np.empty((0,2))]*3
    gray = [cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY) for rgb in (rgb_a,rgb_b,rgb_c)]
    options = dict(winSize=(21,21), maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    uv_b, status_b, err_b = cv2.calcOpticalFlowPyrLK(gray[0],gray[1],uv_a,None,**options)
    uv_c, status_c, err_c = cv2.calcOpticalFlowPyrLK(gray[1],gray[2],uv_b,None,**options)
    back_b, status_bb, _ = cv2.calcOpticalFlowPyrLK(gray[1],gray[0],uv_b,None,**options)
    back_c, status_bc, _ = cv2.calcOpticalFlowPyrLK(gray[2],gray[0],uv_c,None,**options)
    keep = (status_b.ravel()>0)&(status_c.ravel()>0)&(status_bb.ravel()>0)&(status_bc.ravel()>0)
    keep &= (err_b.ravel()<20)&(err_c.ravel()<20)
    keep &= (np.linalg.norm(back_b[:,0]-uv_a[:,0],axis=1)<.8)&(np.linalg.norm(back_c[:,0]-uv_a[:,0],axis=1)<.8)
    uv = [uv_a[:,0],uv_b[:,0],uv_c[:,0]]
    for pixels, codes, labels in zip(uv,(codes_a,codes_b,codes_c),(labels_a,labels_b,labels_c)):
        finite = np.isfinite(pixels).all(1)
        xy = np.rint(np.where(finite[:,None],pixels,-1)).astype(int)
        inside = finite&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        x,y = np.clip(xy[:,0],0,w-1),np.clip(xy[:,1],0,h-1)
        keep &= inside & (codes[y,x]>0)&(codes[y,x]<1020)&(labels[y,x]!=CATEGORIES.index("person"))
    uv = [p[keep].astype(float) for p in uv]
    points, observable = triangulate(uv[0],uv[2],poses[a],poses[c],k)
    uv = [p[observable] for p in uv]
    if not len(points):
        return points, uv
    test, z = project(points,poses[b],k)
    keep = (z>.1)&(z<=80)&(np.linalg.norm(test-uv[1],axis=1)<=2.5)
    for index in (a,c):
        _, z = project(points,poses[index],k)
        keep &= (z>.1)&(z<=80)
    return points[keep], [p[keep] for p in uv]


def rasterize_points(raster, pixels, depths):
    h,w = raster.shape
    xy=np.rint(pixels).astype(int)
    valid=np.isfinite(depths)&(depths>.1)&(depths<=80)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
    np.minimum.at(raster.ravel(),xy[valid,1]*w+xy[valid,0],depths[valid].astype(np.float32))


def run(args):
    if not args.allow_inferred:
        raise ValueError("RGB multi-view blockout requires explicit --allow-inferred")
    seed_output=ROOT/"outputs/stage_ef/real_refined"
    seed=json.loads((seed_output/"refinement_report.json").read_text())
    camera=seed["calibration"]
    cache=Path(seed["frame_cache"])
    indices=sorted(int(p.name[6:12]) for p in cache.glob("frame_*.npz"))
    manifest=verified_frame_cache(cache,indices)
    if not manifest:
        raise ValueError("RGB source cache hashes failed")
    for name,digest in seed["input_hashes"].items():
        if not Path(name).is_file() or sha256(Path(name))!=digest:
            raise ValueError("Source changed since the training-only camera calibration")
    prepared=ROOT/"outputs/stage_a/sample_66ad"
    source=ROOT/"data/input/samples/66ad9535-583f-4a25-b655-6b1697c83a40"
    rows=json.loads((prepared/"camera/poses.json").read_text())["frames"]
    world=json.loads((source/"telemetry/world.json").read_text())
    binding=camera["camera_convention"]
    poses=pose_track(rows,world["world_to_meters"],**{key:binding[key] for key in
        ("angular_unit","yaw_sign","pitch_sign","field","frame_offset")})
    k=np.asarray(camera["intrinsics"],float)
    training=[i for i in indices if i not in HOLDOUT]
    @lru_cache(maxsize=8)
    def raw(index):
        with np.load(cache/f"frame_{index:06d}.npz",allow_pickle=False) as data:
            return data["rgb"].copy(),data["codes"].copy(),data["labels"].copy()
    @lru_cache(maxsize=70)
    def feature(index):
        implementation=seed["code_hashes"][str(ROOT/"reconstruction/refined_calibration.py")]
        key=hashlib.sha256((manifest["hashes"][f"frame_{index:06d}.npz"]+implementation+"sift8000-static-rgb-v2").encode()).hexdigest()[:16]
        path=cache/f"features_{key}_{index:06d}.npz"
        if not path.is_file() or not path.with_suffix(".sha256").is_file() or sha256(path)!=path.with_suffix(".sha256").read_text():
            raise ValueError("Camera calibration feature cache is missing/tampered")
        with np.load(path,allow_pickle=False) as data:
            return data["points"].copy(),data["descriptors"].copy()
    output=args.output.resolve()
    if output==prepared or output==source or prepared in output.parents or source in output.parents or output==seed_output:
        raise ValueError("Choose an independent RGB voxel output")
    if output.is_dir() and any(output.iterdir()) and not (output/"refinement_report.json").is_file():
        raise ValueError("Refusing unmanaged nonempty RGB output")
    output.mkdir(parents=True,exist_ok=True)
    with refinement_lock(output):
        report=dict(schema="refined.run/1",status="running",geometry_method="rgb_three_view",
            source_hashes=seed["input_hashes"],camera_seed_sha256=sha256(seed_output/"refinement_report.json"),
            physical_contract_verified=False,depth_metric_calibration_status=camera["status"],
            depth_inverse_used=False,style="voxel",block_size_metres=args.voxel_size,
            full_video_rendered=False,data_policy="Local internal only; delete by 2026-12-24 Asia/Shanghai")
        json_write(output/"refinement_report.json",report)
        print("Validating training-only multi-view RGB tracks",flush=True)
        tracks,lookup=static_tracks(training,feature,poses,k)
        validation=validate_tracks(training,feature,tracks,lookup,poses,k)
        calibration=dict(schema="refined.calibration/1",status=validation["status"],provenance="inferred",
            physical_contract_verified=False,geometry_method="rgb_three_view",depth_inverse_used=False,
            original_depth_validation=camera["validation"],camera_convention=binding,intrinsics=k.tolist(),
            training_frames=training,holdout_frames=list(HOLDOUT),static_tracks=len(tracks),validation=validation,
            limitations=["Scale conditional on telemetry world units.","Camera focal/tick binding inferred, not SDK-certified.",
                "Static observed RGB surfaces only; no unseen completion or articulated characters."])
        json_write(output/"calibration_report.json",calibration)
        report.update(status=validation["status"],calibration=calibration)
        json_write(output/"refinement_report.json",report)
        print(f"RGB holdout validation: {validation}",flush=True)
        if validation["status"]!="passed":
            return 2
        if args.calibrate_only:
            return 0
        h,w=raw(training[0])[1].shape
        fingerprint=hashlib.sha256(json.dumps(dict(inputs=seed["input_hashes"],camera=binding,k=k.tolist(),
            algorithm=sha256(Path(__file__)),stride=args.stride,frames=training),sort_keys=True).encode()).hexdigest()
        estimate_dir=output/".rgb_geometry_cache"/fingerprint
        estimate_dir.mkdir(parents=True,exist_ok=True)
        receipt_path=estimate_dir/"frames.json"
        receipt=json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
        expected={f"frame_{i:06d}.npz" for i in training}
        reusable=set(receipt.get("hashes",{}))==expected and all((estimate_dir/name).is_file() and
            sha256(estimate_dir/name)==digest for name,digest in receipt.get("hashes",{}).items())
        if not reusable:
            rasters={i:np.full((h,w),np.inf,np.float32) for i in training}
            for point,nodes in tracks:
                for i,idx in nodes:
                    _,z=project(point[None],poses[i],k)
                    rasterize_points(rasters[i],feature(i)[0][idx][None],z)
            dense_count=0
            for n in range(len(training)-3):
                a,b,c=training[n],training[n+1],training[n+3]
                if np.linalg.norm(poses[a][:3,3]-poses[c][:3,3])<.15:
                    continue
                points,pixels=dense_triplet(a,b,c,raw,poses,k,args.stride)
                dense_count+=len(points)
                for i,uv in zip((a,b,c),pixels):
                    _,z=project(points,poses[i],k)
                    rasterize_points(rasters[i],uv,z)
                print(f"Triangulated real RGB views {a}/{b}/{c}: {len(points)} static surface samples",flush=True)
            hashes={}
            for i,raster in rasters.items():
                valid=np.isfinite(raster); raster[~valid]=0
                path=estimate_dir/f"frame_{i:06d}.npz"
                np.savez_compressed(path,depth=raster,valid=valid)
                hashes[path.name]=sha256(path)
            json_write(receipt_path,dict(schema="rgb.geometry_cache/1",hashes=hashes,
                dense_three_view_samples=dense_count,source_fingerprint=fingerprint,physical_contract_verified=False))
            del rasters
        else:
            print("Reusing hash-verified RGB triangulated geometry",flush=True)
        @lru_cache(maxsize=8)
        def frame_loader(i):
            rgb,codes,labels=raw(i)
            with np.load(estimate_dir/f"frame_{i:06d}.npz",allow_pickle=False) as data:
                depth,valid=data["depth"].copy(),data["valid"].copy()
            valid &= labels!=CATEGORIES.index("person")
            return rgb,depth,valid,labels
        fusion=VoxelFusion(k,args.voxel_size,stride=1)
        for i in training:
            rgb,depth,valid,labels=frame_loader(i)
            fusion.integrate(rgb,depth,valid,labels,poses[i],i)
        meshes=fusion.extract()
        bundle=output/"source_bundle"
        if bundle.exists():
            # Preserve every prior editable bundle when regenerating.
            import tempfile
            bundle=Path(tempfile.mkdtemp(prefix="source_bundle_",dir=output))
        else:
            bundle.mkdir()
        video=probe_video(prepared/"original_video.mp4")
        if (video["width"],video["height"],video["frame_count"],video["fps"])!=(1920,1080,720,"60"):
            raise ValueError("Native original video timeline/raster changed")
        scene=make_scene(bundle,meshes,training,frame_loader,poses,{},calibration,rows,k,video,
            dict(poses=prepared/"camera/poses.json"),style="voxel",voxel_size=args.voxel_size)
        scene["reconstruction_metadata"]["provider"]="local_grounding_dino_sam+RGB_three_view+voxel"
        scene["reconstruction_metadata"]["limitations"].append("Native depth inverse failed calibration and was not used; geometry triangulated from RGB.")
        # Only native reserved codes identify sky, never interpret them as metres.
        rgb,codes,labels=raw(training[0]); sky=(codes>=1020)&(np.indices(codes.shape)[0]<h//3)
        if sky.any(): scene["appearance"]["background_color"]=(np.median(rgb[sky],axis=0)/255).tolist()
        json_write(bundle/"scene_schema.json",scene)
        executable=blender_path(args.blender)
        run_pipeline(bundle/"scene_schema.json",prepared/"original_video.mp4",output,blender=executable,
            engine="BLENDER_EEVEE",samples=16,allow_inferred=True,generate_only=True)
        baseline_scene=json.loads(json.dumps(scene))
        for obj in baseline_scene["objects"]:
            obj.pop("mesh_asset"); obj["primitive_type"]="plane" if obj["category"]=="ground" else "box"
            obj["geometry_source"]["point_cloud_path"]=str((bundle/obj["geometry_source"]["point_cloud_path"]).resolve())
            for row in obj["observations"]: row["mask"]["path"]=str((bundle/row["mask"]["path"]).resolve())
        baseline_scene["reconstruction_metadata"]["empirical_validation"]["path"]=str(bundle/"calibration_report.json")
        json_write(bundle/"baseline_scene.json",baseline_scene)
        baseline=output/"baseline"
        run_pipeline(bundle/"baseline_scene.json",prepared/"original_video.mp4",baseline,blender=executable,
            allow_inferred=True,generate_only=True)
        preview_frames=sorted(set(HOLDOUT)|{0,360,719})
        render_previews(output,preview_frames,executable=executable)
        render_previews(baseline,preview_frames,executable=executable)
        def quality_loader(i):
            rgb,codes,labels=raw(i)
            return rgb,np.zeros(codes.shape,np.float32),(codes>0)&(codes<1020),labels
        quality=preview_quality(output,baseline,preview_frames,quality_loader,style="voxel")
        report.update(status="preview_passed" if quality["status"]=="passed" else "quality_rejected",
            quality=quality,voxel_geometry=fusion.report,source_bundle=str(bundle),geometry_cache=str(estimate_dir),mesh_objects=len(scene["objects"]))
        json_write(output/"refinement_report.json",report)
        if quality["status"]!="passed": return 2
        if not args.preview_only:
            run_pipeline(bundle/"scene_schema.json",prepared/"original_video.mp4",output,blender=executable,
                engine="BLENDER_EEVEE",samples=16,allow_inferred=True)
            report.update(status="passed",full_video_rendered=True)
            json_write(output/"refinement_report.json",report)
        return 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=ROOT/"outputs/stage_ef/real_voxel_rgb")
    p.add_argument("--blender",type=Path)
    p.add_argument("--allow-inferred",action="store_true")
    p.add_argument("--preview-only",action="store_true")
    p.add_argument("--calibrate-only",action="store_true")
    p.add_argument("--voxel-size",type=float,default=.2)
    p.add_argument("--stride",type=int,default=4)
    args=p.parse_args(argv)
    if not .05<=args.voxel_size<=1 or args.stride<2: p.error("Bounded voxel size [0.05,1.0] and stride >=2 required")
    try: return run(args)
    except (ValueError,RuntimeError,OSError,KeyError) as error:
        print(f"RGB blockout stopped: {error}",file=__import__('sys').stderr)
        return 1


if __name__=="__main__": raise SystemExit(main())
