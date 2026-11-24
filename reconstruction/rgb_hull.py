"""Bounded multi-view semantic volume envelopes for an RGB voxel blockout.

No new objects or assets outside observed source bounds. Solid interiors are
explicitly inferred, not claimed as measured or as hidden-surface recovery.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, Delaunay

from blender.pipeline import ROOT, blender_path, json_write, run_pipeline, sha256
from blender.media import probe_video
from reconstruction.coordinate_transform import transform_points
from reconstruction.refined_calibration import HOLDOUT, project
from reconstruction.refined_voxels import voxel_shell
from reconstruction.refined_segmentation import CATEGORIES
from reconstruction.refine import make_scene, preview_quality, refinement_lock, render_previews, verified_frame_cache


def envelope_raycast_scene(meshes):
    """Visibility of the inferred mesh, never a measured depth substitute."""
    scene=o3d.t.geometry.RaycastingScene(nthreads=4)
    for mesh in meshes.values():
        if len(mesh.triangles):
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def envelope_depth(scene, pose, k, height, width):
    """Raycast complete faces instead of sparse voxel centres.

    Rays have camera-space z=1, so t_hit is optical z (not range).
    Row batching bounds memory on full-resolution source frames.
    """
    depth=np.zeros((height,width),np.float32)
    inverse=np.linalg.inv(k)
    for first in range(0,height,64):
        v,u=np.mgrid[first:min(first+64,height),0:width]
        pixels=np.stack((u,v,np.ones_like(u)),axis=-1)
        directions=(pixels@inverse.T)@pose[:3,:3].T
        origins=np.broadcast_to(pose[:3,3],directions.shape)
        rays=np.concatenate((origins,directions),axis=-1).astype(np.float32)
        hits=scene.cast_rays(o3d.core.Tensor(rays),nthreads=4)["t_hit"].numpy()
        keep=np.isfinite(hits)&(hits>.1)&(hits<=80)
        depth[first:first+len(v)]=np.where(keep,hits,0)
    return depth,depth>0


def volume_camera_check(cells, size, poses):
    """A generated solid must not enclose the source camera trajectory."""
    occupied={tuple(c) for c in np.asarray(cells,np.int64)}
    blocked=[i for i,pose in enumerate(poses)
        if tuple(np.floor(np.asarray(pose)[:3,3]/size).astype(np.int64)) in occupied]
    return dict(status="failed" if blocked else "passed",camera_inside_frames=blocked,
        checked_frames=len(poses),inside_frame_count=len(blocked),
        evidence_boundary="Camera centres inside inferred occupied cells; not physical geometry certification")


def carve_envelope(cells, size, category, views, raw, poses, k):
    centers=(cells+.5)*size
    support=np.zeros(len(cells),np.int32)
    sky=np.zeros(len(cells),np.int32)
    colors=np.zeros((len(cells),3))
    for i in views:
        rgb,codes,labels=raw(i)
        uv,z=project(centers,poses[i],k)
        h,w=codes.shape
        finite=np.isfinite(uv).all(1)&(z>.1)&(z<=80)
        xy=np.rint(np.where(finite[:,None],uv,-1)).astype(int)
        inside=finite&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        x,y=np.clip(xy[:,0],0,w-1),np.clip(xy[:,1],0,h-1)
        # Four-pixel mask tolerance accounts for predicted silhouette edges;
        # it does not change the independent RGB camera validation thresholds.
        mask=cv2.dilate((labels==CATEGORIES.index(category)).astype(np.uint8),np.ones((9,9),np.uint8))>0
        foreground=(codes[y,x]>0)&(codes[y,x]<1020)&(labels[y,x]!=CATEGORIES.index("person"))
        good=inside&foreground&mask[y,x]
        support+=good
        colors[good]+=rgb[y[good],x[good]]/255.
        sky+=inside&(codes[y,x]>=1020)
    keep=(support>=3)&(sky<=1)
    return cells[keep],colors[keep]/np.maximum(support[keep,None],1),support[keep]


def ground_plane(points):
    if len(points)<30: return None
    o3d.utility.random.seed(0)
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    model,inliers=cloud.segment_plane(distance_threshold=.25,ransac_n=3,num_iterations=1000)
    model=np.asarray(model,float)
    if len(inliers)<30 or len(inliers)/len(points)<.35 or abs(model[2])<.7: return None
    support=points[inliers]
    try: footprint=Delaunay(support[:,:2])
    except Exception: return None
    return model,footprint,len(inliers)


def run(args):
    if not args.allow_inferred:
        raise ValueError("Solid volume envelopes require explicit --allow-inferred")
    base=ROOT/"outputs/stage_ef/real_voxel_rgb"
    source_scene=json.loads((base/"scene_schema.json").read_text())
    source_report=json.loads((base/"refinement_report.json").read_text())
    seed=json.loads((ROOT/"outputs/stage_ef/real_refined/refinement_report.json").read_text())
    if source_report["calibration"]["status"]!="passed": raise ValueError("RGB camera validation required")
    for name,digest in source_report["source_hashes"].items():
        if sha256(Path(name))!=digest: raise ValueError("Original source changed")
    output=args.output.resolve()
    protected=(base,ROOT/"outputs/stage_a/sample_66ad",ROOT/"data/input/samples")
    if any(output==path or path in output.parents for path in protected):
        raise ValueError("Choose an independent solid-voxel output")
    if output.is_dir() and any(output.iterdir()) and not (output/"refinement_report.json").is_file():
        raise ValueError("Unmanaged output directory")
    output.mkdir(parents=True,exist_ok=True)
    with refinement_lock(output):
        cache=Path(seed["frame_cache"])
        training=source_report["calibration"]["training_frames"]
        indices=sorted(set(training)|set(HOLDOUT))
        if not verified_frame_cache(cache,indices): raise ValueError("Source cache failed verification")
        camera=source_scene["cameras"][0]
        poses=[np.asarray(row["world_from_camera"],float) for row in camera["track"]]
        k=np.asarray(camera["track"][0]["intrinsics"],float)
        @lru_cache(maxsize=10)
        def raw(i):
            with np.load(cache/f"frame_{i:06d}.npz",allow_pickle=False) as data:
                return data["rgb"].copy(),data["codes"].copy(),data["labels"].copy()
        size=args.voxel_size
        all_cells,all_colors,all_categories,all_votes=[],[],[],[]
        ground_points=[]
        h,w=raw(0)[1].shape
        for obj in source_scene["objects"]:
            center=np.asarray(obj["transform"]["position"])
            dimensions=np.asarray(obj["dimensions"])
            low,high=center-dimensions/2,center+dimensions/2
            points=np.asarray(o3d.io.read_point_cloud(obj["geometry_source"]["point_cloud_path"]).points)
            if obj["category"]=="ground":
                ground_points.append(points)
            ranges=[np.arange(int(np.ceil(low[j]/size-.5)),int(np.floor(high[j]/size-.5))+1) for j in range(3)]
            if any(not len(r) for r in ranges): continue
            cells=np.stack(np.meshgrid(*ranges,indexing="ij"),-1).reshape(-1,3)
            if len(cells)>100000: raise ValueError("Envelope candidate budget exceeded")
            # Bounded nearest views, plus the actual original representative view.
            views=sorted(set(obj["source_frames"]+sorted(training,key=lambda i:np.linalg.norm(poses[i][:3,3]-center))[:10]))
            views=[i for i in views if i not in HOLDOUT]
            cells,colors,votes=carve_envelope(cells,size,obj["category"],views,raw,poses,k)
            if len(cells):
                all_cells.append(cells); all_colors.append(colors); all_votes.append(votes)
                all_categories.append(np.full(len(cells),CATEGORIES.index(obj["category"]),int))
            print(f"Carved observed {obj['object_id']}: {len(cells)} solid-envelope voxels",flush=True)
        plane=ground_plane(np.concatenate(ground_points)) if ground_points else None
        plane_report=dict(status="unsupported")
        if plane is not None:
            model,footprint,count=plane
            frame_cells,frame_colors=[],[]
            for i in training:
                rgb,codes,labels=raw(i)
                v,u=np.mgrid[0:h:8,0:w:8]
                good=(labels[v,u]==CATEGORIES.index("ground"))&(codes[v,u]>0)&(codes[v,u]<1020)
                uv=np.column_stack((u[good],v[good]))
                rays=np.column_stack((uv,np.ones(len(uv))))@np.linalg.inv(k).T
                rays=rays@poses[i][:3,:3].T
                denominator=rays@model[:3]
                with np.errstate(divide="ignore",invalid="ignore"):
                    distance=-(poses[i][:3,3]@model[:3]+model[3])/denominator
                keep=np.isfinite(distance)&(distance>.1)&(distance<80)&(np.abs(denominator)>.03)
                points=poses[i][:3,3]+rays[keep]*distance[keep,None]
                uv=uv[keep]
                keep=footprint.find_simplex(points[:,:2])>=0
                cells=np.floor(points[keep]/size).astype(int)
                colors=rgb[uv[keep,1],uv[keep,0]]/255.
                unique,inverse,counts=np.unique(cells,axis=0,return_inverse=True,return_counts=True)
                sums=np.zeros((len(unique),3)); np.add.at(sums,inverse,colors)
                frame_cells.append(unique); frame_colors.append(sums/counts[:,None])
            cells=np.concatenate(frame_cells); colors=np.concatenate(frame_colors)
            unique,inverse,counts=np.unique(cells,axis=0,return_inverse=True,return_counts=True)
            sums=np.zeros((len(unique),3)); np.add.at(sums,inverse,colors)
            keep=counts>=2
            if keep.any():
                all_cells.append(unique[keep]); all_colors.append(sums[keep]/counts[keep,None]); all_votes.append(counts[keep])
                all_categories.append(np.full(keep.sum(),CATEGORIES.index("ground"),int))
            plane_report=dict(status="inferred",plane=model.tolist(),inlier_points=count,
                occupied_cells=int(keep.sum()),support="Ground-label RGB rays only inside the measured ground-point XY hull")
            print(f"Observed-supported ground footprint: {int(keep.sum())} cells",flush=True)
        if not all_cells: raise ValueError("No multi-view constrained solid envelopes")
        cells,colors,categories,votes=map(np.concatenate,(all_cells,all_colors,all_categories,all_votes))
        order=np.lexsort((-votes,cells[:,2],cells[:,1],cells[:,0]))
        ordered=cells[order]; first=np.r_[True,np.any(np.diff(ordered,axis=0),axis=1)]
        chosen=order[first]; cells,colors,categories=cells[chosen],colors[chosen],categories[chosen]
        if len(cells)>400000: raise ValueError("Solid voxel budget exceeded; increase block size")
        volume_validation=volume_camera_check(cells,size,poses)
        occupied={tuple(c) for c in cells}
        meshes={CATEGORIES[int(cat)]:voxel_shell(cells[categories==cat],colors[categories==cat],size,occupied)
            for cat in np.unique(categories)}
        np.savez_compressed(output/"solid_voxels.npz",cells=cells,colors=colors,categories=categories)
        raycast_scene=envelope_raycast_scene(meshes)
        estimate_dir=output/"surface_estimates"; estimate_dir.mkdir(exist_ok=True)
        for i in training:
            raster,valid=envelope_depth(raycast_scene,poses[i],k,h,w)
            np.savez_compressed(estimate_dir/f"frame_{i:06d}.npz",depth=raster,valid=valid)
            print(f"Raycast inferred envelope visibility {i}: {int(valid.sum())} pixels",flush=True)
        @lru_cache(maxsize=8)
        def loader(i):
            rgb,codes,labels=raw(i)
            with np.load(estimate_dir/f"frame_{i:06d}.npz",allow_pickle=False) as data:
                return rgb,data["depth"].copy(),data["valid"].copy(),labels
        calibration=source_report["calibration"].copy()
        calibration["geometry_method"]="RGB_multiview_semantic_volume_envelopes"
        calibration["volume_envelope_inferred"]=True
        json_write(output/"calibration_report.json",calibration)
        report=dict(schema="refined.run/1",status="geometry_generated",geometry_method="rgb_three_view",
            geometry_representation="multi-view semantic solid-envelope voxels",volume_envelope_inferred=True,
            depth_inverse_used=False,physical_contract_verified=False,source_hashes=source_report["source_hashes"],
            calibration=calibration,block_size_metres=size,occupied_cells=len(cells),ground_footprint=plane_report,
            full_video_rendered=False,visibility_depth="raycast of inferred solid faces; not measured source depth",
            voxel_cache_sha256=sha256(output/"solid_voxels.npz"),
            volume_validation=volume_validation,
            data_policy="Local internal only; delete by 2026-12-24 Asia/Shanghai")
        json_write(output/"refinement_report.json",report)
        # Preserve old source assets and receipts on every reconstruction rerun.
        bundle=Path(tempfile.mkdtemp(prefix="source_bundle_",dir=output))
        video=probe_video(ROOT/"outputs/stage_a/sample_66ad/original_video.mp4")
        source_rows=[dict(timestamp_seconds=row["timestamp_seconds"],original_tick=row["original_tick"])
            for row in source_scene["timeline"]["frames"]]
        scene=make_scene(bundle,meshes,training,loader,poses,{},calibration,source_rows,k,video,
            dict(poses=ROOT/"outputs/stage_a/sample_66ad/camera/poses.json"),style="voxel",voxel_size=size)
        scene["appearance"]=source_scene["appearance"]
        meta=scene["reconstruction_metadata"]
        meta["provider"]="local_RGB_multiview+semantic_visual_envelope+voxel"
        meta["limitations"]=[s for s in meta["limitations"] if "no solid interior" not in s.lower()]
        meta["limitations"].append("Solid volume envelopes are inferred from training silhouettes inside observed bounds; interiors are not measured or certified hidden geometry.")
        meta["limitations"].append("Ground plane is an inferred coarse approximation only inside measured point support; weak/unobserved regions can remain incomplete.")
        json_write(bundle/"scene_schema.json",scene)
        executable=blender_path(args.blender)
        original=ROOT/"outputs/stage_a/sample_66ad/original_video.mp4"
        run_pipeline(bundle/"scene_schema.json",original,output,blender=executable,engine="BLENDER_EEVEE",samples=16,
            allow_inferred=True,generate_only=True)
        baseline_scene=json.loads(json.dumps(scene))
        for obj in baseline_scene["objects"]:
            obj.pop("mesh_asset"); obj["primitive_type"]="plane" if obj["category"]=="ground" else "box"
            obj["geometry_source"]["point_cloud_path"]=str((bundle/obj["geometry_source"]["point_cloud_path"]).resolve())
            for row in obj["observations"]: row["mask"]["path"]=str((bundle/row["mask"]["path"]).resolve())
        baseline_scene["reconstruction_metadata"]["empirical_validation"]["path"]=str(bundle/"calibration_report.json")
        json_write(bundle/"baseline_scene.json",baseline_scene)
        baseline=output/"baseline"
        run_pipeline(bundle/"baseline_scene.json",original,baseline,blender=executable,allow_inferred=True,generate_only=True)
        frames=sorted(set(HOLDOUT)|{0,360,719})
        render_previews(output,frames,executable=executable); render_previews(baseline,frames,executable=executable)
        def quality_loader(i):
            rgb,codes,labels=raw(i)
            return rgb,np.zeros((h,w),np.float32),(codes>0)&(codes<1020),labels
        quality=preview_quality(output,baseline,frames,quality_loader,style="voxel")
        quality["image_agreement_status"]=quality["status"]
        quality["volume_validation"]=volume_validation
        if volume_validation["status"]!="passed": quality["status"]="failed"
        json_write(output/"quality_report.json",quality)
        report.update(status="preview_passed" if quality["status"]=="passed" else "quality_rejected",quality=quality,
            source_bundle=str(bundle),mesh_objects=len(scene["objects"]))
        json_write(output/"refinement_report.json",report)
        return 0 if quality["status"]=="passed" else 2


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=ROOT/"outputs/stage_ef/real_voxel_solid")
    p.add_argument("--voxel-size",type=float,default=.4)
    p.add_argument("--blender",type=Path)
    p.add_argument("--allow-inferred",action="store_true")
    args=p.parse_args(argv)
    if not .2<=args.voxel_size<=1: p.error("Solid block size [0.2,1.0] required")
    try: return run(args)
    except (ValueError,RuntimeError,OSError,KeyError) as error:
        print(f"RGB solid blockout stopped: {error}",file=__import__('sys').stderr); return 1


if __name__=="__main__": raise SystemExit(main())
