"""User-approved RGB-layout voxel completion, NOT physical reconstruction.

The original failed depth receipts are never modified or upgraded. Native
codes provide ordinal shape cues only; an inferred ground plane supplies a
bounded proxy scale. Static geometry and palette remain fixed throughout the
clip. No current video frame is projected into the render.
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

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.optimize import isotonic_regression
from scipy.spatial import cKDTree

from agents.object_detection import confidence
from blender.pipeline import ROOT, blender_path, json_write, run_pipeline, sha256
from reconstruction.refine import HOLDOUT, masked_ssim, refinement_lock, render_previews, verified_frame_cache
from reconstruction.refined_calibration import project
from reconstruction.refined_geometry import split_chunks, write_glb
from reconstruction.refined_segmentation import CATEGORIES
from reconstruction.refined_voxels import palette_mesh, voxel_shell
from reconstruction.rgb_hull import envelope_depth, envelope_raycast_scene, volume_camera_check
from scene_schema import validate_scene

AUTHORIZATION="user-approved RGB-layout completion"


def proxy_lut(codes, distances):
    """Monotone ordinal cue fitted to a hypothesized plane, not a DepthSpec."""
    codes,distances=np.asarray(codes),np.asarray(distances)
    good=np.isfinite(distances)&(distances>=2)&(distances<=70)&(codes>0)&(codes<800)
    codes,distances=codes[good],distances[good]
    if len(codes)<100: raise ValueError("Insufficient ground-plane proxy anchors")
    edges=np.unique(np.quantile(codes,np.linspace(0,1,33)))
    xs,ys,weights=[],[],[]
    for a,b in zip(edges,edges[1:]):
        keep=(codes>=a)&(codes<=b)
        if keep.sum()<5: continue
        x=float(np.median(codes[keep]))
        if xs and x<=xs[-1]: continue
        xs.append(x); ys.append(float(np.median(np.log(distances[keep])))); weights.append(int(keep.sum()))
    if len(xs)<4: raise ValueError("Ground proxy anchors have insufficient ordinal range")
    ys=isotonic_regression(np.asarray(ys),weights=np.asarray(weights)).x
    xs=np.asarray(xs); domain=np.arange(1024)
    logs=np.interp(domain,xs,ys)
    slope=np.clip((ys[-1]-ys[0])/(xs[-1]-xs[0]),.001,.02)
    logs=np.where(domain<xs[0],ys[0]+(domain-xs[0])*slope,logs)
    logs=np.where(domain>xs[-1],ys[-1]+(domain-xs[-1])*slope,logs)
    lut=np.clip(np.exp(logs),2.5,70)
    return lut,dict(method="ordinal codes anchored to an inferred ground plane",anchor_count=len(codes),
        code_knots=xs.tolist(),log_proxy_distance=ys.tolist(),range=[2.5,70],
        inferred=True,physical_inverse_used=False,geometry_accuracy_validated=False)


def exclude_camera_corridor(cells, categories, size, poses, radius=2.5):
    """Reserve a free corridor around every source camera, not just previews."""
    centers=(np.asarray(cells)+.5)*size
    distance=cKDTree(np.asarray(poses)[:,:3,3]).query(centers)[0]
    return (categories==CATEGORIES.index("ground")) | (distance>=radius+np.sqrt(3)*size/2)


def solid_columns(cells, colors, categories, floor_cell, *, max_cells=500000):
    """Ground rocks/buildings; fill only sampled canopy-column intervals."""
    result,shade,owners=[],[],[]
    for category in np.unique(categories):
        use=categories==category
        part,col=cells[use],colors[use]
        xy,inverse=np.unique(part[:,:2],axis=0,return_inverse=True)
        low=np.full(len(xy),100000); high=np.full(len(xy),-100000)
        np.minimum.at(low,inverse,part[:,2]); np.maximum.at(high,inverse,part[:,2])
        sums=np.zeros((len(xy),3)); counts=np.bincount(inverse); np.add.at(sums,inverse,col)
        if CATEGORIES[int(category)] in {"rock","environment","building","barrier","grass","shrub"}:
            low[:]=floor_cell+1
        low=np.maximum(low,floor_cell+1); high=np.maximum(high,low)
        for point,a,b,color in zip(xy,low,high,sums/counts[:,None]):
            z=np.arange(a,b+1,dtype=np.int64)
            result.append(np.column_stack((np.full(len(z),point[0]),np.full(len(z),point[1]),z)))
            shade.append(np.broadcast_to(color,(len(z),3)))
            owners.append(np.full(len(z),category,int))
        if sum(len(r) for r in result)>max_cells: raise ValueError("Proxy cell budget exceeded")
    return tuple(map(np.concatenate,(result,shade,owners)))


def build_voxels(raw, training, poses, k, size):
    ground=float(np.min(np.asarray(poses)[:,2,3])-1.85)
    floor_cell=int(np.floor(ground/size))
    anchors,distances=[],[]
    for i in training:
        rgb,codes,labels=raw(i); h,w=codes.shape
        v,u=np.mgrid[0:h:24,0:w:24]
        rays=np.stack((u,v,np.ones_like(u)),axis=-1)@np.linalg.inv(k).T@poses[i][:3,:3].T
        with np.errstate(divide="ignore",invalid="ignore"):
            z=(ground-poses[i][2,3])/rays[:,:,2]
        use=((labels[v,u]==1)|(labels[v,u]==6))&(rays[:,:,2]<-.06)&(z>=2)&(z<=70)
        anchors.append(codes[v[use],u[use]]); distances.append(z[use])
    lut,curve=proxy_lut(np.concatenate(anchors),np.concatenate(distances))
    records,record_colors=[],[]
    category_colors={}
    for i in training:
        rgb,codes,labels=raw(i); h,w=codes.shape
        for cat in range(8):
            sample=rgb[::12,::12][labels[::12,::12]==cat]
            if len(sample): category_colors.setdefault(cat,[]).append(np.median(sample,axis=0)/255)
        v,u=np.mgrid[0:h:8,0:w:8]
        code,cat=codes[v,u],labels[v,u]
        z=lut[code]
        rays=np.stack((u,v,np.ones_like(u)),axis=-1)@np.linalg.inv(k).T
        points=(rays*z[:,:,None])@poses[i][:3,:3].T+poses[i][:3,3]
        use=(code>0)&(code<800)&(cat!=8)&(cat!=1)&(points[:,:,2]>ground)&(points[:,:,2]<ground+20)
        # Low unclassified/grass regions become grounded coarse landscape.
        cells=np.floor(points[use]/size).astype(np.int64)
        owner=cat[use].copy()
        colors=rgb[v[use],u[use]]/255.
        keep=exclude_camera_corridor(cells,owner,size,poses)
        keys=np.column_stack((cells[keep],owner[keep])); colors=colors[keep]
        unique,inverse,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
        sums=np.zeros((len(unique),3)); np.add.at(sums,inverse,colors)
        records.append(unique); record_colors.append(sums/counts[:,None])
        print(f"Inferred layout cues {i}: {len(unique)} frame cells",flush=True)
    keys=np.concatenate(records); colors=np.concatenate(record_colors)
    unique,inverse,votes=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
    sums=np.zeros((len(unique),3)); np.add.at(sums,inverse,colors)
    keep=votes>=2
    cells,colors,categories=unique[keep,:3],sums[keep]/votes[keep,None],unique[keep,3]
    cells,colors,categories=solid_columns(cells,colors,categories,floor_cell)
    keep=exclude_camera_corridor(cells,categories,size,poses)
    cells,colors,categories=cells[keep],colors[keep],categories[keep]
    # Ground completion is explicitly outside the observed-point hull.
    xy=np.asarray(poses)[:,:2,3]
    low=np.floor((xy.min(0)-35)/size).astype(int); high=np.ceil((xy.max(0)+35)/size).astype(int)
    x,y=np.meshgrid(np.arange(low[0],high[0]+1),np.arange(low[1],high[1]+1),indexing="ij")
    floor=np.column_stack((x.ravel(),y.ravel(),np.full(x.size,floor_cell)))
    ground_color=np.median(category_colors[1],axis=0)
    # Stable palette variation, not a video/image projected onto the floor.
    modulation=1+(.025*np.sin(floor[:,0]*.61)+.02*np.cos(floor[:,1]*.47))
    floor_colors=np.clip(ground_color*modulation[:,None],0,1)
    cells=np.concatenate((floor,cells)); colors=np.concatenate((floor_colors,colors))
    categories=np.concatenate((np.full(len(floor),1),categories))
    # Deterministic one-owner arbitration and global interior-face removal.
    order=np.lexsort((categories,cells[:,2],cells[:,1],cells[:,0]))
    ordered=cells[order]; first=np.r_[True,np.any(np.diff(ordered,axis=0),axis=1)]
    take=order[first]; cells,colors,categories=cells[take],colors[take],categories[take]
    if len(cells)>500000: raise ValueError("Proxy total cell budget exceeded")
    report=dict(curve=curve,ground_plane_height=ground,ground_surface_height=(floor_cell+1)*size,
        floor_completion_cells=len(floor),occupied_cells=len(cells),camera_corridor_radius=2.5,
        physical_geometry_verified=False,inferred=True,hidden_completion="user-approved proxy, not recovered geometry")
    return cells,colors,categories,report


def proxy_coverage(meshes, raw, frames, poses, k):
    scene=envelope_raycast_scene(meshes); rows=[]
    for i in frames:
        rgb,codes,labels=raw(i); h,w=codes.shape
        small_k=k.copy(); small_k[0]*=384/w; small_k[1]*=216/h
        depth,visible=envelope_depth(scene,poses[i],small_k,216,384)
        target=cv2.resize(((codes>0)&(codes<1020)&(labels!=8)).astype(np.uint8),(384,216),interpolation=cv2.INTER_NEAREST)>0
        coverage=float(visible[target].mean())
        close=float(((depth>0)&(depth<1.2))[target].mean())
        rows.append(dict(frame_index=i,environment_coverage=coverage,near_wall_fraction=close))
    passed=min(r["environment_coverage"] for r in rows)>=.45 and np.mean([r["environment_coverage"] for r in rows])>=.65 and max(r["near_wall_fraction"] for r in rows)<=.01
    return dict(status="passed" if passed else "failed",frames=rows,
        thresholds=dict(minimum_frame_coverage=.45,minimum_mean_coverage=.65,maximum_near_wall_fraction=.01),
        evidence_boundary="Proxy coverage/free camera view, not recovered geometry accuracy")


def export_scene(bundle, meshes, base_scene, raw, training, poses, k, size, proxy_receipt):
    assets=bundle/"assets"; assets.mkdir()
    conf=confidence(.35,"inferred","heuristic_rgb_layout","User-approved coarse completion; not measured geometry")
    # Native label/validity planes fit comfortably in memory. Decode each NPZ
    # once, not once per mesh/view; retain no full RGB image in this cache.
    view_labels={}
    for i in training:
        _,codes,labels=raw(i)
        view_labels[i]=(labels,(codes>0)&(codes<1020))
    h,w=view_labels[training[0]][0].shape
    objects=[]
    for category,mesh in meshes.items():
        for key,chunk in split_chunks(mesh,preserve_voxels=True,voxel_size=size):
            points,faces,uv,png=palette_mesh(chunk)
            low,high=points.min(0),points.max(0); center=(low+high)/2; dims=high-low
            if (dims<=1e-6).any(): continue
            name="proxy_"+category+"_"+"_".join(map(str,key))
            asset=assets/(name+".glb"); cloud_path=assets/(name+".ply")
            write_glb(asset,points-center,faces,uv,png)
            cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[::max(1,len(points)//1000)]))
            if not o3d.io.write_point_cloud(str(cloud_path),cloud): raise ValueError("Proxy reference export failed")
            observations=[]; source_frames=[]; visibility=[]
            sample=points[::max(1,len(points)//200)]
            for i in sorted(training,key=lambda i:np.linalg.norm(poses[i][:3,3]-center)):
                pix,z=project(sample,poses[i],k)
                finite=np.isfinite(pix).all(1)&(z>.1)
                xy=np.rint(np.where(finite[:,None],pix,-1)).astype(int)
                inside=finite&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
                if inside.sum()<3: continue
                x,y=xy[inside,0],xy[inside,1]
                labels,valid=view_labels[i]
                mask=np.zeros((h,w),bool)
                rectangle=np.s_[y.min():y.max()+1,x.min():x.max()+1]
                mask[rectangle]=(labels[rectangle]==CATEGORIES.index(category))&valid[rectangle]
                if mask.sum()<4: continue
                yy,xx=np.where(mask); mask_path=assets/f"{name}_frame_{i:06d}_mask.png"
                Image.fromarray(mask.astype(np.uint8)*255).save(mask_path)
                stamp=base_scene["timeline"]["frames"][i]["timestamp_seconds"]
                observations.append(dict(frame_index=i,timestamp_seconds=stamp,camera_id="camera_main",
                    bbox_xyxy=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],raster_size=[w,h],
                    mask=dict(path=str(mask_path.relative_to(bundle)),kind="instance",confidence=conf),
                    position=center.tolist(),dimensions=dims.tolist(),geometry_confidence=conf,point_count=len(points)))
                source_frames.append(i); visibility.append(dict(frame_index=i,timestamp_seconds=stamp,camera_id="camera_main",state="visible"))
                break
            objects.append(dict(object_id=name,category=category,primitive_type="mesh",
                mesh_asset=dict(path=str(asset.relative_to(bundle)),sha256=sha256(asset),format="glb"),
                transform=dict(position=center.tolist(),rotation=[0,0,0,1],scale=[1,1,1]),dimensions=dims.tolist(),parent_id=None,
                is_static=True,classification_confidence=conf,confidence=conf,tracking_confidence=conf,
                geometry_source=dict(kind="heuristic_voxel",point_cloud_path=str(cloud_path.relative_to(bundle)),bounds_type="inferred_proxy_aabb",confidence=conf),
                source_frames=source_frames,observations=observations,visibility=visibility,motion_track=[]))
            print(f"Exported editable {name}: {len(faces)} faces",flush=True)
    scene=json.loads(json.dumps(base_scene)); scene["objects"]=objects
    scene["scene_id"]="user-approved-voxel-proxy-66ad"
    scene["created_at"]=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    calibration=json.loads((ROOT/"outputs/stage_ef/real_voxel_rgb/calibration_report.json").read_text())
    json_write(bundle/"camera_calibration.json",calibration)
    json_write(bundle/"proxy_validation.json",proxy_receipt)
    scene["reconstruction_metadata"].update(provider="local_heuristic_RGB_layout_voxel_completion",
        data_source="REAL LOCAL VIDEO / USER-APPROVED HEURISTIC PROXY / NOT GEOMETRY RECOVERY",
        empirical_validation=dict(path="camera_calibration.json",sha256=sha256(bundle/"camera_calibration.json")),
        heuristic_proxy=dict(path="proxy_validation.json",sha256=sha256(bundle/"proxy_validation.json")),
        limitations=["User approved completion beyond verified geometry on 2026-11-24.",
            "Ordinal depth codes and an inferred ground plane supply proxy layout, NOT the native physical depth inverse.",
            "RGB calibration validates sampled camera correspondences only; it does not certify proxy geometry.",
            "Ground, rock/building interiors and camera-clear corridor are heuristic completions.",
            "Static editable voxels; source palette only, no per-frame projection, PBR recovery or characters.",
            "Camera snapshot/focal binding remains inferred; original timeline, ticks and FPS retained.",
            "Local internal only; delete with restricted source by 2026-12-24 Asia/Shanghai."])
    json_write(bundle/"scene_draft.json",scene)
    validate_scene(scene); json_write(bundle/"scene_schema.json",scene)
    return scene


def render_accepted(output, *, blender=None, reuse=False):
    report=json.loads((output/"proxy_run.json").read_text())
    if report.get("schema")!="heuristic.run/1" or report.get("preview_status")!="passed":
        raise ValueError("Proxy preview/safety checks must pass before rendering")
    for name,digest in report["source_hashes"].items():
        if sha256(Path(name))!=digest: raise ValueError("Proxy original source changed")
    if not report.get("preview_hashes"):
        raise ValueError("Proxy preview hashes are missing")
    for name,digest in report["preview_hashes"].items():
        if sha256(Path(name))!=digest: raise ValueError("Proxy preview changed since acceptance")
    build=json.loads((output/"pipeline_report.json").read_text())
    for name,digest in build["input_hashes"].items():
        if sha256(Path(name))!=digest: raise ValueError("Proxy scene/assets/render code changed; regenerate previews")
    result=run_pipeline(Path(report["source_bundle"])/"scene_schema.json",ROOT/"outputs/stage_a/sample_66ad/original_video.mp4",
        output,blender=blender,engine="BLENDER_EEVEE",samples=16,allow_inferred=True,allow_heuristic=True,reuse=reuse)
    for name in ("blockout_video.mp4","original_vs_blockout.mp4"):
        if result["media"][name]["frame_count"]!=720 or result["media"][name]["fps"]!="60":
            raise ValueError("Proxy MP4 original timeline mismatch")
    report.update(status="passed",full_video_rendered=True,artifact_hashes=result["artifact_hashes"])
    json_write(output/"proxy_run.json",report)
    return result


def refresh_previews(output, *, blender=None):
    """Rebuild unchanged Scene JSON after renderer/verification code changes."""
    report=json.loads((output/"proxy_run.json").read_text())
    source=Path(report["source_bundle"]).resolve()/"scene_schema.json"
    if output not in source.parents: raise ValueError("Proxy source bundle outside managed output")
    for name,digest in report["source_hashes"].items():
        if sha256(Path(name))!=digest: raise ValueError("Original source changed before preview refresh")
    receipt=json.loads((source.parent/"proxy_validation.json").read_text())
    if sha256(output/"proxy_voxels.npz")!=receipt["voxel_sha256"]:
        raise ValueError("Proxy voxels changed before preview refresh")
    scene=json.loads(source.read_text()); validate_scene(scene)
    seed=json.loads((ROOT/"outputs/stage_ef/real_refined/refinement_report.json").read_text())
    cache=Path(seed["frame_cache"]); frames=sorted(set(HOLDOUT)|{0,360,719})
    training=json.loads((source.parent/"camera_calibration.json").read_text())["training_frames"]
    if not verified_frame_cache(cache,sorted(set(training)|set(HOLDOUT))): raise ValueError("Original RGB cache changed")
    executable=blender_path(blender)
    run_pipeline(source,ROOT/"outputs/stage_a/sample_66ad/original_video.mp4",output,blender=executable,
        engine="BLENDER_EEVEE",samples=16,allow_inferred=True,allow_heuristic=True,generate_only=True)
    render_previews(output,frames,executable=executable)
    rows=[]; bg=np.asarray(scene["appearance"]["background_color"])*255
    for i in frames:
        with np.load(cache/f"frame_{i:06d}.npz",allow_pickle=False) as data:
            rgb,codes,labels=data["rgb"],data["codes"],data["labels"]
            rendered=np.asarray(Image.open(output/f"frames/frame_{i+1:06d}.png").convert("RGB"))
            mask=(codes>0)&(codes<1020)&(labels!=8)
            visible=np.max(np.abs(rendered.astype(float)-bg),axis=2)>4
            rows.append(dict(frame_index=i,rendered_environment_coverage=float(visible[mask].mean()),
                masked_ssim_informational=masked_ssim(rgb,rendered,mask)))
    passed=min(r["rendered_environment_coverage"] for r in rows)>=.45 and np.mean([r["rendered_environment_coverage"] for r in rows])>=.65
    report.update(status="preview_passed" if passed else "quality_rejected",preview_status="passed" if passed else "failed",
        full_video_rendered=False,mesh_objects=len(scene["objects"]),preview_frames=rows,
        preview_hashes={str(output/f"frames/frame_{i+1:06d}.png"):sha256(output/f"frames/frame_{i+1:06d}.png") for i in frames})
    json_write(output/"proxy_run.json",report)
    if not passed: raise ValueError("Refreshed proxy preview coverage failed")
    return report


def run(args):
    if not args.allow_heuristic: raise ValueError("Explicit --allow-heuristic required for the user-approved completion")
    output=args.output.resolve()
    if output==ROOT or ROOT/"data" in output.parents or ROOT/"outputs/stage_a" in output.parents:
        raise ValueError("Independent proxy output required")
    if args.refresh_previews:
        with refinement_lock(output): return refresh_previews(output,blender=args.blender)
    if args.render_only:
        with refinement_lock(output): return render_accepted(output,blender=args.blender,reuse=args.reuse)
    if output.exists() and any(output.iterdir()) and not (output/"proxy_run.json").is_file():
        raise ValueError("Unmanaged proxy output; choose an independent directory")
    output.mkdir(parents=True,exist_ok=True)
    with refinement_lock(output):
        base=ROOT/"outputs/stage_ef/real_voxel_rgb"
        source=json.loads((base/"refinement_report.json").read_text())
        if source["calibration"]["status"]!="passed": raise ValueError("Frozen RGB camera calibration required")
        for name,digest in source["source_hashes"].items():
            if sha256(Path(name))!=digest: raise ValueError("Original inputs changed")
        seed=json.loads((ROOT/"outputs/stage_ef/real_refined/refinement_report.json").read_text())
        cache=Path(seed["frame_cache"]); training=source["calibration"]["training_frames"]
        if not verified_frame_cache(cache,sorted(set(training)|set(HOLDOUT))): raise ValueError("Selected frame cache failed hashes")
        @lru_cache(maxsize=8)
        def raw(i):
            with np.load(cache/f"frame_{i:06d}.npz",allow_pickle=False) as data:
                return data["rgb"].copy(),data["codes"].copy(),data["labels"].copy()
        scene=json.loads((base/"scene_schema.json").read_text())
        poses=np.asarray([r["world_from_camera"] for r in scene["cameras"][0]["track"]])
        k=np.asarray(scene["cameras"][0]["track"][0]["intrinsics"])
        report=dict(schema="heuristic.run/1",status="running",authorization=AUTHORIZATION,
            source_hashes=source["source_hashes"],geometry_accuracy_validated=False,full_video_rendered=False,
            block_size_metres=args.voxel_size,data_policy="Local internal only; delete by 2026-12-24 Asia/Shanghai")
        previous=json.loads((output/"proxy_run.json").read_text()) if (output/"proxy_run.json").is_file() else {}
        if args.resume and (previous.get("source_hashes")!=source["source_hashes"]
                or previous.get("block_size_metres")!=args.voxel_size):
            raise ValueError("Resume source/voxel parameters differ from the interrupted run")
        json_write(output/"proxy_run.json",report)
        prior_path=output/"proxy_validation.json"
        if args.resume and prior_path.is_file():
            prior=json.loads(prior_path.read_text())
            if sha256(output/"proxy_voxels.npz")!=prior.get("voxel_sha256"):
                raise ValueError("Interrupted proxy voxel cache changed")
            if prior.get("authorization")!=AUTHORIZATION:
                raise ValueError("Unapproved proxy cache")
            with np.load(output/"proxy_voxels.npz",allow_pickle=False) as data:
                cells,colors,categories=data["cells"].copy(),data["colors"].copy(),data["categories"].copy()
            layout=prior["layout"]
            if "block_size_metres" in prior and args.voxel_size!=prior["block_size_metres"]:
                raise ValueError("Resume must retain the original proxy block size")
            print("Resumed SHA-verified inferred voxels; rerunning clearance and coverage",flush=True)
        else:
            cells,colors,categories,layout=build_voxels(raw,training,poses,k,args.voxel_size)
            np.savez_compressed(output/"proxy_voxels.npz",cells=cells,colors=colors,categories=categories)
        occupied={tuple(c) for c in cells}
        meshes={CATEGORIES[int(cat)]:voxel_shell(cells[categories==cat],colors[categories==cat],args.voxel_size,occupied)
            for cat in np.unique(categories)}
        frames=sorted(set(HOLDOUT)|{0,360,719})
        safety=volume_camera_check(cells,args.voxel_size,poses)
        coverage=proxy_coverage(meshes,raw,frames,poses,k)
        print(f"Proxy clearance: {safety['inside_frame_count']} camera intersections; coverage: {coverage}",flush=True)
        passed=safety["status"]=="passed" and coverage["status"]=="passed"
        receipt=dict(schema="heuristic.proxy/1",status="passed" if passed else "failed",authorization=AUTHORIZATION,
            geometry_accuracy_validated=False,camera_inside_frame_count=safety["inside_frame_count"],coverage_status=coverage["status"],
            camera_sha256=hashlib.sha256(json.dumps(scene["cameras"],sort_keys=True).encode()).hexdigest(),
            volume=safety,coverage=coverage,layout=layout,voxel_sha256=sha256(output/"proxy_voxels.npz"))
        receipt["block_size_metres"]=args.voxel_size
        json_write(output/"proxy_validation.json",receipt)
        report.update(status="geometry_generated" if passed else "quality_rejected",validation=receipt)
        json_write(output/"proxy_run.json",report)
        if not passed: raise ValueError("Proxy camera clearance/coverage failed")
        bundle=Path(tempfile.mkdtemp(prefix="source_bundle_",dir=output))
        report["source_bundle"]=str(bundle); json_write(output/"proxy_run.json",report)
        export_scene(bundle,meshes,scene,raw,training,poses,k,args.voxel_size,receipt)
        executable=blender_path(args.blender)
        run_pipeline(bundle/"scene_schema.json",ROOT/"outputs/stage_a/sample_66ad/original_video.mp4",output,
            blender=executable,engine="BLENDER_EEVEE",samples=16,allow_inferred=True,allow_heuristic=True,generate_only=True)
        render_previews(output,frames,executable=executable)
        rows=[]
        bg=np.asarray(scene["appearance"]["background_color"])*255
        for i in frames:
            rgb,codes,labels=raw(i)
            rendered=np.asarray(Image.open(output/f"frames/frame_{i+1:06d}.png").convert("RGB"))
            mask=(codes>0)&(codes<1020)&(labels!=8)
            visible=np.max(np.abs(rendered.astype(float)-bg),axis=2)>4
            rows.append(dict(frame_index=i,rendered_environment_coverage=float(visible[mask].mean()),
                masked_ssim_informational=masked_ssim(rgb,rendered,mask)))
        passed=min(r["rendered_environment_coverage"] for r in rows)>=.45 and np.mean([r["rendered_environment_coverage"] for r in rows])>=.65
        report.update(status="preview_passed" if passed else "quality_rejected",preview_status="passed" if passed else "failed",
            source_bundle=str(bundle),mesh_objects=len(json.loads((bundle/"scene_schema.json").read_text())["objects"]),preview_frames=rows)
        report["preview_hashes"]={str(output/f"frames/frame_{i+1:06d}.png"):sha256(output/f"frames/frame_{i+1:06d}.png") for i in frames}
        json_write(output/"proxy_run.json",report)
        if not passed: raise ValueError("Background rendered proxy coverage failed")
        if not args.preview_only: render_accepted(output,blender=executable)
        return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--allow-heuristic",action="store_true")
    p.add_argument("--output",type=Path,default=ROOT/"outputs/stage_ef/heuristic_voxel_proxy")
    p.add_argument("--voxel-size",type=float,default=.5)
    p.add_argument("--preview-only",action="store_true")
    p.add_argument("--render-only",action="store_true")
    p.add_argument("--refresh-previews",action="store_true",help="Rebuild the unchanged Scene JSON and previews with current Blender code")
    p.add_argument("--reuse",action="store_true")
    p.add_argument("--resume",action="store_true",help="Resume a hash-verified interrupted voxel export, with the same block size")
    p.add_argument("--blender",type=Path)
    args=p.parse_args(argv)
    if not .25<=args.voxel_size<=1: p.error("Proxy block size must be [0.25,1.0]")
    try: run(args); return 0
    except (ValueError,RuntimeError,OSError,KeyError) as error:
        report_path=args.output.resolve()/"proxy_run.json"
        if report_path.is_file():
            report=json.loads(report_path.read_text())
            report.update(status="failed" if report.get("status")!="quality_rejected" else "quality_rejected",full_video_rendered=False,error_type=type(error).__name__)
            json_write(report_path,report)
        print(f"Heuristic proxy stopped: {error}",file=__import__('sys').stderr); return 1


if __name__=="__main__": raise SystemExit(main())
