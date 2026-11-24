"""Training-only camera tick-offset diagnostics on existing local caches."""
from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blender.pipeline import json_write, sha256
from reconstruction.refined_calibration import HOLDOUT, epipolar_errors, match_features, pose_track


def main():
    output=ROOT/"outputs/stage_ef/real_refined"
    report=json.loads((output/"refinement_report.json").read_text())
    cache=Path(report["frame_cache"])
    manifest=json.loads((cache/"frames.json").read_text())
    for name,digest in manifest["hashes"].items():
        if sha256(cache/name)!=digest:
            raise ValueError("Frame cache integrity failed")
    @lru_cache(maxsize=70)
    def feature(i):
        candidates=list(cache.glob(f"features_*_{i:06d}.npz"))
        paths=[p for p in candidates if p.with_suffix(".sha256").is_file() and
               p.with_suffix(".sha256").read_text()==sha256(p)]
        if not paths:
            raise ValueError("No verified feature cache")
        path=max(paths,key=lambda p:p.stat().st_mtime_ns)
        with np.load(path,allow_pickle=False) as data:
            return data["points"].copy(),data["descriptors"].copy()
    frames=sorted(int(name[6:12]) for name in manifest["hashes"] if int(name[6:12]) not in HOLDOUT)
    pairs=[]
    for step in (1,2,3,4):
        for a,b in zip(frames,frames[step:]):
            ua,ub=match_features(feature(a),feature(b))
            if len(ua)>=30:
                pairs.append((a,b,ua,ub))
    poses_doc=json.loads((ROOT/"outputs/stage_a/sample_66ad/camera/poses.json").read_text())
    rows=poses_doc["frames"]
    k=np.array([[960.,0,960],[0,960,540],[0,0,1.]])
    candidates=[]
    for field in ("camera_raw","depth_referenced_camera_raw"):
        track=pose_track(rows,100,field=field)
        for offset in range(-8,4):
            errors=np.concatenate([epipolar_errors(ua,ub,track[a+offset],track[b+offset],k)
                for a,b,ua,ub in pairs if 0<=a+offset<len(rows) and 0<=b+offset<len(rows)])
            entry=dict(field=field,offset_ticks=offset,median_epipolar_pixels=float(np.median(errors)),
                p90_epipolar_pixels=float(np.percentile(errors,90)),samples=len(errors))
            print(entry,flush=True)
            candidates.append(entry)
    json_write(output/"alignment_diagnostic.json",dict(schema="refined.alignment_diagnostic/1",
        training_frames=frames,holdouts_excluded=list(HOLDOUT),candidates=candidates,
        source_hashes=report["input_hashes"],physical_contract_verified=False))
    focal_candidates=[]
    for field,offsets in (("camera_raw",(-2,-1,0)),("depth_referenced_camera_raw",(0,1,2))):
        track=pose_track(rows,100,field=field)
        for offset in offsets:
            for focal in np.arange(720.,1345.,48.):
                intrinsic=k.copy(); intrinsic[0,0]=intrinsic[1,1]=focal
                errors=np.concatenate([epipolar_errors(ua,ub,track[a+offset],track[b+offset],intrinsic)
                    for a,b,ua,ub in pairs if 0<=a+offset<len(rows) and 0<=b+offset<len(rows)])
                focal_candidates.append(dict(field=field,offset_ticks=offset,focal_pixels=float(focal),
                    median_epipolar_pixels=float(np.median(errors)),p90_epipolar_pixels=float(np.percentile(errors,90))))
    for entry in sorted(focal_candidates,key=lambda r:r["median_epipolar_pixels"])[:10]:
        print("Focal",entry,flush=True)
    json_write(output/"focal_diagnostic.json",dict(schema="refined.focal_diagnostic/1",
        training_frames=frames,holdouts_excluded=list(HOLDOUT),candidates=focal_candidates,
        source_hashes=report["input_hashes"],physical_contract_verified=False))
    return 0


if __name__=="__main__": raise SystemExit(main())
