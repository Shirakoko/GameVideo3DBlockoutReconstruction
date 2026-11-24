"""Render an accepted RGB voxel Scene JSON without repeating reconstruction."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from blender.pipeline import json_write, run_pipeline, sha256
from reconstruction.refine import refinement_lock


def render(output, *, blender=None, reuse=False):
    output=Path(output).resolve()
    report_path=output/"refinement_report.json"
    report=json.loads(report_path.read_text())
    if report.get("schema")!="refined.run/1" or report.get("geometry_method")!="rgb_three_view" \
            or report.get("calibration",{}).get("status")!="passed" or report.get("quality",{}).get("status")!="passed":
        raise ValueError("RGB geometry calibration and same-source preview must both pass before full rendering")
    if report.get("volume_envelope_inferred"):
        import numpy as np
        from reconstruction.rgb_hull import volume_camera_check
        cache=output/"solid_voxels.npz"
        if not cache.is_file() or sha256(cache)!=report.get("voxel_cache_sha256"):
            raise ValueError("Solid-volume cache is missing or changed")
        scene=json.loads((output/"scene_schema.json").read_text())
        with np.load(cache,allow_pickle=False) as data:
            check=volume_camera_check(data["cells"],report["block_size_metres"],
                [row["world_from_camera"] for row in scene["cameras"][0]["track"]])
        if check["status"]!="passed":
            report.update(status="quality_rejected",full_video_rendered=False,volume_validation=check)
            report["quality"]["image_agreement_status"]=report["quality"].get("image_agreement_status",report["quality"]["status"])
            report["quality"].update(status="failed",volume_validation=check)
            json_write(output/"quality_report.json",report["quality"])
            json_write(report_path,report)
            raise ValueError("Inferred solids enclose the original camera trajectory; full rendering rejected")
    source=Path(report["source_bundle"]).resolve()/"scene_schema.json"
    if output not in source.parents:
        raise ValueError("Source bundle is outside the managed RGB output")
    build=json.loads((output/"pipeline_report.json").read_text())
    for name,digest in build["input_hashes"].items():
        path=Path(name)
        if not path.is_file() or sha256(path)!=digest:
            raise ValueError("Scene/assets/render code changed since the accepted preview; regenerate previews first")
    for name,digest in report["source_hashes"].items():
        if sha256(Path(name))!=digest:
            raise ValueError("Original inputs changed since multi-view validation")
    video=ROOT/"outputs/stage_a/sample_66ad/original_video.mp4"
    with refinement_lock(output):
        result=run_pipeline(source,video,output,blender=blender,engine="BLENDER_EEVEE",samples=16,
            allow_inferred=True,reuse=reuse)
        if result["stage_f"]!="passed" or any(result["media"][name]["frame_count"]!=720 or
                result["media"][name]["fps"]!="60" for name in ("blockout_video.mp4","original_vs_blockout.mp4")):
            raise ValueError("Final MP4 timeline verification failed")
        report.update(status="passed",full_video_rendered=True,artifact_hashes=result["artifact_hashes"],
            completed_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
        json_write(report_path,report)
        return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=ROOT/"outputs/stage_ef/real_voxel_rgb")
    p.add_argument("--blender",type=Path)
    p.add_argument("--reuse",action="store_true")
    args=p.parse_args(argv)
    try:
        render(args.output,blender=args.blender,reuse=args.reuse)
        return 0
    except (ValueError,RuntimeError,OSError,KeyError) as error:
        print(f"RGB render stopped: {error}",file=sys.stderr)
        return 1


if __name__=="__main__": raise SystemExit(main())
