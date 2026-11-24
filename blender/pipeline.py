"""Stage E/F orchestration: isolated runs, background subprocesses, receipts."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from zoneinfo import ZoneInfo

from blender.media import encode_outputs, probe_video
from blender.planner import prepare_plan, read_json

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ["scene.blend", "blockout_video.mp4", "original_video.mp4", "original_vs_blockout.mp4",
             "build_plan.json", "frame_mapping.json", "build_report.json", "saved_scene_verification.json",
             "pipeline_report.json", "build.log", "render.log", "source_video.probe.log",
             "encode_blockout.log", "encode_original.log", "encode_comparison.log",
             "original_video.mp4.probe.log", "blockout_video.mp4.probe.log", "original_vs_blockout.mp4.probe.log"]


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blender_path(configured=None):
    configured = configured or os.environ.get("STAGE_EF_BLENDER")
    if configured:
        path = Path(configured).resolve()
        if not path.is_file():
            raise ValueError("Configured Blender executable does not exist")
        return str(path)
    found = shutil.which("blender")
    if found:
        return found
    candidates = sorted(Path("C:/Program Files/Blender Foundation").glob("Blender */blender.exe"), reverse=True)
    if candidates:
        return str(candidates[0])
    raise ValueError("Blender not found; use --blender or STAGE_EF_BLENDER")


@contextmanager
def output_lock(output):
    lock = output/".stage_ef.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise ValueError(f"Output is locked: {lock}; concurrent jobs cannot share an output directory") from error
    with os.fdopen(descriptor, "w") as stream:
        stream.write(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink()


def background(executable, script, workspace, *, blend=None, verify_only=False, timeout=3600):
    name = "render" if blend else "build"
    args = [executable, "--background", "--factory-startup", "--threads", "4"]
    if blend:
        args.append(str(blend))
    args += ["--python-exit-code", "1", "--python", str(script), "--", str(workspace/"build_plan.json"), str(workspace)]
    if verify_only:
        args.append("--verify-only")
    with (workspace/(name+".log")).open("w", encoding="utf-8") as log:
        result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        tail = (workspace/(name+".log")).read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"Blender {name} exited {result.returncode}; see {workspace/(name+'.log')}\n{tail}")


def run_pipeline(scene_json, original_video, output, *, blender=None, camera_poses=None, point_cloud=None,
                 active_camera=None, resolution=None, engine="BLENDER_WORKBENCH", samples=16,
                 video_start_frame=0, video_frame_count=None, generate_only=False, reuse=False, timeout=3600):
    output = Path(output).resolve()
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("Positive integer process timeout required")
    scene_json, original_video = Path(scene_json).resolve(), Path(original_video).resolve()
    if any(output/name in {scene_json, original_video} for name in ARTIFACTS):
        raise ValueError("Output artifacts would overwrite an input file; choose a dedicated output directory")
    executable = blender_path(blender)
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        receipt_path = output/"pipeline_report.json"
        previous = read_json(receipt_path) if receipt_path.is_file() else {}
        if previous and previous.get("schema") != "stage_ef.receipt/1":
            raise ValueError("Existing pipeline_report.json is not owned by Stage E/F")
        owned = set(previous.get("owned_files", []))
        for name in ARTIFACTS:
            if (output/name).exists() and name not in owned:
                raise ValueError(f"Refusing to overwrite unmanaged file: {output/name}")
        workspace = Path(tempfile.mkdtemp(prefix=".stage_ef_run_", dir=output)).resolve()
        try:
            print("Stage E/F: inspecting every original video PTS", flush=True)
            video = probe_video(original_video, start_frame=video_start_frame, frame_count=video_frame_count,
                                log_path=workspace/"source_video.probe.log")
            plan = prepare_plan(scene_json, video, resolution=resolution, camera_poses=camera_poses,
                                active_camera=active_camera, engine=engine, samples=samples, point_cloud=point_cloud)
            source_files = [scene_json, original_video, *map(Path, plan["reference_sources"])]
            if camera_poses:
                source_files.append(Path(camera_poses).resolve())
            if any(output/name in {p.resolve() for p in source_files} for name in ARTIFACTS):
                raise ValueError("Output artifact would overwrite a Camera Pose/point cloud input")
            # Bind code and schema too; changed implementation cannot reuse an
            # old binary merely because source inputs happen to be unchanged.
            source_files += sorted((ROOT/"blender").glob("*.py"))
            source_files += [ROOT/"config/scene_schema.json", ROOT/"config/camera_pose_schema.json", ROOT/"scene_schema/validation.py", ROOT/"reconstruction/coordinate_transform.py",
                             ROOT/"stage_a/media.py"]
            inputs = {str(p.resolve()): sha256(p) for p in source_files}
            parameters = dict(resolution=plan["resolution"], engine=engine, samples=samples, active_camera=plan["active_camera"],
                              video_start_frame=video_start_frame, video_frame_count=video["frame_count"], generate_only=generate_only,
                              blender_path=executable, blender_version=subprocess.check_output([executable, "--version"], text=True).splitlines()[0])
            fingerprint = hashlib.sha256(json.dumps(dict(inputs=inputs, parameters=parameters), sort_keys=True).encode()).hexdigest()
            if reuse and previous.get("status") == "passed" and previous.get("fingerprint") == fingerprint:
                if all((output/name).is_file() and sha256(output/name) == digest for name, digest in previous["artifact_hashes"].items()):
                    print(f"Stage E/F: verified matching input/code/output hashes; reusing {output}", flush=True)
                    return previous
            json_write(workspace/"build_plan.json", plan)
            json_write(workspace/"frame_mapping.json", dict(fps=plan["fps"], duration_seconds=video["duration_seconds"],
                start_seconds=plan["frame_mapping"][0]["timestamp_seconds"], frames=plan["frame_mapping"],
                tick_policy="Original ticks only where supplied. Interpolated/render-only frames have null, never fabricated."))
            print(f"Stage E: generating {len(plan['objects'])} primitives / {len(plan['cameras'])} cameras / {plan['frame_end']} frames", flush=True)
            background(executable, ROOT/"blender/scene_generator.py", workspace, timeout=timeout)
            if not (workspace/"scene.blend").is_file() or (workspace/"scene.blend").stat().st_size < 12:
                raise ValueError("Blender did not save a scene file")
            with (workspace/"scene.blend").open("rb") as saved:
                signature = saved.read(7)
                # Blender 5.x may use Zstandard containers even when the
                # save operator requests no compression. Reopen verification
                # is the authoritative check for either supported container.
                if signature != b"BLENDER" and signature[:4] not in {b"\x28\xb5\x2f\xfd", b"\x1f\x8b\x08\x00"}:
                    raise ValueError("Invalid Blender/container file signature")
            print("Stage E/F: reopening saved scene for numerical verification"+ (" and background rendering" if not generate_only else ""), flush=True)
            background(executable, ROOT/"blender/render.py", workspace, blend=workspace/"scene.blend",
                       verify_only=generate_only, timeout=timeout)
            media = {}
            if not generate_only:
                print("Stage F: encoding and checking original / blockout / split-screen MP4", flush=True)
                media = encode_outputs(workspace, video, plan["resolution"])
            generated = [name for name in ARTIFACTS if name != "pipeline_report.json" and (workspace/name).is_file()]
            hashes = {name: sha256(workspace/name) for name in generated}
            # Receipt paths refer to committed files, not the disposable run.
            for name, report in media.items():
                report["path"] = str(output/name)
            report = dict(schema="stage_ef.receipt/1", status="passed", stage_e="passed", stage_f="not_requested" if generate_only else "passed",
                created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), fingerprint=fingerprint,
                input_hashes=inputs, parameters=parameters, source_video=video, media=media,
                scene_verification=read_json(workspace/"saved_scene_verification.json"), artifact_hashes=hashes,
                owned_files=sorted(owned | set(generated) | {"pipeline_report.json"}), output_directory=str(output),
                data_source=plan["data_source"], real_session_accuracy_validated=False,
                limitations=["Sparse camera/object samples are interpolated, not newly measured poses.",
                    "Observed-surface primitive bounds do not recover hidden volume or articulated motion.",
                    "not_detected does not hide objects; no inferred occlusion.",
                    "Reference vertices are saved in .blend but excluded from render.",
                    "Videos are silent; audio reconstruction is not part of Stage E/F."])
            json_write(workspace/"pipeline_report.json", report)
            # Publish receipt last. Existing owned outputs are recoverable by
            # regenerating the exact same input/code; unrelated files stay put.
            for name in [*generated, "pipeline_report.json"]:
                os.replace(workspace/name, output/name)
            print(f"Stage E/F passed: {output}", flush=True)
            return report
        except Exception as error:
            json_write(workspace/"failure.json", dict(status="failed", error=str(error), outputs_published=False))
            # Preserve the isolated failed run and its complete logs for diagnosis.
            print(f"Stage E/F failed; prior outputs unchanged; logs: {workspace}", flush=True)
            raise
        finally:
            # Exact resolved mkdtemp child, never an input, workspace root or
            # computed broad target. Failed runs are intentionally preserved.
            if not (workspace/"failure.json").exists():
                if workspace.parent != output or not workspace.name.startswith(".stage_ef_run_"):
                    raise ValueError("Unsafe staging cleanup target")
                shutil.rmtree(workspace)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene_json", type=Path, nargs="?")
    p.add_argument("--original-video", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--batch", type=Path, help="stage_ef.batch/1 JSON; each job gets its own output directory")
    p.add_argument("--blender", type=Path)
    p.add_argument("--camera-poses", type=Path, help="Explicit verified stage_e.camera_poses/1 (dense original track)")
    p.add_argument("--point-cloud", type=Path, help="Optional fused PLY; otherwise object source PLYs")
    p.add_argument("--active-camera")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--engine", choices=["BLENDER_WORKBENCH", "BLENDER_EEVEE", "CYCLES"], default="BLENDER_WORKBENCH")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--video-start-frame", type=int, default=0)
    p.add_argument("--video-frame-count", type=int)
    p.add_argument("--generate-only", action="store_true")
    p.add_argument("--reuse", action="store_true", help="Reuse only when all input/code/output hashes match")
    p.add_argument("--timeout", type=int, default=3600, help="Per Blender process timeout in seconds")
    return p


def main(argv=None, *, generate_only_default=False):
    p = parser()
    args = p.parse_args(argv)
    if (args.width is None) != (args.height is None):
        p.error("--width and --height must be supplied together")
    options = dict(blender=args.blender, camera_poses=args.camera_poses, point_cloud=args.point_cloud,
        active_camera=args.active_camera, resolution=(args.width, args.height) if args.width is not None else None,
        engine=args.engine, samples=args.samples, video_start_frame=args.video_start_frame,
        video_frame_count=args.video_frame_count, generate_only=args.generate_only or generate_only_default,
        reuse=args.reuse, timeout=args.timeout)
    try:
        if args.batch:
            if args.scene_json or args.original_video or args.output:
                p.error("--batch cannot be combined with single-job input/output")
            document = read_json(args.batch)
            if document.get("schema") != "stage_ef.batch/1" or not document.get("jobs"):
                raise ValueError("Nonempty stage_ef.batch/1 manifest required")
            base, results, used = args.batch.resolve().parent, [], set()
            for index, job in enumerate(document["jobs"]):
                try:
                    destination = (base/job["output"]).resolve()
                    if destination in used:
                        raise ValueError("Batch jobs must have distinct output directories")
                    used.add(destination)
                    config = dict(options)
                    allowed = {"camera_poses", "point_cloud", "active_camera", "resolution", "engine", "samples", "video_start_frame", "video_frame_count", "generate_only"}
                    if set(job)-allowed-{"scene_json", "original_video", "output"}:
                        raise ValueError("Unknown batch job option")
                    for key in allowed & set(job):
                        config[key] = base/job[key] if key in {"camera_poses", "point_cloud"} and job[key] else job[key]
                    run_pipeline(base/job["scene_json"], base/job["original_video"], destination, **config)
                    results.append(dict(job=index, status="passed", output=str(destination)))
                except (ValueError, KeyError, OSError, RuntimeError, subprocess.SubprocessError) as error:
                    results.append(dict(job=index, status="failed", error=str(error)))
            print(json.dumps(results, ensure_ascii=False, indent=2))
            return 0 if all(r["status"] == "passed" for r in results) else 1
        if not args.scene_json or not args.original_video or not args.output:
            p.error("scene_json, --original-video and --output are required (or use --batch)")
        run_pipeline(args.scene_json, args.original_video, args.output, **options)
        return 0
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Stage E/F: {error}", file=__import__('sys').stderr)
        return 1
