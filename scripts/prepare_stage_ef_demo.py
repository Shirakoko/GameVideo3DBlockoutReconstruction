"""Prepare a clearly synthetic 12-second / 60-FPS input, no dataset download."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.object_detection import SyntheticColorProvider
from agents.scene_builder import build_scene
from blender.pipeline import json_write
from blender.synthetic_demo import demo_frame
from scripts.understand_scene import rgb_records
from stage_a.media import ffmpeg_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT/"outputs/stage_ef/demo_input")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fps, count = 60, 720
    frames = [demo_frame(i, fps) for i in [0, 120, 240, 360, 480, 600, 719]]
    source = "SYNTHETIC 12s RAY/AABB FIVE-CATEGORY DEMO ONLY — NOT HUGGING FACE"
    scene, report = build_scene(rgb_records(frames), SyntheticColorProvider(), args.output, scene_id="synthetic-stage-ef-12s",
        data_source=source, fps=fps, metric_frames=frames, min_depth_m=.1, max_depth_m=35)
    if report["status"] != "complete" or len(scene["objects"]) != 5:
        raise ValueError("Synthetic scene preparation failed")
    samples = []
    command = [ffmpeg_path(), "-y", "-nostdin", "-hide_banner", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "240x140",
               "-r", "60", "-i", "-", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               str((args.output/"original_video.mp4").resolve())]
    with (args.output/"encode_demo.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log, stdout=log)
        try:
            for i in range(count):
                frame = demo_frame(i, fps)
                process.stdin.write(frame.rgb.tobytes())
                samples.append(dict(camera_id="camera_main", frame_index=frame.frame_index, original_tick=frame.original_tick,
                    timestamp_seconds=frame.timestamp_seconds, world_from_camera=frame.world_from_camera.tolist(), intrinsics=frame.intrinsics.tolist()))
        finally:
            process.stdin.close()
            code = process.wait(timeout=120)
        if code:
            raise RuntimeError("Synthetic original encoding failed")
    json_write(args.output/"camera_poses.json", dict(schema="stage_e.camera_poses/1", verified=True, evidence=source+"; same-call capture and known basis",
        pose_format="world_from_optical_camera", coordinate_system="blender_rh_z_up", translation_unit="m", samples=samples))
    print(f"Prepared SYNTHETIC ONLY: {args.output.resolve()} (720 frames / 60 FPS / 12 seconds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
