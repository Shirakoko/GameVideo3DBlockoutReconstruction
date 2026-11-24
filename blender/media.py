"""FFmpeg-only media inspection/encoding with exact CFR PTS validation."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import re
import subprocess

from stage_a.media import ffmpeg_path, probe_local


def command(args, log_path=None, timeout=3600):
    try:
        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        output = (error.output or b"").decode("utf-8", errors="replace")
        output += f"\nProcess timed out after {timeout} seconds.\n"
        if log_path:
            Path(log_path).write_text(output, encoding="utf-8")
        raise RuntimeError(f"Command timed out after {timeout}s; {log_path or output[-2000:]}") from error
    output = result.stdout.decode("utf-8", errors="replace")
    if log_path:
        Path(log_path).write_text(output, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); {log_path or output[-2000:]}")
    return output


def probe_video(path, *, start_frame=0, frame_count=None, log_path=None, timeout=3600):
    """Decode the selected stream, checking every PTS (including B-frame order).

    showinfo exposes integer PTS and rational time_base/frame_rate; ffprobe is
    not required. VFR, rotation metadata and nonmonotonic PTS are refused.
    """
    path = Path(path).resolve()
    if (isinstance(start_frame, bool) or not isinstance(start_frame, int) or start_frame < 0
            or (frame_count is not None and (isinstance(frame_count, bool)
                or not isinstance(frame_count, int) or frame_count <= 0))):
        raise ValueError("Invalid video frame window")
    spec = probe_local(path)
    output = command([ffmpeg_path(), "-hide_banner", "-nostdin", "-i", str(path),
                      "-map", "0:v:0", "-an", "-sn", "-vf", "showinfo",
                      "-fps_mode", "passthrough", "-f", "null", "-"], log_path, timeout=timeout)
    config = re.findall(r"config in time_base:\s*(\d+/\d+), frame_rate:\s*(\d+/\d+)", output)
    if not config or len(set(config)) != 1:
        raise ValueError("Cannot establish stable rational video timebase/FPS")
    timebase, fps = map(Fraction, config[0])
    if fps <= 0 or "displaymatrix" in output.lower() or "rotation of" in output.lower():
        raise ValueError("Positive FPS and unrotated video required")
    pts = [(int(n), int(p)) for n, p in re.findall(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:", output)]
    if not pts or [n for n, _ in pts] != list(range(len(pts))):
        raise ValueError("Missing/duplicate decoded frame PTS")
    count = len(pts)-start_frame if frame_count is None else frame_count
    window = pts[start_frame:start_frame+count]
    if len(window) != count or not window:
        raise ValueError("Original video frame window is incomplete")
    times = [Fraction(p)*timebase for _, p in window]
    if any(b-a != 1/fps for a, b in zip(times, times[1:])):
        raise ValueError("VFR/nonmonotonic video: cannot preserve CFR FPS without changing source timing")
    return dict(path=str(path), width=spec.width, height=spec.height, fps=str(fps),
                frame_count=count, total_decoded_frames=len(pts), start_frame=start_frame,
                first_pts_seconds=float(times[0]), duration_seconds=float(count/fps),
                timebase=str(timebase), pts_verified=True)


def encode_outputs(workspace, video, resolution, *, timeout=3600):
    workspace = Path(workspace)
    ffmpeg = ffmpeg_path()
    fps, count = video["fps"], video["frame_count"]
    common = ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
              "-fps_mode", "passthrough", "-movflags", "+faststart", "-an"]
    command([ffmpeg, "-y", "-nostdin", "-hide_banner", "-framerate", fps, "-start_number", "1",
             "-i", str(workspace/"frames/frame_%06d.png"), "-frames:v", str(count),
             *common, str(workspace/"blockout_video.mp4")], workspace/"encode_blockout.log", timeout=timeout)
    start = video["start_frame"]
    # trim by decoded ordinal, never by rounded seconds or keyframe seek.
    trim = f"trim=start_frame={start}:end_frame={start+count},setpts=PTS-STARTPTS"
    command([ffmpeg, "-y", "-nostdin", "-hide_banner", "-i", video["path"], "-map", "0:v:0",
             "-vf", trim, *common, str(workspace/"original_video.mp4")], workspace/"encode_original.log", timeout=timeout)
    w, h = resolution
    split = (f"[0:v]scale={w}:{h}:flags=lanczos,setsar=1[left];"
             "[1:v]setsar=1[right];[left][right]hstack=inputs=2:shortest=1[out]")
    command([ffmpeg, "-y", "-nostdin", "-hide_banner", "-i", str(workspace/"original_video.mp4"),
             "-i", str(workspace/"blockout_video.mp4"), "-filter_complex", split, "-map", "[out]",
             *common, str(workspace/"original_vs_blockout.mp4")], workspace/"encode_comparison.log", timeout=timeout)
    reports = {}
    for name, size in [("original_video.mp4", (video["width"], video["height"])),
                       ("blockout_video.mp4", (w, h)), ("original_vs_blockout.mp4", (w*2, h))]:
        report = probe_video(workspace/name, log_path=workspace/(name+".probe.log"), timeout=timeout)
        if report["frame_count"] != count or Fraction(report["fps"]) != Fraction(fps):
            raise ValueError(f"Output frame count/FPS mismatch: {name}")
        if (report["width"], report["height"]) != size or abs(report["first_pts_seconds"]) > 1e-8:
            raise ValueError(f"Output raster/PTS origin mismatch: {name}")
        reports[name] = report
    return reports
