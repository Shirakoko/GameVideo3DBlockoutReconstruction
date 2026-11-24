"""Bounded local decoding, keeping native 10-bit luma codes without swscale."""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class DecodeError(ValueError):
    pass


def ffmpeg_path() -> str:
    configured = os.environ.get("STAGE_A_FFMPEG")
    if configured:
        if not Path(configured).is_file():
            raise DecodeError("STAGE_A_FFMPEG does not name a local executable")
        return configured
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as error:
        raise DecodeError("Install requirements.txt or configure STAGE_A_FFMPEG") from error


@dataclass(frozen=True)
class VideoSpec:
    width: int
    height: int
    pixel_format: str


def probe_local(path: Path, *, elementary_hevc=False) -> VideoSpec:
    if not path.is_file():
        raise DecodeError("Local media file does not exist")
    args = [ffmpeg_path(), "-hide_banner", "-nostdin"]
    if elementary_hevc:
        args += ["-f", "hevc"]
    args += ["-i", str(path), "-map", "0:v:0", "-frames:v", "1", "-f", "null", "-"]
    result = subprocess.run(args, capture_output=True, timeout=30)
    text = result.stderr.decode("utf-8", errors="replace").split("Output #0", 1)[0]
    stream = next((line for line in text.splitlines() if "Stream #0:" in line and "Video:" in line), "")
    # A bare NxN regex also matches MP4 stream ids such as [0x1]. Raster
    # dimensions are a comma-delimited field AFTER the Video: declaration.
    size = re.search(r",\s*(\d+)x(\d+)(?=[,\s])", stream.split("Video:", 1)[-1])
    fmt = re.search(r",\s*((?:yuv|gray|rgb|bgr)[a-z0-9]+)(?:\(|,|\s)", stream)
    if result.returncode or not size or not fmt or min(int(size[1]), int(size[2])) <= 0:
        raise DecodeError("Cannot establish source raster/pixel format from FFmpeg; do not guess")
    return VideoSpec(int(size[1]), int(size[2]), fmt[1])


def _read_exact(pipe, size):
    chunks = []
    remaining = size
    while remaining:
        part = pipe.read(remaining)
        if not part:
            break
        chunks.append(part)
        remaining -= len(part)
    return b"".join(chunks)


def iter_frames(path: Path, *, start: int, count: int, depth=False, max_decoded_bytes=4 * 1024**3):
    """Yield (original decoded ordinal, array); no implicit FPS/frame duplication.

    Depth supports native yuv420p10le only. It exports that exact format and
    extracts Y, rather than converting to gray16le (which may rescale luma).
    No metric conversion, camera mapping or source-capture index inference.
    """
    if isinstance(start, bool) or isinstance(count, bool) or not isinstance(start, int) or not isinstance(count, int) or start < 0 or count <= 0:
        raise DecodeError("Nonnegative integer start and positive integer count required")
    spec = probe_local(path, elementary_hevc=depth)
    if depth and (spec.pixel_format != "yuv420p10le" or spec.width % 2 or spec.height % 2):
        raise DecodeError("Depth must be native even-sized yuv420p10le; other layouts require a reviewed adapter")
    frame_size = spec.width * spec.height * 3
    if count * frame_size > max_decoded_bytes:
        raise DecodeError("Requested raw decode exceeds the explicit byte budget")
    args = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-nostdin"]
    if depth:
        args += ["-f", "hevc"]
    args += ["-i", str(path), "-map", "0:v:0", "-an", "-sn",
             "-vf", f"select=between(n\\,{start}\\,{start+count-1})",
             "-frames:v", str(count), "-fps_mode", "passthrough",
             "-pix_fmt", "yuv420p10le" if depth else "rgb24", "-f", "rawvideo", "-"]
    # Temporary stderr spool avoids pipe deadlocks on decoder errors.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=errors)
        delivered = 0
        try:
            for ordinal in range(start, start + count):
                raw = _read_exact(process.stdout, frame_size)
                if len(raw) != frame_size:
                    raise DecodeError(f"Truncated/missing decoded frame {ordinal}; got {len(raw)} of {frame_size} bytes")
                if depth:
                    frame = np.frombuffer(raw, dtype="<u2", count=spec.width * spec.height).reshape(spec.height, spec.width).copy()
                    if frame.max() > 1023:
                        raise DecodeError("Native 10-bit code exceeds 1023; raster/layout mismatch")
                else:
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(spec.height, spec.width, 3).copy()
                delivered += 1
                yield ordinal, frame
            # Drain to EOF before closing: FFmpeg may still be flushing the last
            # packet after its bytes have already reached the reader.
            unexpected = process.stdout.read(1)
            if unexpected:
                raise DecodeError("FFmpeg produced extra frames/bytes beyond the requested count")
            code = process.wait(timeout=30)
            if code:
                errors.seek(0)
                diagnostic = errors.read().decode("utf-8", errors="replace")[-2000:]
                raise DecodeError(f"FFmpeg decoder failed (exit {code}): {diagnostic}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            if process.stdout and not process.stdout.closed:
                process.stdout.close()
        if delivered != count:
            raise DecodeError("Decoded frame count differs from request")


class DuplicateTracker:
    """Exact decoded-content equality, NOT equivalent to source dup_count."""
    def __init__(self):
        self.previous = None
        self.frames = 0
        self.adjacent_equal = 0

    def add(self, frame: np.ndarray) -> bool:
        digest = hashlib.sha256(str((frame.shape, frame.dtype.str)).encode() + frame.tobytes()).digest()
        duplicate = self.previous == digest
        self.adjacent_equal += int(duplicate)
        self.frames += 1
        self.previous = digest
        return duplicate

    def report(self):
        return {"decoded_frames": self.frames, "adjacent_exact_duplicate_count": self.adjacent_equal,
                "definition": "exact equality after decode; static views may match; lossy copies may differ",
                "first_frame_previous_outside_clip_checked": False}
