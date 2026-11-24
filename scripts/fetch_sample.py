"""Fetch at most 128 MiB for one short real-session prefix. Never snapshot a session."""
import hashlib
import json
import struct
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage_a.access import BASE, REPO, Client, InspectionError, credential
from stage_a.mp4 import video_prefix_end, relocate_chunks

ROOT = Path(__file__).resolve().parents[1]
SESSION = "66ad9535-583f-4a25-b655-6b1697c83a40"


def main():
    token, _ = credential()
    if not token:
        raise InspectionError("Authenticated access required")
    revision = json.loads((ROOT / "outputs/stage_a/access_report.json").read_text())["revision"]
    client = Client(token, max_total_bytes=128*1024**2)
    target = ROOT / "data/input/samples" / SESSION
    receipts = []

    def read_range(name, start, size):
        url = f"{BASE}/datasets/{REPO}/resolve/{revision}/samples/{SESSION}/{name}"
        _, headers, body, _ = client.request(url, max_bytes=size, byte_range=(start, start+size-1))
        receipts.append({"source_path": name, "byte_start": start, "bytes": size,
                         "sha256": hashlib.sha256(body).hexdigest(), "content_range": headers["content-range"]})
        return body

    for name in ("depth/depth_meta.jsonl", "telemetry/camera.jsonl", "telemetry/present_ts.jsonl"):
        body = read_range(name, 0, 1024**2)
        path = target / (name+".prefix")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    # A 14-second pre-roll prefix permits choosing an in-range 10–12 second clip.
    source_size = json.loads((target / "session.json").read_text())["files"]["video/prehud.mp4"]["size_bytes"]
    cursor = 0
    ftyp = None
    moov = None
    mdat_start = None
    top_boxes = []
    for _ in range(100):
        header = read_range("video/prehud.mp4", cursor, 16)
        size, kind = struct.unpack_from(">I4s", header)
        header_size = 8
        if size == 1:
            size = struct.unpack_from(">Q", header, 8)[0]
            header_size = 16
        elif size == 0:
            size = source_size-cursor
        if size < header_size or cursor+size > source_size:
            raise InspectionError("Invalid source MP4 top-level box")
        top_boxes.append({"type": kind.decode(), "offset": cursor, "size": size})
        if kind == b"ftyp":
            ftyp = read_range("video/prehud.mp4", cursor, size)
        elif kind == b"moov":
            if size > 16*1024**2:
                raise InspectionError("MP4 index exceeds small-test budget")
            moov = read_range("video/prehud.mp4", cursor, size)
        elif kind == b"mdat":
            if mdat_start is not None:
                raise InspectionError("Multiple mdat layout requires a reviewed adapter")
            mdat_start = cursor+header_size
        cursor += size
        if cursor == source_size:
            break
    if ftyp is None or moov is None or mdat_start is None:
        raise InspectionError("Required MP4 top-level boxes missing")
    prefix_end, mp4_info = video_prefix_end(moov, 900)  # 15 seconds decoder input; final clip is 12 seconds.
    if prefix_end-mdat_start > 64*1024**2:
        raise InspectionError("RGB prefix exceeds 64 MiB cap")
    pieces = []
    for first in range(mdat_start, prefix_end, 4*1024**2):
        size = min(4*1024**2, prefix_end-first)
        pieces.append(read_range("video/prehud.mp4", first, size))
        print(f"RGB range received: {client.bytes_read} bytes total", flush=True)
    payload = b"".join(pieces)
    delta = len(ftyp)+len(moov)+8-mdat_start
    intermediate = ftyp+relocate_chunks(moov, delta)+struct.pack(">I4s", len(payload)+8, b"mdat")+payload
    path = target / "video/prehud_prefix.partial.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(intermediate)
    # Depth is a raw HEVC stream, so download only a bounded prefix, not all 3 GB.
    depth_pieces = []
    for first in range(0, 24*1024**2, 4*1024**2):
        depth_pieces.append(read_range("depth/depth.hevc", first, 4*1024**2))
        print(f"Depth range received: {client.bytes_read} bytes total", flush=True)
    (target / "depth/depth_prefix.partial.hevc").write_bytes(b"".join(depth_pieces))
    # Entire small table is 5.2 MiB, not the video/session; it verifies last_in_frame.
    table_size = json.loads((target / "session.json").read_text())["files"]["tables/frames.parquet"]["size_bytes"]
    if table_size > 8*1024**2:
        raise InspectionError("Frame table exceeds 8 MiB cap")
    table = read_range("tables/frames.parquet", 0, table_size)
    (target / "tables/frames.parquet").write_bytes(table)
    now = datetime.now(timezone(timedelta(hours=8)))
    receipt = {"repository": REPO, "revision": revision, "session_id": SESSION,
               "downloaded_at": now.isoformat(), "delete_by": (now+timedelta(days=30)).isoformat(),
               "bytes_received": client.bytes_read, "hard_cap_bytes": client.max_total_bytes,
               "media_is_prefix_only": True, "mp4_top_boxes": top_boxes, "mp4_info": mp4_info,
               "mp4_intermediate_note": "Original complete index with relocated offsets and truncated media payload. NOT a valid complete session; only used for bounded early decode and re-muxing.",
               "receipts": receipts}
    (ROOT / "outputs/stage_a/sample_download_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"session_id": SESSION, "bytes_received": client.bytes_read, "prefix_video_samples": mp4_info["prefix_video_samples_including_preroll"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (InspectionError, OSError, ValueError) as error:
        print(json.dumps({"status": "bounded_fetch_failed", "error_type": type(error).__name__, "message": str(error) if isinstance(error, InspectionError) else "Local or format error"}))
        raise SystemExit(1)
