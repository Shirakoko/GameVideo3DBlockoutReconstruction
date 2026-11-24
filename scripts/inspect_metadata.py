"""Download only small contracts and explicit JSONL prefixes after approved access."""
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage_a.access import BASE, REPO, Client, InspectionError, credential

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT / "outputs/stage_a/access_report.json").read_text(encoding="utf-8"))
    token, _ = credential()
    if not token:
        raise InspectionError("Authorized local credential required")
    client = Client(token, max_total_bytes=8*1024**2)
    receipts = []
    for session in report["samples"]:
        session_id = session["session_id"]
        for name in ("session.json", "depth/decode_contract.json", "telemetry/world.json", "tables/conversion_manifest.json"):
            url = f"{BASE}/datasets/{REPO}/resolve/{report['revision']}/samples/{session_id}/{name}"
            status, _, body, _ = client.request(url, max_bytes=128*1024)
            if status != 200:
                raise InspectionError(f"Contract access returned HTTP {status}")
            json.loads(body)
            path = ROOT / "data/input/samples" / session_id / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            receipts.append({"session_id": session_id, "path": name, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(), "partial": False})
    candidate = report["provisional_candidate"]
    for name in ("depth/depth_meta.jsonl", "telemetry/camera.jsonl", "telemetry/present_ts.jsonl"):
        size = 1024**2
        url = f"{BASE}/datasets/{REPO}/resolve/{report['revision']}/samples/{candidate}/{name}"
        status, headers, body, _ = client.request(url, max_bytes=size, byte_range=(0, size-1))
        path = ROOT / "data/input/samples" / candidate / (name + ".prefix")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        receipts.append({"session_id": candidate, "path": name, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(), "partial": True, "content_range": headers["content-range"]})
    # Official loader is data-format evidence, not executed code.
    url = f"{BASE}/datasets/{REPO}/resolve/{report['revision']}/tools/load_session.py"
    status, _, body, _ = client.request(url, max_bytes=128*1024)
    if status == 200:
        path = ROOT / "data/input/reference/load_session.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        receipts.append({"path": "tools/load_session.py", "bytes": len(body), "partial": False})
    now = datetime.now(timezone(timedelta(hours=8)))
    manifest = {"repository": REPO, "revision": report["revision"], "downloaded_at": now.isoformat(),
                "delete_by": (now+timedelta(days=30)).isoformat(), "total_bytes": client.bytes_read,
                "media_bytes": 0, "receipts": receipts, "purpose": "internal Stage A evaluation only"}
    (ROOT / "outputs/stage_a/metadata_download_receipt.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"metadata_bytes": client.bytes_read, "candidate": candidate, "media_bytes": 0, "delete_by": manifest["delete_by"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (InspectionError, OSError, ValueError) as error:
        print(json.dumps({"status": "metadata_inspection_failed", "error_type": type(error).__name__, "message": str(error) if isinstance(error, InspectionError) else "Local or format error"}))
        raise SystemExit(1)
