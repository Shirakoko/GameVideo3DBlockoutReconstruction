"""Read-only, size-bounded Hugging Face access inspection. No media downloads."""

from __future__ import annotations

import html
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from datetime import timezone, timedelta

REPO = "originlab/game-recordings-v4"
BASE = "https://huggingface.co"
REQUIRED = (
    "session.json", "depth/decode_contract.json", "depth/depth_meta.jsonl",
    "depth/depth.hevc", "video/prehud.mp4", "telemetry/camera.jsonl",
    "telemetry/world.json", "tables/frames.parquet", "tables/conversion_manifest.json",
)


class InspectionError(ValueError):
    pass


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Do not forward HF bearer credentials to storage/CDN redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
            redirected.remove_header("Authorization")
        return redirected


def credential() -> tuple[str | None, str]:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(name, "").strip():
            return os.environ[name].strip(), name
    candidates = []
    if os.environ.get("HF_TOKEN_PATH"):
        candidates.append(Path(os.environ["HF_TOKEN_PATH"]))
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    candidates.extend([root / "token", Path.home() / ".huggingface" / "token"])
    for candidate in candidates:
        if candidate.is_file():
            token = candidate.read_text(encoding="utf-8").strip()
            if token:
                return token, "local Hugging Face credential file"
    return None, "none"


class Client:
    def __init__(self, token: str | None = None, max_total_bytes: int = 8 * 1024 * 1024):
        self._token = token
        self.max_total_bytes = max_total_bytes
        self.bytes_read = 0
        self.opener = urllib.request.build_opener(SafeRedirect())

    def request(self, url: str, method="GET", max_bytes=2 * 1024 * 1024, byte_range=None):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.netloc != "huggingface.co":
            raise InspectionError("Only HTTPS requests to huggingface.co are allowed")
        headers = {"User-Agent": "StageA-small-sample-inspector/0.1"}
        if byte_range is not None:
            first, last = byte_range
            if not isinstance(first, int) or not isinstance(last, int) or first < 0 or last < first:
                raise InspectionError("Invalid explicit byte range")
            headers["Range"] = f"bytes={first}-{last}"
        if self._token:
            headers["Authorization"] = "Bearer " + self._token
        req = urllib.request.Request(url, headers=headers, method=method)
        try:
            response = self.opener.open(req, timeout=30)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            status = response.code
            allowed = {key.lower(): response.headers.get(key) for key in
                       ("Content-Length", "Content-Type", "X-Error-Code", "Content-Range") if response.headers.get(key)}
            if method == "HEAD":
                return status, allowed, b"", response.headers.get("Link")
            if byte_range is not None:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if status != 206 or not match or (int(match[1]), int(match[2])) != byte_range:
                    raise InspectionError("Server did not honor exact HTTP Range; stopped before reading payload")
            limit = min(max_bytes, self.max_total_bytes - self.bytes_read)
            if limit < 0:
                raise InspectionError("Inspection byte budget exhausted")
            # One extra byte is bounded too, and detects truncation instead of parsing partial JSON.
            body = response.read(limit + 1)
            self.bytes_read += len(body)
            if len(body) > limit:
                raise InspectionError("Response exceeds inspection byte budget")
            return status, allowed, body, response.headers.get("Link")

    def get_json(self, url):
        status, _, body, link = self.request(url)
        if status != 200:
            raise InspectionError(f"Metadata API returned HTTP {status}")
        return json.loads(body), link


def rank_samples(entries):
    grouped = {}
    for entry in entries:
        if entry.get("type") != "file":
            continue
        parts = entry["path"].split("/", 2)
        if len(parts) != 3 or parts[0] != "samples":
            continue
        grouped.setdefault(parts[1], {})[parts[2]] = int(entry["size"])
    results = []
    for session_id, files in grouped.items():
        missing = [name for name in REQUIRED if name not in files]
        results.append({
            "session_id": session_id, "files": files, "missing_required_paths": missing,
            "required_bytes": sum(files.get(name, 0) for name in REQUIRED),
            "rgb_depth_bytes": sum(files.get(name, 0) for name in ("video/prehud.mp4", "depth/depth.hevc")),
            "suitability": "unverified; paths are not proof of metric depth or pose coverage",
        })
    return sorted(results, key=lambda item: (bool(item["missing_required_paths"]), item["required_bytes"]))


def public_card_text(page: bytes) -> str:
    text = page.decode("utf-8")
    start = text.find('id="origin-lab-game-recordings-v040"')
    end = text.find('id="citation"', start)
    if start < 0 or end < 0:
        raise InspectionError("Public README boundaries not found; do not guess card contents")
    return html.unescape(re.sub(r"<[^>]*>", "", text[start:end]))


def environment_summary():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg and importlib.util.find_spec("imageio_ffmpeg"):
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    version = None
    if ffmpeg:
        result = subprocess.run([ffmpeg, "-version"], capture_output=True, timeout=10)
        version = result.stdout.decode(errors="replace").splitlines()[0] if result.returncode == 0 else None
    blender = shutil.which("blender")
    if not blender and os.name == "nt":
        directory = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Blender Foundation"
        matches = sorted(directory.glob("Blender */blender.exe")) if directory.is_dir() else []
        blender = str(matches[-1]) if matches else None
    return {"python_version": sys.version.split()[0], "python_executable": sys.executable,
            "ffmpeg_executable": ffmpeg, "ffmpeg_version": version,
            "ffprobe_executable": shutil.which("ffprobe"),
            "packages": {name: bool(importlib.util.find_spec(name)) for name in ("numpy", "PIL", "imageio_ffmpeg", "pyarrow")},
            "blender_installed_path": blender, "blender_launched": False}


def inspect_dataset(client: Client, credential_source: str):
    info, _ = client.get_json(f"{BASE}/api/datasets/{REPO}")
    revision = info["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise InspectionError("Cannot pin a valid repository commit")
    url = f"{BASE}/api/datasets/{REPO}/tree/{revision}/samples?recursive=true&expand=false"
    entries = []
    for _ in range(10):
        page, link = client.get_json(url)
        entries.extend(page)
        next_match = re.search(r'<([^>]+)>;\s*rel="next"', link or "")
        if not next_match:
            break
        url = next_match.group(1)
    else:
        raise InspectionError("Tree pagination exceeded 10-page safety cap")
    ranked = rank_samples(entries)
    if not ranked:
        raise InspectionError("No sample sessions in public tree")
    candidate = next((item for item in ranked if not item["missing_required_paths"]), None)
    probes = []
    paths = ["README.md", "LICENSE.md"]
    if candidate:
        paths += [f"samples/{candidate['session_id']}/{name}" for name in REQUIRED]
    for path in paths:
        status, headers, _, _ = client.request(f"{BASE}/datasets/{REPO}/resolve/{revision}/{path}", method="HEAD")
        probes.append({"path": path, "http_status": status, "headers": headers})
    card = None
    license_text = None
    document_reads = []
    card_error = None
    try:
        for path in ("README.md", "LICENSE.md"):
            status, _, body, _ = client.request(f"{BASE}/datasets/{REPO}/resolve/{revision}/{path}", max_bytes=128*1024)
            document_reads.append({"path": path, "method": "GET", "http_status": status})
            if status == 200:
                if path == "README.md":
                    card = body.decode("utf-8")
                else:
                    license_text = body.decode("utf-8")
        if card is None:
            status, _, body, _ = client.request(f"{BASE}/datasets/{REPO}")
            if status == 200:
                card = public_card_text(body)
    except InspectionError as error:
        card_error = str(error)
    try:
        local_zone = ZoneInfo("Asia/Shanghai")
    except ZoneInfoNotFoundError:
        local_zone = timezone(timedelta(hours=8), "Asia/Shanghai")
    ready = candidate is not None and all(item["http_status"] == 200 for item in probes)
    return {
        "schema_version": "0.1.0", "checked_at": datetime.now(local_zone).isoformat(),
        "repository": REPO, "revision": revision, "gated": info.get("gated"),
        "environment": environment_summary(),
        "credential_source": credential_source, "public_api_http_status": 200,
        "status": "metadata_accessible_not_validated" if ready else "blocked_access",
        "samples": ranked, "provisional_candidate": candidate["session_id"] if candidate else None,
        "selection_basis": "smallest required-file total; suitability must be checked after authorization",
        "file_probes": probes, "probe_method": "HEAD", "document_reads": document_reads,
        "inspection_bytes_read": client.bytes_read,
        "dataset_payload_bytes_downloaded": 0, "media_downloaded": False,
        "public_readme_read": card is not None, "public_readme_error": card_error,
        "public_license_read": license_text is not None, "public_license_text": license_text,
        "license_draft_notice": bool(license_text and "DRAFT" in license_text),
        "public_evaluation_terms": info.get("cardData", {}).get("extra_gated_prompt"),
        "retention_deadline": None,
        "retention_note": "No restricted payload downloaded; start 30-day retention tracking on first authorized download.",
        "checks": {name: {"status": "not_run", "reason": "Requires actual Session metadata and media"} for name in (
            "depth_transform", "metric_depth", "near_plane_cm", "intrinsics",
            "world_coordinates", "decoded_frame_counts", "duplicate_frames",
            "frame_index_mapping", "camera_qpc_binning", "timestamp_sync", "10_15_second_sample")},
        "stage_a_complete": False, "blender_used": False,
    }, card


def write_report(report, output: Path, card: str | None = None):
    output.mkdir(parents=True, exist_ok=True)
    (output / "access_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if card:
        (output / "public_readme.txt").write_text(card, encoding="utf-8")
    if report.get("public_license_text"):
        (output / "dataset_LICENSE.md").write_text(report["public_license_text"], encoding="utf-8")
    lines = ["# Stage A 数据访问与完整性报告", "", f"状态：`{report['status']}`；Stage A 尚未完成。",
             "", f"检查时间：{report['checked_at']}", f"固定版本：`{report['revision']}`", "",
             "## 公开目录分析", "", "下表仅验证文件路径存在，不代表内容完整、米制深度或同步合格。", "",
             "| sample Session | RGB+Depth / GiB | 必需文件 / GiB | 缺失路径 |", "|---|---:|---:|---|"]
    for item in report["samples"]:
        lines.append(f"| {item['session_id']} | {item['rgb_depth_bytes']/2**30:.3f} | {item['required_bytes']/2**30:.3f} | {', '.join(item['missing_required_paths']) or '无'} |")
    lines += ["", f"暂定候选：`{report['provisional_candidate']}`，仅按下载成本排序，不能声称已选出合格 Session。", "",
              "## 实际访问", "", "下表为 HEAD 检查，200 不表示已经读取或验证数据内容。README 和 LICENSE 另行执行有界 GET。", "", "| 文件 | HEAD HTTP |", "|---|---:|"]
    lines += [f"| {item['path']} | {item['http_status']} |" for item in report["file_probes"]]
    env = report["environment"]
    lines += ["", "## 环境", "", f"Python：{env['python_version']}；FFmpeg：{env['ffmpeg_version']}。",
              f"FFprobe 可执行文件：{env['ffprobe_executable'] or '未找到'}；Blender 安装：{bool(env['blender_installed_path'])}，未启动。",
              "本地 `.venv` 仅安装 NumPy、Pillow、imageio-ffmpeg；未安装 Open3D、未运行重建或 Blender。", "",
              "## 核心验证", "", "Depth 公式、near_plane_cm、内参、坐标系、重复帧、Frame Index、Camera QPC 和 10–15 秒样本：均未获得真实数据验证。",
              "", "公开 README 给出了相对 log-depth 示例，但不能据此为任何 Session 默认 K=4000，也不能将该示例直接认定为已验证的米制公式。",
              "公开文档确认：Frame Index 为关联键；camera.jsonl 可能每帧有多个 QPC 采样；对齐时丢弃的原始 ticks 记录于 session.json；帧表的相机 binning 规则在 conversion_manifest.json。",
              "", "## 本地实现边界", "", "已提供原生 10-bit HEVC luma / RGB 读取、流式 JSONL 读取、显式索引关联和重复帧统计。仅合成数据测试通过不等于真实 Session 验收。",
              "可运行 `python scripts/run_tests.py`；完整测试结果在 `tests/test_results.json`（相对于本报告目录）。测试不再额外生成合成诊断图片或报告。",
              "米制 Depth、实际 Session 字段适配及自动截取流水线尚待授权后实现与验证。不做相机坐标转换，不进入 Blender。",
              "", "## 下载与许可", "", f"受限数据下载：{report['dataset_payload_bytes_downloaded']} bytes；媒体下载：否。",
              "仅限内部评估，不再分发，不上传第三方，不公开发布；获批下载后按许可记录 30 天删除期限。",
              f"README 内容已读取：{report['public_readme_read']}；LICENSE 内容已读取：{report.get('public_license_read', False)}。",
              "本次公开 LICENSE 标有 DRAFT / pending legal review；不自动接受条款，不假定具备生产、发布或商业授权。" if report.get("license_draft_notice") else "以实际许可条款为准。",
              "", "## 继续条件", "", "请由用户本人在 Hugging Face 请求并接受访问条款，然后在本机设置具有该仓库 read 权限的 token（不要在聊天中粘贴）。",
              "重新执行 `python scripts/inspect_dataset.py`。成功访问后先读 README、LICENSE、session.json、decode_contract.json、depth_meta.jsonl、world.json 与 conversion_manifest.json，再设计受字节上限约束的 10–15 秒局部读取。"]
    (output / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
