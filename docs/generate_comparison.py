"""Generate README comparisons from local receipts, previews, and the H3 video.

No model inference, reconstruction, Blender execution, or network access.
The shared image mask is predicted, not independent semantic ground truth.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import imageio_ffmpeg
import json
from pathlib import Path
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reconstruction.refine import masked_ssim
from reconstruction.primitive_blockout import masked_luminance_mae
from reconstruction.refined_segmentation import CATEGORIES

ROUTES = {
    "voxel": ROOT / "outputs/stage_ef/heuristic_voxel_proxy",
    "primitive": ROOT / "outputs/stage_ef/heuristic_primitive_from_video",
}
METRIC_ROUTES = (*ROUTES, "h3")
COLORS = {"voxel": "#2563a6", "primitive": "#c16b24", "h3": "#7c4ba6"}
NAMES = {"voxel": "Voxel", "primitive": "Primitive", "h3": "H3"}
MARKERS = {"voxel": "o", "primitive": "s", "h3": "^"}
H3_ROUTE = ROOT / "outputs/h3_controlled_graybox/full_v3"
METRIC_FRAME_INDICES = (0, 60, 180, 300, 360, 420, 540, 660, 719)
VISUAL_FRAME_INDICES = (0, 360, 719)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def last_render_seconds(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    match = re.search(r"(?m)^(\d+):(\d+\.\d+)\s+render\s+\| Saved: .*frame_000720\.png", text)
    if match is None:
        raise ValueError(f"Final-frame elapsed time missing: {path}")
    return int(match[1]) * 60 + float(match[2])


def score_frame(route, index, rgb, predicted, mask):
    """Apply identical source-space metrics; never resize or remask a route."""
    if predicted.shape != rgb.shape:
        raise ValueError("Delivery raster comparison requires equal image shapes")
    score = masked_ssim(rgb, predicted, mask)
    mae = masked_luminance_mae(rgb, predicted, mask)
    if score is None or mae is None or not np.isfinite([score, mae]).all():
        raise ValueError("Empty evaluation mask or non-finite image metric")
    return dict(route=route, frame_index=index, timestamp_seconds=index / 60,
                evaluated_pixels=int(mask.sum()), masked_ssim=score, rgb_mean_mae_255=mae)


def metric_means(rows, route):
    current = [row for row in rows if row["route"] == route]
    return dict(mean_masked_ssim=float(np.mean([row["masked_ssim"] for row in current])),
                mean_rgb_mean_mae_255=float(np.mean([row["rgb_mean_mae_255"] for row in current])))


def collect():
    reports = {
        "voxel": read_json(ROUTES["voxel"] / "proxy_run.json"),
        "primitive": read_json(ROUTES["primitive"] / "primitive_run.json"),
    }
    if any(r.get("status") != "passed" or not r.get("full_video_rendered") for r in reports.values()):
        raise ValueError("Both existing routes must have complete delivery receipts")
    if reports["voxel"]["source_hashes"] != reports["primitive"]["source_hashes"]:
        raise ValueError("Routes do not share the same source receipt")
    scenes = {key: read_json(folder / "scene_schema.json") for key, folder in ROUTES.items()}
    if scenes["voxel"]["cameras"] != scenes["primitive"]["cameras"]:
        raise ValueError("Camera tracks differ")
    receipts = {key: read_json(folder / "pipeline_report.json") for key, folder in ROUTES.items()}
    for receipt in receipts.values():
        if receipt["parameters"]["engine"] != "BLENDER_EEVEE":
            raise ValueError("Unexpected rendering engine")
        for media in receipt["media"].values():
            if media["frame_count"] != 720 or media["fps"] != "60" or not media["pts_verified"]:
                raise ValueError("Media time lattice is not the reviewed 720-frame clip")
    seed = read_json(ROOT / "outputs/stage_ef/real_refined/refinement_report.json")
    cache = Path(seed["frame_cache"])
    indices = [row["frame_index"] for row in reports["voxel"]["preview_frames"]]
    if indices != [row["frame_index"] for row in reports["primitive"]["preview_frames"]]:
        raise ValueError("Preview frame sets differ")
    if tuple(indices) != METRIC_FRAME_INDICES:
        raise ValueError("Metrics require the reviewed nine source frame indices")
    h3_frames = load_h3_frames(indices)
    rows, hashes = [], {
        "h3_delivery_report": digest(H3_ROUTE / "report.json"),
        "h3_delivery_video": digest(H3_ROUTE / "graybox.mp4"),
    }
    for index in indices:
        source = cache / f"frame_{index:06d}.npz"
        hashes[f"cached_frame_{index:06d}"] = digest(source)
        with np.load(source, allow_pickle=False) as data:
            rgb, codes, labels = data["rgb"], data["codes"], data["labels"]
            mask = (codes > 0) & (codes < 1020) & (labels != CATEGORIES.index("person"))
            predictions = {}
            for route, folder in ROUTES.items():
                path = folder / f"frames/frame_{index+1:06d}.png"
                hashes[f"{route}_preview_{index:06d}"] = digest(path)
                with Image.open(path) as image:
                    predictions[route] = np.asarray(image.convert("RGB"))
            predictions["h3"] = np.asarray(h3_frames[index])
            hashes[f"h3_rgb24_frame_{index:06d}"] = hashlib.sha256(predictions["h3"].tobytes()).hexdigest()
            for route in METRIC_ROUTES:
                row = score_frame(route, index, rgb, predictions[route], mask)
                if route in reports:
                    expected = reports[route]["preview_frames"][indices.index(index)]
                    field, metric = (("masked_ssim_informational", "masked_ssim") if route == "voxel"
                                     else ("masked_luminance_mae", "rgb_mean_mae_255"))
                    if abs(expected[field] - row[metric]) > 1e-7:
                        raise ValueError("Existing preview metric and source raster disagree")
                rows.append(row)
    summary = {}
    for route, folder in ROUTES.items():
        objects = scenes[route]["objects"]
        with_observation = sum(bool(obj["observations"]) for obj in objects)
        summary[route] = dict(
            render_elements=len(objects), elements_with_observations=with_observation,
            observation_records=sum(len(obj["observations"]) for obj in objects),
            observation_link_fraction=with_observation / len(objects),
            parent_links=sum(obj["parent_id"] is not None for obj in objects),
            categories=sorted({obj["category"] for obj in objects}),
            blend_mib=(folder / "scene.blend").stat().st_size / 1024**2,
            blockout_mib=(folder / "blockout_video.mp4").stat().st_size / 1024**2,
            **metric_means(rows, route),
            generation_seconds_approx=187 if route == "voxel" else 38,
            render_seconds_to_last_saved_frame=last_render_seconds(folder / "render.log"),
        )
    # H3 has no scene objects, camera matrices, or Blender resource measurements.
    summary["h3"] = metric_means(rows, "h3")
    return rows, summary, hashes, cache


def style():
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.is_file():
        plt.rcParams["font.family"] = FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.spines.bottom": False,
                         "axes.unicode_minus": False, "figure.facecolor": "white"})


def save_resources(output, summary):
    fig = plt.figure(figsize=(10.5, 7.5), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
    time_ax, size_ax, link_ax = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    for y, route in enumerate(ROUTES):
        s = summary[route]
        build, render = s["generation_seconds_approx"] / 60, s["render_seconds_to_last_saved_frame"] / 60
        time_ax.barh(y, build, color=COLORS[route], height=.45)
        time_ax.barh(y, render, left=build, color=COLORS[route], alpha=.35, height=.45)
        time_ax.text(build / 2, y, f"{build:.2f}", ha="center", va="center", color="white", fontsize=12)
        time_ax.text(build + render / 2, y, f"{render:.2f}", ha="center", va="center")
        time_ax.text(build + render + .15, y, f"合计 ≈ {build+render:.2f}", va="center")
        size_ax.barh(y, s["blend_mib"], color=COLORS[route], height=.45)
        size_ax.text(s["blend_mib"] + 3, y, f"{s['blend_mib']:.1f}", va="center")
        fraction = s["observation_link_fraction"] * 100
        link_ax.barh(y, 100, color="#edf0f3", height=.45)
        if fraction:
            link_ax.barh(y, fraction, color=COLORS[route], height=.45)
        link_ax.text(2, y, f"{s['elements_with_observations']:,} / {s['render_elements']:,}  ({fraction:.1f}%)",
                     va="center", color="white" if fraction else "#222222", fontsize=12)
    for ax in (time_ax, size_ax, link_ax):
        ax.set_yticks([0, 1], ["Voxel", "Primitive"])
        ax.invert_yaxis()
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e8ecf0", linewidth=.7)
    time_ax.set(title="场景生成与渲染时间：深色为生成，浅色为渲染", xlabel="分钟", xlim=(0, 11.8))
    size_ax.set(title="工程文件体积", xlabel="MiB", xlim=(0, 190))
    link_ax.set(title="导出元素的观测回链比例", xlabel="%（非识别准确率）", xlim=(0, 102))
    fig.suptitle("两路交付的资源与证据结构", fontsize=18)
    fig.savefig(output / "route_resources.png", dpi=160)
    plt.close(fig)


def save_metrics(output, rows, summary):
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), layout="constrained", sharex=True)
    for route in METRIC_ROUTES:
        current = [row for row in rows if row["route"] == route]
        t = [row["timestamp_seconds"] for row in current]
        for ax, metric in zip(axes, ("masked_ssim", "rgb_mean_mae_255")):
            mean = summary[route]["mean_" + metric]
            ax.plot(t, [row[metric] for row in current], marker=MARKERS[route], color=COLORS[route],
                    linewidth=1.7, markersize=6, label=f"{NAMES[route]}  均值 {mean:.3f}")
    axes[0].set(title="遮罩 SSIM ↑", ylabel="SSIM", ylim=(0, 1))
    mae_max = max(100, max(row["rgb_mean_mae_255"] for row in rows) * 1.1)
    axes[1].set(title="RGB 通道均值灰度 MAE ↓", ylabel="灰度级（0–255）", xlabel="视频时间 / 秒", ylim=(0, mae_max))
    for ax in axes:
        ax.legend(loc="lower left" if ax is axes[0] else "upper left", frameon=False, fontsize=12)
        ax.grid(color="#e8ecf0", linewidth=.7)
        ax.set_xlim(-.25, 12.3)
    axes[1].set_xticks([0, 1, 3, 5, 6, 7, 9, 11, 12])
    fig.suptitle("三路同源九帧、1920×1080 交付栅格、统一环境遮罩\n图像一致性（含风格影响），非几何或动作还原率", fontsize=16)
    fig.savefig(output / "route_preview_metrics.png", dpi=160)
    plt.close(fig)


def load_h3_frames(indices):
    """Decode by final video frame index, not a rounded timestamp or part index."""
    report = read_json(H3_ROUTE / "report.json")
    if (report.get("status") != "complete_source_indexed_approximation"
            or not report.get("temporal_pass") or not report.get("source_mapping_pass")):
        raise ValueError("H3 requires a verified complete delivery receipt")
    probe = report["probe"]
    if (probe["frames"] != 720 or Fraction(probe["fps"]) != 60
            or Fraction(probe["duration_seconds"]) != 12
            or len(probe["pts_seconds"]) != 720
            or any(Fraction(pts) != Fraction(index, 60) for index, pts in enumerate(probe["pts_seconds"]))):
        raise ValueError("H3 media time lattice is not the reviewed 720-frame clip")
    mapping = report["mapping"]
    if len(mapping) != 720 or any(
            item["source_index"] != index or item["padding"]
            or Fraction(item["source_pts_seconds"]) != Fraction(index, 60)
            for index, item in enumerate(mapping)):
        raise ValueError("H3 source mapping is not the reviewed 720-frame clip")
    source_hashes = read_json(ROUTES["voxel"] / "proxy_run.json")["source_hashes"]
    if source_hashes.get(report["source"]) != report["source_sha256"]:
        raise ValueError("H3 does not share the RGB source used by the geometry routes")
    video = H3_ROUTE / "graybox.mp4"
    if digest(video) != report["output_sha256"]:
        raise ValueError("H3 video differs from the verified delivery")
    indices = tuple(indices)
    if not indices or indices != tuple(sorted(set(indices))) or any(i < 0 or i >= 720 for i in indices):
        raise ValueError("H3 frame indices must be unique, increasing, and in 0..719")
    width, height = report["probe"]["width"], report["probe"]["height"]
    selector = "+".join(f"eq(n,{index})" for index in indices)
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
         "-i", str(video), "-map", "0:v:0", "-vf", f"select='{selector}'",
         "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = width * height * 3
    if len(result.stdout) != len(indices) * frame_bytes:
        raise ValueError("H3 decoder did not return all requested native frames")
    return {index: Image.frombytes("RGB", (width, height), result.stdout[row * frame_bytes:(row + 1) * frame_bytes])
            for row, index in enumerate(indices)}


def save_contact_sheet(output, cache):
    h3_frames = load_h3_frames(VISUAL_FRAME_INDICES)
    names = ("原始 RGB", "Voxel", "Primitive", "H3（生成式灰模）")
    width, height, left, gap, top, row_label = 600, 338, 150, 8, 54, 12
    sheet = Image.new("RGB", (left + len(names) * width + (len(names) - 1) * gap,
                              top + len(VISUAL_FRAME_INDICES) * (height + row_label)), "#f4f6f8")
    draw = ImageDraw.Draw(sheet)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = ImageFont.truetype(str(font_path), 30) if font_path.is_file() else ImageFont.load_default()
    for column, name in enumerate(names):
        draw.text((left + column * (width + gap) + 12, 8), name, fill="#20262e", font=font)
    for row, index in enumerate(VISUAL_FRAME_INDICES):
        y = top + row * (height + row_label)
        with np.load(cache / f"frame_{index:06d}.npz", allow_pickle=False) as data:
            images = [Image.fromarray(data["rgb"])]
        for folder in ROUTES.values():
            with Image.open(folder / f"frames/frame_{index+1:06d}.png") as image:
                images.append(image.convert("RGB"))
        images.append(h3_frames[index])
        draw.text((5, y + 135), f"{index/60:.2f}s", fill="#20262e", font=font)
        for column, image in enumerate(images):
            sheet.paste(image.resize((width, height), Image.Resampling.LANCZOS), (left + column * (width + gap), y))
    sheet.save(output / "route_visual_comparison.jpg", quality=94, subsampling=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/figures/comparison")
    parser.add_argument("--visual-only", action="store_true", help="Only rebuild the four-column contact sheet; leave metrics unchanged")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "docs/figures"):
        raise ValueError("Use a documentation figure directory; source artifacts must remain unchanged")
    if args.visual_only:
        cache = Path(read_json(ROOT / "outputs/stage_ef/real_refined/refinement_report.json")["frame_cache"])
        output.mkdir(parents=True, exist_ok=True)
        save_contact_sheet(output, cache)
        print(output / "route_visual_comparison.jpg")
        return
    rows, summary, hashes, cache = collect()
    output.mkdir(parents=True, exist_ok=True)
    document = dict(schema="readme.route_comparison/2", source="existing local delivery receipts, nine preview PNGs per geometry route, and source-indexed H3 delivery frames",
                    semantic_ground_truth_available=False, physical_geometry_validated=False,
                    metric_routes=list(METRIC_ROUTES), metric_frame_indices=list(METRIC_FRAME_INDICES),
                    mask="0 < codes10 < 1020 AND predicted label != person; source-space mask identical for all three routes, not recomputed on H3",
                    ssim="Existing preview metric: Gaussian 11x11, sigma 1.5; 11x11-eroded mask; not Stage G uniform-window SSIM",
                    mae="Mean absolute difference of RGB-channel-mean grayscale, on mask, scale 0..255",
                    aggregation="Arithmetic mean of the same nine per-frame scores; not pixel-weighted or a full-video score",
                    raster="1920x1080 source/geometry previews/H3 delivery frames; no resize during metric calculation. H3 was generated at 768x432 and spatially upscaled during delivery assembly.",
                    alignment="H3 decoded by final video frame index after validating source mapping, exact PTS and delivery SHA256; no temporal seeking, interpolation, or content registration",
                    evidence_boundary="Source image agreement, affected by material, brightness, style, resolution and generated layout; not geometry, semantics, action accuracy or temporal continuity. Source person mask does not exclude a displaced generated person.",
                    time_scope="Generation times are historical approximations; render time is elapsed log time to last PNG, including read/verification. Not full cold start or a controlled speed benchmark.",
                    observation_scope="Voxel associations are inferred representative projections, not verified instances; each linked mesh has one record. Primitive observations are not exported. H3 has no 3D objects or observation links; resource plots remain geometry-only.",
                    summary=summary, preview_metrics=rows, input_sha256=hashes,
                    data_policy="Local internal evaluation only; derived data shares source retention policy")
    (output / "metrics.json").write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    style()
    save_resources(output, summary)
    save_metrics(output, rows, summary)
    save_contact_sheet(output, cache)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
