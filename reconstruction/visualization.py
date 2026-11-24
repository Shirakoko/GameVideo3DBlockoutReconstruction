"""Headless static scientific plots, entirely local (no browser/CDN/services)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def cloud_plot(path: Path, first_cloud, fused_cloud, *, cameras, label):
    fig = plt.figure(figsize=(14, 9), layout="constrained")
    for number, (cloud, title) in enumerate([(first_cloud, "Single frame"), (fused_cloud, "Multi-frame fusion")], 1):
        ax = fig.add_subplot(2, 2, number, projection="3d")
        points, colors = np.asarray(cloud.points), np.asarray(cloud.colors)
        step = max(1, len(points)//16000)
        ax.scatter(*points[::step].T, c=colors[::step], s=1, depthshade=False, rasterized=True)
        for t in cameras:
            origin, forward = t[:3, 3], t[:3, 2]
            ax.quiver(*origin, *forward, length=0.6, color="#dc3345", linewidth=1)
        span = np.maximum(np.ptp(points, axis=0), 1.)
        ax.set_box_aspect(span)
        ax.view_init(elev=25, azim=145)
        ax.set(xlabel="Blender X [m]", ylabel="Blender Y [m]", zlabel="Blender Z [m]",
               title=f"{title}: {len(points):,} points")
    points, colors = np.asarray(fused_cloud.points), np.asarray(fused_cloud.colors)
    for slot, axes, title in [(3, (0, 1), "Top view"), (4, (0, 2), "Side view")]:
        ax = fig.add_subplot(2, 2, slot)
        step = max(1, len(points)//20000)
        ax.scatter(points[::step, axes[0]], points[::step, axes[1]], c=colors[::step], s=1, rasterized=True)
        trajectory = np.array([t[:3, 3] for t in cameras])
        ax.plot(trajectory[:, axes[0]], trajectory[:, axes[1]], "o-", color="#dc3345", markersize=3, label="camera")
        ax.set_aspect("equal", adjustable="box")
        ax.set(xlabel=f"Blender {'XYZ'[axes[0]]} [m]", ylabel=f"Blender {'XYZ'[axes[1]]} [m]", title=title)
        ax.legend(loc="upper left")
    fig.suptitle(label, color="#ad2638", fontsize=15)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def depth_plot(path, frame, bp):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    axes[0].imshow(frame.rgb)
    axes[0].set_title(f"SYNTHETIC RGB / source frame {frame.frame_index}")
    valid = frame.valid_mask & np.isfinite(frame.depth) & (frame.depth > 0) & (frame.depth <= 4000)
    image = np.where(valid, frame.depth*frame.depth_spec.meters_per_unit, np.nan)
    heat = axes[1].imshow(image, cmap="viridis")
    fig.colorbar(heat, ax=axes[1], label="Verified SYNTHETIC planar Z [m]")
    axes[1].set_title("Metric depth / invalid masked")
    axes[2].scatter(bp.points_camera_m[:, 0], bp.points_camera_m[:, 2], c=bp.colors_rgb8/255., s=2)
    axes[2].set(xlabel="Optical X [m]", ylabel="Optical Z [m]", title="Back-projection (camera frame)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def trajectory_plot(path, candidates):
    matrices = np.asarray([row["world_from_optical_candidate_m"] for row in candidates])
    positions = matrices[:, :3, 3]
    times = np.array([row["timestamp_seconds"] for row in candidates])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    axes[0].scatter(positions[:, 0], positions[:, 1], c=times, cmap="viridis", s=4)
    for i in range(0, len(positions), 60):
        forward = matrices[i, :2, 2]
        axes[0].arrow(*positions[i, :2], *(forward*0.7), width=0.025, color="#c53138")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set(xlabel="Blender X [m]", ylabel="Blender Y [m]", title="Telemetry positions; UE orientation hypothesis")
    axes[1].plot(times, positions[:, 2])
    axes[1].set(xlabel="Source time [s]", ylabel="Blender Z [m]", title="Camera height (world metadata scale)")
    fig.suptitle("REAL TELEMETRY ONLY / NOT RECONSTRUCTED SCENE / render-pose binding UNVERIFIED", color="#ad2638", fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)
