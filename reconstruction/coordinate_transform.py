"""Explicit UE Rotator / optical / Blender bases; column-vector transforms.

Engine documentation fixes FRotator math, NOT a dataset's render-pose latch.
The caller must separately verify that the telemetry really uses that contract.
All returned optical-to-Blender transforms use metres and proper rotations.
"""
from __future__ import annotations

import numpy as np

# Optical: +X image right, +Y image down, +Z forward.
# UE camera-local: +X forward, +Y right, +Z up (left-handed).
UE_FROM_OPTICAL = np.array([[0., 0., 1.], [1., 0., 0.], [0., -1., 0.]])
# Chosen Blender world convention: UE +X -> +X, +Y -> -Y, +Z -> +Z.
# This basis reflection is NOT an ordinary rotation or a camera Euler offset.
BLENDER_FROM_UE = np.diag([1., -1., 1.])
# Blender camera-local: +X right, +Y up, -Z viewing direction.
OPTICAL_FROM_BLENDER_CAMERA = np.diag([1., -1., -1.])


def ue_rotator_matrix(*, pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    """UE local -> UE world, column-vector Rz(yaw) @ Ry(-pitch) @ Rx(-roll).

    Intrinsic Yaw -> Pitch -> Roll. +yaw turns +X toward +Y; +pitch
    raises +X toward +Z; +roll lowers local +Y toward -Z. UE FMatrix's
    row-vector representation would contain the transpose of this matrix.
    """
    angles = np.asarray([pitch_deg, yaw_deg, roll_deg], dtype=np.float64)
    if not np.isfinite(angles).all():
        raise ValueError("Finite UE Rotator angles in explicitly declared degrees required")
    p, y, r = np.deg2rad(angles)
    cp, cy, cr = np.cos([p, y, r])
    sp, sy, sr = np.sin([p, y, r])
    rz = np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])
    ry = np.array([[cp, 0., -sp], [0., 1., 0.], [sp, 0., cp]])
    rx = np.array([[1., 0., 0.], [0., cr, sr], [0., -sr, cr]])
    return rz @ ry @ rx


def validate_rigid(matrix) -> np.ndarray:
    """Reject scale, reflection, NaN and wrong homogeneous storage layout.

    Both a view matrix and its inverse can be rigid: matrix DIRECTION must be
    reviewed explicitly by the caller, not inferred from orthogonality.
    """
    t = np.asarray(matrix, dtype=np.float64)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError("Finite 4x4 transform required")
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-10, rtol=0):
        raise ValueError("Expected column-vector homogeneous bottom row [0,0,0,1]")
    rotation = t[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-9, rtol=0):
        raise ValueError("Expected proper rigid rotation; convert bases and units separately")
    return t.copy()


def world_from_optical_ue_pose(position_ue, *, pitch_deg: float, yaw_deg: float,
                               roll_deg: float, world_units_per_meter: float) -> np.ndarray:
    """T_BlenderWorld_OpticalCamera; depth input must ALREADY be in metres.

    world_to_meters in UE is world units per metre, so translation is divided
    by it. No depth unit is inferred from this world-coordinate scale.
    """
    scale = world_units_per_meter
    position = np.asarray(position_ue, dtype=np.float64)
    if isinstance(scale, bool) or not np.isfinite(scale) or scale <= 0 or position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("Finite position and explicit positive world units per metre required")
    result = np.eye(4)
    result[:3, :3] = BLENDER_FROM_UE @ ue_rotator_matrix(
        pitch_deg=pitch_deg, yaw_deg=yaw_deg, roll_deg=roll_deg) @ UE_FROM_OPTICAL
    result[:3, 3] = BLENDER_FROM_UE @ position / scale
    return validate_rigid(result)


def blender_camera_from_optical_pose(world_from_optical) -> np.ndarray:
    """Return a Blender camera object's matrix_world; NOT a view matrix."""
    t = validate_rigid(world_from_optical)
    local_change = np.eye(4)
    local_change[:3, :3] = OPTICAL_FROM_BLENDER_CAMERA
    return validate_rigid(t @ local_change)


def inverse_rigid(matrix) -> np.ndarray:
    t = validate_rigid(matrix)
    result = np.eye(4)
    result[:3, :3] = t[:3, :3].T
    result[:3, 3] = -t[:3, :3].T @ t[:3, 3]
    return result


def transform_points(points, matrix) -> np.ndarray:
    """For Nx3 row storage, P_out = P_in @ R.T + t (equivalent to T @ p)."""
    t = validate_rigid(matrix)
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("Finite Nx3 points required")
    return p @ t[:3, :3].T + t[:3, 3]
