"""Only the public README's relative transform, NOT a session metric decoder."""
from __future__ import annotations

import numpy as np


class UnverifiedContractError(ValueError):
    pass


def relative_log_depth(q: np.ndarray, *, log_constant: float) -> np.ndarray:
    """Public example: expm1(q * log1p(K)) / K; K must be supplied, not guessed.

    q is already normalized by a reviewed code-range rule. This function makes
    no claim about native HEVC luma range, sky masks or metric units.
    """
    q = np.asarray(q, dtype=np.float64)
    if not np.isfinite(log_constant) or log_constant <= 0:
        raise ValueError("log_constant must be positive and finite")
    if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("Normalized codes must be finite and within [0, 1]")
    return np.expm1(q * np.log1p(log_constant)) / log_constant


def metric_depth(*args, **kwargs):
    raise UnverifiedContractError(
        "Metric decoding is intentionally unavailable until the actual session "
        "depth_transform, code range, invalid-value rules and near_plane_cm have been reviewed. "
        "Do not multiply the README's relative example by an assumed near plane."
    )


def validate_intrinsics(matrix, *, width: int, height: int) -> dict:
    """Check an explicitly supplied pinhole K at its own raster resolution.

    Does not infer K from FOV or scale depth K to RGB K.
    """
    k = np.asarray(matrix, dtype=np.float64)
    if width <= 0 or height <= 0 or k.shape != (3, 3) or not np.isfinite(k).all():
        raise ValueError("Finite 3x3 K and positive image dimensions required")
    if not np.allclose(k[2], [0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError("Invalid pinhole homogeneous row")
    if k[0, 0] <= 0 or k[1, 1] <= 0 or abs(k[1, 0]) > 1e-8:
        raise ValueError("Invalid focal length or unsupported intrinsic convention")
    if not (0 <= k[0, 2] < width and 0 <= k[1, 2] < height):
        raise ValueError("Principal point outside declared raster")
    return {"K": k.tolist(), "width": width, "height": height,
            "scope": "numeric validation only; source provenance and projection convention still required"}


def validate_near_plane_cm(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise ValueError("near_plane_cm must be explicitly present, positive and finite")
    return float(value)


def decode_v3_raw(codes: np.ndarray, contract: dict):
    """Preserve native v3 codes and mask reserved/sentinel range; NOT metric.

    The delivered contract names log_sentinel_v3 but does not state its inverse
    equation or the unit of far_plane_encode. The bundled loader describes v2.
    Refuse to fabricate a physical scale from those incomplete specifications.
    """
    if contract.get("schema") != "originlab.depth.decode_contract/1" or contract.get("depth_transform") != "log_sentinel_v3":
        raise UnverifiedContractError("Expected reviewed log_sentinel_v3 contract")
    if contract.get("vui_full_range") is not True:
        raise UnverifiedContractError("Limited-range v3 requires a separately reviewed adapter")
    values = contract["transform_constants"]
    maximum = values["max_valid_code10"]
    sentinel = values["sentinel_code10"]
    if not (0 < maximum < sentinel == 1023) or values["K"] <= 0 or values["far_plane_encode"] <= 0:
        raise UnverifiedContractError("Invalid or unknown v3 constants")
    validate_near_plane_cm(contract.get("near_plane_cm"))
    array = np.asarray(codes)
    if array.dtype != np.uint16 or array.ndim != 2 or array.max() > 1023:
        raise ValueError("Native uint16 2D 10-bit luma codes required")
    return array.copy(), array <= maximum
