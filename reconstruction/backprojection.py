"""Depth back-projection with a required, evidence-bearing physical contract."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stage_a.depth import UnverifiedContractError, validate_intrinsics


@dataclass(frozen=True)
class DepthSpec:
    unit: str
    meters_per_unit: float
    depth_type: str
    pixel_center_offset: float
    verified: bool
    evidence: str

    def validate(self):
        if self.verified is not True or not isinstance(self.evidence, str) or not self.evidence.strip():
            raise UnverifiedContractError("Depth units, inverse curve and sampling convention require explicit verification evidence")
        if self.depth_type != "planar_z":
            raise UnverifiedContractError("Only reviewed planar_z supported; ray length / disparity / log codes are NOT planar depth")
        # A custom world unit is allowed only with its explicit reviewed scale.
        if not isinstance(self.unit, str) or not self.unit.strip() or self.unit.lower() in ("unknown", "relative", "codes", "code10"):
            raise UnverifiedContractError("Unknown/relative/encoded depth cannot be fused with metric poses")
        if isinstance(self.meters_per_unit, bool) or not np.isfinite(self.meters_per_unit) or self.meters_per_unit <= 0:
            raise UnverifiedContractError("Explicit positive depth-unit-to-metres conversion required")
        known = {"m": 1., "cm": 0.01, "mm": 0.001}
        if self.unit in known and not np.isclose(self.meters_per_unit, known[self.unit], atol=0, rtol=1e-12):
            raise UnverifiedContractError("Declared depth unit and metres conversion contradict each other")
        if isinstance(self.pixel_center_offset, bool) or self.pixel_center_offset not in (0., 0.5):
            raise UnverifiedContractError("Explicit pixel-centre convention (0 or 0.5) required")


@dataclass
class Backprojection:
    points_camera_m: np.ndarray
    colors_rgb8: np.ndarray
    pixels_uv: np.ndarray
    stats: dict


def backproject(depth, intrinsics, *, spec: DepthSpec, valid_mask, rgb,
                min_depth_m: float, max_depth_m: float, stride: int = 1) -> Backprojection:
    """p_optical = planar_z_m * inv(K) @ [u+offset,v+offset,1].

    K belongs to this exact raster; RGB must already be calibrated/aligned.
    Masks must exclude sky, saturated far clip and invalid decoder values.
    We never guess a log inverse, denominator, unit, or RGB resize rule.
    """
    spec.validate()
    d = np.asarray(depth)
    if d.ndim != 2 or not np.issubdtype(d.dtype, np.floating):
        raise UnverifiedContractError("Decoded floating planar depth required, not integer native HEVC codes")
    h, w = d.shape
    validate_intrinsics(intrinsics, width=w, height=h)
    k = np.asarray(intrinsics, dtype=np.float64)
    mask = np.asarray(valid_mask)
    colors = np.asarray(rgb)
    if mask.shape != d.shape or mask.dtype != np.bool_:
        raise ValueError("Explicit boolean depth validity mask at native depth raster required")
    if colors.shape != (h, w, 3) or colors.dtype != np.uint8:
        raise ValueError("Aligned native-raster RGB uint8 required; no implicit resize")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("Stride must be a positive integer")
    if not np.isfinite([min_depth_m, max_depth_m]).all() or not 0 < min_depth_m < max_depth_m:
        raise ValueError("Explicit finite 0 < min_depth_m < max_depth_m required")
    # Downsample pixel locations, not K or depth values.
    vv, uu = np.mgrid[0:h:stride, 0:w:stride]
    with np.errstate(over="ignore", invalid="ignore"):
        z = d[vv, uu].astype(np.float64) * spec.meters_per_unit
    allowed = mask[vv, uu]
    finite = np.isfinite(z)
    inside = finite & (z >= min_depth_m) & (z <= max_depth_m)
    keep = allowed & inside
    uv = np.column_stack([uu[keep], vv[keep]]).astype(np.int32)
    pixels = np.column_stack([uv.astype(np.float64) + spec.pixel_center_offset, np.ones(len(uv))])
    rays = pixels @ np.linalg.inv(k).T
    points = rays * z[keep, None]
    return Backprojection(points, colors[vv[keep], uu[keep]].copy(), uv,
        {"raster_pixels": h*w, "sampled_pixels": int(keep.size), "valid_points": int(keep.sum()),
         "masked_pixels": int((~allowed).sum()), "nonfinite_unmasked": int((allowed & ~finite).sum()),
         "range_rejected_unmasked": int((allowed & finite & ~inside).sum()), "stride": stride,
         "depth_unit": spec.unit, "meters_per_unit": spec.meters_per_unit,
         "pixel_center_offset": spec.pixel_center_offset})
