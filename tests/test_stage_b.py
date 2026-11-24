"""Known geometry tests; never certify real dataset units from a test fixture."""
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import asdict
from pathlib import Path

import numpy as np

from reconstruction.backprojection import DepthSpec, backproject
from reconstruction.coordinate_transform import (BLENDER_FROM_UE, UE_FROM_OPTICAL,
    blender_camera_from_optical_pose, inverse_rigid, transform_points,
    ue_rotator_matrix, validate_rigid, world_from_optical_ue_pose)
from reconstruction.pointcloud_fusion import FrameInput, fuse_frames, write_verified_ply
from stage_a.depth import UnverifiedContractError


def synthetic_spec(**changes):
    fields = dict(unit="cm", meters_per_unit=0.01, depth_type="planar_z",
                  pixel_center_offset=0., verified=True, evidence="synthetic analytical planar depth in centimetres")
    return DepthSpec(**(fields | changes))


def fixture_frame(index=100, timestamp=0., *, position=(0, 0, 100), depth=None, observation=None):
    d = np.full((8, 8), 200., dtype=np.float64) if depth is None else depth
    return FrameInput(index, index, index, index+900, timestamp, d,
        np.ones(d.shape, dtype=bool), np.full((*d.shape, 3), 128, dtype=np.uint8),
        np.array([[20., 0., 4.], [0., 20., 4.], [0., 0., 1.]]),
        world_from_optical_ue_pose(position, pitch_deg=0, yaw_deg=0, roll_deg=0, world_units_per_meter=100),
        synthetic_spec(), True, "synthetic renderer: pose and pixels from the same observation", observation or f"synthetic-{index}")


class CoordinateTests(unittest.TestCase):
    def test_known_axis_rotations_not_generic_xyz(self):
        cases = [(dict(pitch_deg=0, yaw_deg=90, roll_deg=0), [1, 0, 0], [0, 1, 0]),
                 (dict(pitch_deg=90, yaw_deg=0, roll_deg=0), [1, 0, 0], [0, 0, 1]),
                 (dict(pitch_deg=0, yaw_deg=0, roll_deg=90), [0, 1, 0], [0, 0, -1])]
        for angles, point, expected in cases:
            np.testing.assert_allclose(ue_rotator_matrix(**angles) @ point, expected, atol=1e-12)

    def test_mixed_rotation_against_independent_ue_basis_formula(self):
        # Independent forward/right/up closed form, not matrix product under test.
        for p, y, r in [(30., 45., 60.), (-20., 110., -37.), (89.9, -179., 90.)]:
            sp, sy, sr = np.sin(np.deg2rad([p, y, r]))
            cp, cy, cr = np.cos(np.deg2rad([p, y, r]))
            forward = [cp*cy, cp*sy, sp]
            right = [sr*sp*cy-cr*sy, sr*sp*sy+cr*cy, -sr*cp]
            up = [-cr*sp*cy-sr*sy, -cr*sp*sy+sr*cy, cr*cp]
            np.testing.assert_allclose(ue_rotator_matrix(pitch_deg=p, yaw_deg=y, roll_deg=r),
                                       np.column_stack([forward, right, up]), atol=1e-12)

    def test_handedness_and_known_blender_point(self):
        self.assertAlmostEqual(np.linalg.det(UE_FROM_OPTICAL), -1.)
        self.assertAlmostEqual(np.linalg.det(BLENDER_FROM_UE), -1.)
        # UE camera at (100,200,300) units, looking +X; optical point (1,2,5)m.
        t = world_from_optical_ue_pose([100, 200, 300], pitch_deg=0, yaw_deg=0, roll_deg=0, world_units_per_meter=100)
        np.testing.assert_allclose(transform_points([[1, 2, 5]], t), [[6, -3, 1]], atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(t[:3, :3]), 1.)

    def test_matrix_direction_and_roundtrip(self):
        rng = np.random.default_rng(41)
        for _ in range(20):
            p, y, r = rng.uniform(-180, 180, 3)
            t = world_from_optical_ue_pose(rng.normal(size=3)*1000, pitch_deg=p, yaw_deg=y, roll_deg=r, world_units_per_meter=100)
            points = rng.normal(size=(20, 3))
            world = transform_points(points, t)
            homogeneous = np.column_stack([points, np.ones(len(points))])
            np.testing.assert_allclose(world, (t @ homogeneous.T).T[:, :3], atol=1e-12)
            np.testing.assert_allclose(transform_points(world, inverse_rigid(t)), points, atol=1e-12)
            np.testing.assert_allclose(inverse_rigid(t) @ t, np.eye(4), atol=1e-12)

    def test_blender_camera_local_basis(self):
        t = world_from_optical_ue_pose([100, 200, 300], pitch_deg=10, yaw_deg=-31, roll_deg=17, world_units_per_meter=100)
        camera = blender_camera_from_optical_pose(t)
        np.testing.assert_allclose(transform_points([[1, -2, -5]], camera), transform_points([[1, 2, 5]], t), atol=1e-12)
        np.testing.assert_allclose(camera[:3, 3], t[:3, 3])

    def test_invalid_transform_and_units_rejected(self):
        for matrix in (np.diag([1., -1., 1., 1.]), np.diag([2., 1., 1., 1.]), np.full((4, 4), np.nan)):
            with self.assertRaises(ValueError):
                validate_rigid(matrix)
        with self.assertRaises(ValueError):
            world_from_optical_ue_pose([0, 0, 0], pitch_deg=0, yaw_deg=0, roll_deg=0, world_units_per_meter=0)


class BackprojectionTests(unittest.TestCase):
    def test_exact_pinhole_point_and_pixel_offset(self):
        k = np.array([[2., 0., 1.], [0., 2., 1.], [0., 0., 1.]])
        mask = np.zeros((3, 3), dtype=bool)
        mask[2, 2] = True
        rgb = np.full((3, 3, 3), [1, 2, 3], dtype=np.uint8)
        for offset, expected in [(0., [1., 1., 2.]), (0.5, [1.5, 1.5, 2.])]:
            bp = backproject(np.full((3, 3), 200.), k, spec=synthetic_spec(pixel_center_offset=offset),
                valid_mask=mask, rgb=rgb, min_depth_m=0.1, max_depth_m=10)
            np.testing.assert_allclose(bp.points_camera_m, [expected])
            np.testing.assert_array_equal(bp.pixels_uv, [[2, 2]])
            np.testing.assert_array_equal(bp.colors_rgb8, [[1, 2, 3]])

    def test_invalid_sky_far_and_nonfinite_filtering(self):
        d = np.array([[0., -1., np.nan, np.inf, 200., 10000., 400.]])
        mask = np.ones(d.shape, dtype=bool)
        mask[0, 6] = False
        bp = backproject(d, [[1, 0, 3], [0, 1, 0], [0, 0, 1]], spec=synthetic_spec(),
            valid_mask=mask, rgb=np.zeros((1, 7, 3), dtype=np.uint8), min_depth_m=0.1, max_depth_m=10)
        self.assertEqual(bp.stats["valid_points"], 1)
        self.assertEqual(bp.stats["masked_pixels"], 1)
        self.assertEqual(bp.stats["nonfinite_unmasked"], 2)
        np.testing.assert_array_equal(bp.pixels_uv, [[4, 0]])

    def test_reprojection_with_skew_and_stride(self):
        frame = fixture_frame()
        frame.intrinsics[0, 1] = 0.3
        bp = backproject(frame.depth, frame.intrinsics, spec=frame.depth_spec,
            valid_mask=frame.valid_mask, rgb=frame.rgb, min_depth_m=0.1, max_depth_m=10, stride=2)
        projected = bp.points_camera_m @ frame.intrinsics.T
        np.testing.assert_allclose(projected[:, :2]/projected[:, 2, None], bp.pixels_uv, atol=1e-12)
        self.assertEqual(len(bp.points_camera_m), 16)

    def test_unknown_units_curve_and_integer_codes_fail_closed(self):
        frame = fixture_frame()
        for spec in (synthetic_spec(verified=False), synthetic_spec(unit="unknown"),
                     synthetic_spec(unit="relative"), synthetic_spec(depth_type="ray_length"),
                     synthetic_spec(meters_per_unit=1), synthetic_spec(evidence="")):
            with self.assertRaises(UnverifiedContractError):
                backproject(frame.depth, frame.intrinsics, spec=spec, valid_mask=frame.valid_mask,
                            rgb=frame.rgb, min_depth_m=0.1, max_depth_m=10)
        with self.assertRaises(UnverifiedContractError):
            backproject(frame.depth.astype(np.uint16), frame.intrinsics, spec=frame.depth_spec,
                valid_mask=frame.valid_mask, rgb=frame.rgb, min_depth_m=0.1, max_depth_m=10)

    def test_raster_and_stride_mismatches_fail(self):
        f = fixture_frame()
        for rgb, stride in [(f.rgb[:4], 1), (f.rgb, 0), (f.rgb, True)]:
            with self.assertRaises(ValueError):
                backproject(f.depth, f.intrinsics, spec=f.depth_spec, valid_mask=f.valid_mask,
                            rgb=rgb, min_depth_m=0.1, max_depth_m=10, stride=stride)


class FusionTests(unittest.TestCase):
    def fuse(self, frames, **changes):
        return fuse_frames(frames, **(dict(voxel_size_m=0.1, min_depth_m=0.1, max_depth_m=100,
                           sor_neighbors=20, sor_std_ratio=100) | changes))

    def test_same_static_plane_from_two_translating_poses(self):
        a = fixture_frame()
        b = fixture_frame(102, 0.1, position=(100, 0, 100), depth=np.full((8, 8), 100.))
        result = self.fuse([a, b])
        points = np.asarray(result.cloud.points)
        np.testing.assert_allclose(points[:, 0], 2., atol=1e-12)
        self.assertEqual(result.stats["accepted_observations"], 2)
        self.assertIn(100, result.provenance_frame_indices)
        self.assertIn(102, result.provenance_frame_indices)

    def test_source_and_exact_duplicates_do_not_add_weight(self):
        a = fixture_frame(observation="capture-1")
        b = fixture_frame(101, 0.1, observation="capture-1")
        c = fixture_frame(102, 0.2)
        reference = self.fuse([a])
        result = self.fuse([a, b, c])
        self.assertEqual(result.stats["accepted_observations"], 1)
        self.assertEqual(result.stats["skipped_duplicate_frames"], 2)
        self.assertEqual(result.frame_reports[1]["status"], "skipped_source_duplicate")
        self.assertEqual(result.frame_reports[2]["status"], "skipped_exact_geometric_input")
        np.testing.assert_allclose(result.cloud.points, reference.cloud.points)

    def test_identical_depth_under_moving_pose_not_discarded(self):
        a = fixture_frame()
        b = fixture_frame(101, 0.1, position=(100, 0, 100))
        result = self.fuse([a, b])
        self.assertEqual(result.stats["accepted_observations"], 2)
        self.assertGreater(np.ptp(np.asarray(result.cloud.points)[:, 0]), 0.9)

    def test_inconsistent_source_capture_fails(self):
        a = fixture_frame(observation="same")
        b = fixture_frame(101, 0.1, position=(100, 0, 100), observation="same")
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.fuse([a, b])

    def test_per_observation_not_pixel_count_weighting(self):
        # Keep the footprint away from grid boundaries at y=0 / z=0.
        a = fixture_frame(position=(0, 5000, 5000), depth=np.full((8, 8), 201.))
        b = fixture_frame(101, 0.1, position=(0, 5000, 5000), depth=np.full((8, 8), 209.))
        b.valid_mask[:] = False
        b.valid_mask[0, 0] = True
        # Whole scene in a single voxel: 64 pixels vs 1 pixel, equal frame means.
        result = self.fuse([a, b], voxel_size_m=100.)
        np.testing.assert_allclose(np.asarray(result.cloud.points)[:, 0], [2.05], atol=1e-12)
        np.testing.assert_array_equal(result.provenance_offsets, [0, 2])
        np.testing.assert_array_equal(result.provenance_frame_indices, [100, 101])

    def test_sor_removes_remote_spike_and_keeps_provenance_aligned(self):
        frame = fixture_frame()
        frame.depth[0, 0] = 5000.
        result = self.fuse([frame], sor_neighbors=8, sor_std_ratio=1.)
        self.assertGreaterEqual(result.stats["outliers_removed"], 1)
        self.assertLess(np.max(np.asarray(result.cloud.points)[:, 0]), 10)
        self.assertEqual(len(result.provenance_offsets), len(result.cloud.points)+1)
        self.assertEqual(result.provenance_offsets[-1], len(result.provenance_frame_indices))

    def test_missing_pose_binding_and_frame_order_rejected(self):
        frame = fixture_frame()
        frame.geometry_pose_verified = False
        with self.assertRaises(UnverifiedContractError):
            self.fuse([frame])
        with self.assertRaises(ValueError):
            self.fuse([fixture_frame(), fixture_frame(99, 0.1)])
        frame = fixture_frame()
        frame.depth_frame_index += 1
        with self.assertRaises(ValueError):
            self.fuse([frame])

    def test_ply_roundtrip_and_provenance(self):
        result = self.fuse([fixture_frame()])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"single.ply"
            receipt = write_verified_ply(path, result.cloud, data_source="SYNTHETIC ONLY")
            self.assertTrue(receipt["roundtrip_verified"])
            self.assertIn(b"comment units meter", path.read_bytes()[:400])
        np.testing.assert_array_equal(np.unique(result.provenance_frame_indices), [100])

    def test_vertex_provenance_tracks_geometry_after_sor(self):
        result = self.fuse([fixture_frame(), fixture_frame(101, 0.1, position=(200, 0, 100))],
                           sor_neighbors=8, sor_std_ratio=2.)
        for i, point in enumerate(np.asarray(result.cloud.points)):
            members = result.provenance_frame_indices[result.provenance_offsets[i]:result.provenance_offsets[i+1]]
            expected = 100 if point[0] < 3 else 101
            np.testing.assert_array_equal(members, [expected])

    def test_point_budget_and_empty_depth_fail(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            self.fuse([fixture_frame()], max_input_points=10)
        frame = fixture_frame()
        frame.valid_mask[:] = False
        with self.assertRaisesRegex(ValueError, "No valid depth"):
            self.fuse([frame])


class ManifestTests(unittest.TestCase):
    def create_manifest(self, folder, *, spec=None, depth=None):
        from scripts.reconstruct_scene import json_write
        f = fixture_frame(depth=depth)
        np.savez_compressed(folder/"frame.npz", depth=f.depth, valid_mask=f.valid_mask, rgb=f.rgb, K=f.intrinsics)
        row = {key: getattr(f, key) for key in ("frame_index", "rgb_frame_index", "depth_frame_index", "original_tick", "timestamp_seconds",
                                             "geometry_pose_verified", "sync_evidence", "source_observation_id")}
        row.update(data_npz="frame.npz", world_from_camera=f.world_from_camera.tolist())
        manifest = {"schema": "stage_b.metric_frames/1", "data_source": "SYNTHETIC ONLY",
                    "coordinate_system": "blender_rh_z_up", "pose_direction": "world_from_optical_camera",
                    "pose_translation_unit": "m", "depth_spec": asdict(spec or f.depth_spec), "frames": [row]}
        path = folder/"manifest.json"
        json_write(path, manifest)
        return path, manifest

    def test_reviewed_manifest_readback(self):
        from scripts.reconstruct_scene import load_metric_manifest
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.create_manifest(Path(directory))
            frames, metadata = load_metric_manifest(path)
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0].original_tick, 1000)
            self.assertEqual(metadata["data_source"], "SYNTHETIC ONLY")

    def test_unknown_depth_contract_rejected_before_geometry(self):
        from scripts.reconstruct_scene import load_metric_manifest
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.create_manifest(Path(directory), spec=synthetic_spec(verified=False))
            with self.assertRaises(UnverifiedContractError):
                load_metric_manifest(path)

    def test_view_matrix_direction_cannot_be_assumed(self):
        from scripts.reconstruct_scene import json_write, load_metric_manifest
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = self.create_manifest(Path(directory))
            manifest["pose_direction"] = "camera_from_world"
            json_write(path, manifest)
            with self.assertRaisesRegex(ValueError, "camera-to-world"):
                load_metric_manifest(path)

    def test_native_integer_codes_in_manifest_fail_backprojection(self):
        from scripts.reconstruct_scene import load_metric_manifest
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.create_manifest(Path(directory), depth=np.full((8, 8), 166, dtype=np.uint16))
            frames, _ = load_metric_manifest(path)
            with self.assertRaises(UnverifiedContractError):
                fuse_frames(frames, voxel_size_m=0.1, min_depth_m=0.1, max_depth_m=10)

    def test_cli_no_implicit_clip_plane_for_reviewed_input(self):
        from scripts.reconstruct_scene import main
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.create_manifest(Path(directory))
            output = Path(directory)/"out"
            error_log = io.StringIO()
            with redirect_stderr(error_log):
                self.assertEqual(main(["--input-manifest", str(path), "--output", str(output)]), 1)
            self.assertIn("explicit --min-depth-m", error_log.getvalue())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
