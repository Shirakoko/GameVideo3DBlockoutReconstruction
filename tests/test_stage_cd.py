"""Stage C/D contracts and synthetic five-class reliability, never real GT."""
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

from agents.object_detection import (AnnotationProvider, Detection, SyntheticColorProvider, confidence, mask_iou)
from agents.scene_builder import build_scene, json_write
from agents.synthetic_scene import make_scene_fixture
from agents.tracking import Observation, PersistentTracker, appearance, classify_motion
from reconstruction.object_geometry import frame_points, load_pixel_cloud, measure, static_bounds
from scene_schema import SceneValidationError, validate_scene
from scripts.understand_scene import rgb_records
from stage_a.depth import UnverifiedContractError


def simple_observation(index, time, position, *, category="person", color=(220, 50, 60), hint=None, source=None):
    mask = np.ones((8, 8), dtype=bool)
    c = confidence(0.8, "inferred", "test", "synthetic test only")
    detection = Detection(str(index), category, [0, 0, 8, 8], mask, "instance", c, c, hint)
    p = np.array(position, dtype=float)
    points = p + np.random.default_rng(42).uniform(-0.15, 0.15, (64, 3))
    m = dict(position=p, dimensions=np.array([0.6, 0.6, 1.8]), points=points)
    rgb = np.full((8, 8, 3), color, dtype=np.uint8)
    return Observation(index, time, source or str(index), detection, m, c, appearance(rgb, mask), "mask.png")


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        cls.frames, cls.truth = make_scene_fixture()
        cls.scene, cls.report = build_scene(rgb_records(cls.frames), SyntheticColorProvider(), cls.folder,
            scene_id="unit-five-category", data_source="SYNTHETIC ONLY", fps=60, metric_frames=cls.frames, min_depth_m=0.1, max_depth_m=35)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_five_categories_stable_ids_and_static_dynamic(self):
        self.assertEqual(self.report["status"], "complete")
        self.assertEqual(set(self.report["categories"]), set(SyntheticColorProvider.palette))
        self.assertEqual(len(self.scene["objects"]), 5)
        for o in self.scene["objects"]:
            self.assertEqual(len(o["observations"]), 6)
            self.assertIs(o["is_static"], o["category"] != "person")
            self.assertEqual(o["confidence"]["provenance"], "inferred")

    def test_rgb_detection_mask_precision_recall_fixture(self):
        provider = SyntheticColorProvider()
        for frame in self.frames:
            labels = self.truth[frame.frame_index]["labels"]
            detections = provider.detect(frame.rgb, frame.frame_index)
            self.assertEqual(len(detections), 5)
            for d in detections:
                self.assertEqual(mask_iou(d.mask, labels == d.category), 1.)

    def test_independent_depth_geometry_matches_raycaster(self):
        for frame in self.frames:
            p, uv = frame_points(frame, 0.1, 35)
            expected = self.truth[frame.frame_index]["points"][uv[:, 1], uv[:, 0]]
            np.testing.assert_allclose(p, expected, atol=1e-11)

    def test_person_positions_and_dimensions_from_geometry(self):
        person = next(o for o in self.scene["objects"] if o["category"] == "person")
        self.assertEqual(len(person["motion_track"]), 5)  # duplicate is not new evidence
        errors = []
        for r in person["observations"]:
            low, high = np.array(self.truth[r["frame_index"]]["boxes"]["person"])
            errors.append(float(np.linalg.norm(np.array(r["position"])-(low+high)/2)))
            self.assertTrue(np.all(np.array(r["dimensions"]) <= high-low+0.02))
        self.assertLess(max(errors), 0.35)  # surface centre, not full-volume centre
        self.assertGreater(person["motion_track"][-1]["transform"]["position"][0]-person["motion_track"][0]["transform"]["position"][0], 1.4)

    def test_dynamic_cloud_is_not_fused_over_time(self):
        person = next(o for o in self.scene["objects"] if o["category"] == "person")
        import open3d as o3d
        points = np.asarray(o3d.io.read_point_cloud(str(self.folder/person["geometry_source"]["point_cloud_path"])).points)
        self.assertLess(points[:, 0].max(), 4.7)
        self.assertGreater(person["motion_track"][-1]["transform"]["position"][0], 5.5)

    def test_duplicate_delivery_does_not_change_static_bounds(self):
        ground = next(o for o in self.scene["objects"] if o["category"] == "ground")
        measurements = []
        for frame in self.frames:
            if frame.frame_index == 2030:
                continue
            d = next(d for d in SyntheticColorProvider().detect(frame.rgb, frame.frame_index) if d.category == "ground")
            p, uv = frame_points(frame, 0.1, 35)
            measurements.append(measure(d, p, uv)[0])
        p, dims, _ = static_bounds(measurements)
        np.testing.assert_allclose(p, ground["transform"]["position"])
        np.testing.assert_allclose(dims, ground["dimensions"])

    def test_debug_masks_ply_and_roundtrip_json_exist(self):
        self.assertTrue((self.folder/"debug/world_tracks.png").is_file())
        loaded = json.loads((self.folder/"scene_schema.json").read_text(encoding="utf-8"))
        self.assertTrue(validate_scene(loaded)["valid"])
        for o in loaded["objects"]:
            self.assertTrue((self.folder/o["geometry_source"]["point_cloud_path"]).is_file())
            for r in o["observations"]:
                self.assertTrue((self.folder/r["mask"]["path"]).is_file())

    def test_relations_only_use_same_timestamp(self):
        self.assertTrue(self.scene["relationships"])
        by_id = {o["object_id"]: o for o in self.scene["objects"]}
        for r in self.scene["relationships"]:
            for oid in [r["subject_id"], r["object_id"]]:
                self.assertTrue(any(o["frame_index"] == r["frame_index"] and o["timestamp_seconds"] == r["timestamp_seconds"] for o in by_id[oid]["observations"]))

    def test_unverified_contract_fails_before_output(self):
        frames = [replace(self.frames[0], geometry_pose_verified=False)]
        output = self.folder/"must_not_exist"
        with self.assertRaises(UnverifiedContractError):
            build_scene(rgb_records(frames), SyntheticColorProvider(), output, scene_id="bad", data_source="SYNTHETIC", fps=60, metric_frames=frames, min_depth_m=0.1, max_depth_m=35)
        self.assertFalse(output.exists())

    def test_misordered_or_mismatched_records_rejected(self):
        rows = rgb_records(self.frames)
        rows[0]["timestamp_seconds"] = 0.2
        with self.assertRaisesRegex(ValueError, "contradict"):
            build_scene(rows, SyntheticColorProvider(), self.folder/"bad-record", scene_id="bad", data_source="SYNTHETIC", fps=60, metric_frames=self.frames, min_depth_m=0.1, max_depth_m=35)
        with self.assertRaisesRegex(ValueError, "increasing"):
            build_scene(rgb_records(self.frames)[::-1], SyntheticColorProvider(), self.folder/"bad-order", scene_id="bad", data_source="SYNTHETIC", fps=60)

    def test_rgb_only_explicitly_null_geometry_not_fake_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene, report = build_scene(rgb_records(self.frames), SyntheticColorProvider(), tmp, scene_id="rgb-only", data_source="SYNTHETIC RGB", fps=60)
        self.assertEqual(report["status"], "partial_geometry_unverified")
        for o in scene["objects"]:
            self.assertIsNone(o["transform"])
            self.assertIsNone(o["is_static"])
            self.assertEqual(o["motion_track"], [])
        self.assertIsNone(scene["cameras"][0]["track"][0]["world_from_camera"])


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.frame = make_scene_fixture()[0][0]
        self.d = next(d for d in SyntheticColorProvider().detect(self.frame.rgb, self.frame.frame_index) if d.category == "person")
        self.points, self.pixels = frame_points(self.frame, 0.1, 35)

    def test_bbox_proxy_and_missing_depth_never_create_geometry(self):
        d = replace(self.d, mask_kind="bbox_proxy")
        for points, pixels, detection in [(self.points, self.pixels, d), (None, None, self.d)]:
            m, c = measure(detection, points, pixels)
            self.assertIsNone(m)
            self.assertEqual(c["provenance"], "unobserved")

    def test_too_few_valid_mask_points_stay_unobserved(self):
        m, c = measure(self.d, self.points[:3], self.pixels[:3])
        self.assertIsNone(m)
        self.assertEqual(c["score"], 0)

    def test_pixel_cloud_contract_and_stale_pose_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"cloud.npz"
            np.savez(path, points_world_m=self.points, pixels_uv=self.pixels, frame_index=self.frame.frame_index)
            p, uv = load_pixel_cloud(path, self.frame, 0.1, 35)
            np.testing.assert_array_equal(p, self.points)
            np.savez(path, points_world_m=self.points+1, pixels_uv=self.pixels, frame_index=self.frame.frame_index)
            with self.assertRaisesRegex(ValueError, "contradicts"):
                load_pixel_cloud(path, self.frame, 0.1, 35)

    def test_metric_units_not_confused_with_world_units(self):
        spec = replace(self.frame.depth_spec, unit="cm", meters_per_unit=0.01)
        cm = replace(self.frame, depth=self.frame.depth*100, depth_spec=spec)
        p, _ = frame_points(cm, 0.1, 35)
        np.testing.assert_allclose(p, self.points, atol=1e-12)


class TrackingTests(unittest.TestCase):
    def test_crossing_objects_use_appearance_and_world_velocity(self):
        tracker = PersistentTracker("crossing")
        a0 = simple_observation(0, 0, [0, 0, 0])
        b0 = simple_observation(1, 0, [2, 0, 0], color=(30, 190, 70))
        tracker.update([a0, b0], 0)
        ids = [t.object_id for t in tracker.tracks]
        tracker.update([simple_observation(2, 1, [0.6, 0, 0]), simple_observation(3, 1, [1.4, 0, 0], color=(30, 190, 70))], 1)
        tracker.update([simple_observation(4, 2, [1.4, 0, 0]), simple_observation(5, 2, [0.6, 0, 0], color=(30, 190, 70))], 2)
        self.assertEqual(len(tracker.tracks), 2)
        self.assertEqual([t.object_id for t in tracker.tracks], ids)
        self.assertEqual(tracker.tracks[0].observations[-1].frame_index, 4)

    def test_occlusion_reappearance_and_reproducible_ids(self):
        ids = []
        for _ in range(2):
            tracker = PersistentTracker("repeatable")
            tracker.update([simple_observation(0, 0, [0, 0, 0])], 0)
            tracker.update([], 1)
            tracker.update([simple_observation(2, 2, [0.1, 0, 0])], 2)
            self.assertEqual(len(tracker.tracks), 1)
            ids.append(tracker.tracks[0].object_id)
        self.assertEqual(ids[0], ids[1])

    def test_ambiguous_matches_abstain(self):
        tracker = PersistentTracker("ambiguous")
        tracker.update([simple_observation(0, 0, [-1, 0, 0]), simple_observation(1, 0, [1, 0, 0])], 0)
        tracker.update([simple_observation(2, 1, [0, 0, 0])], 1)
        self.assertEqual(len(tracker.tracks), 3)

    def test_identity_hint_conflict_and_expiry(self):
        tracker = PersistentTracker("hint")
        tracker.update([simple_observation(0, 0, [0, 0, 0], hint="player")], 0)
        with self.assertRaisesRegex(ValueError, "changed category"):
            tracker.update([simple_observation(1, 1, [0, 0, 0], category="rock", hint="player")], 1)
        tracker = PersistentTracker("expiry", max_gap_seconds=1)
        tracker.update([simple_observation(0, 0, [0, 0, 0])], 0)
        tracker.update([simple_observation(2, 2, [0, 0, 0])], 2)
        self.assertEqual(len(tracker.tracks), 2)

    def test_duplicate_captures_not_three_motion_observations(self):
        tracker = PersistentTracker("dup")
        for i in range(3):
            tracker.update([simple_observation(i, float(i), [0, 0, 0], source="same")], i)
        value, c = classify_motion(tracker.tracks[0])
        self.assertIsNone(value)
        self.assertEqual(c["provenance"], "unobserved")


class SchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        frames, _ = make_scene_fixture()
        cls.scene, _ = build_scene(rgb_records(frames), SyntheticColorProvider(), cls.temp.name,
            scene_id="schema-fixture", data_source="SYNTHETIC ONLY", fps=60, metric_frames=frames, min_depth_m=0.1, max_depth_m=35)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_invalid_scene_contracts(self):
        person_index = next(i for i, o in enumerate(self.scene["objects"]) if o["category"] == "person")
        cases = [
            ("version", lambda s: s.update(schema_version="2.0.0")),
            ("unknown field", lambda s: s.update(unrecognized=True)),
            ("nonfinite units", lambda s: s.update(unit_scale=float("nan"))),
            ("invalid date", lambda s: s.update(created_at="yesterday")),
            ("duplicate id", lambda s: s["objects"][1].update(object_id=s["objects"][0]["object_id"])),
            ("dangling parent", lambda s: s["objects"][0].update(parent_id="missing")),
            ("parent cycle", lambda s: s["objects"][0].update(parent_id=s["objects"][0]["object_id"])),
            ("nonunit quaternion", lambda s: s["objects"][0]["transform"].update(rotation=[0, 0, 0, 2])),
            ("negative dimension", lambda s: s["objects"][0].update(dimensions=[1, -1, 1])),
            ("camera reflection", lambda s: s["cameras"][0]["track"][0].update(world_from_camera=np.diag([-1., 1., 1., 1.]).tolist())),
            ("invalid timeline", lambda s: s["timeline"].update(end_seconds=-1)),
            ("reversed camera track", lambda s: s["cameras"][0]["track"].reverse()),
            ("observation time", lambda s: s["objects"][0]["observations"][0].update(timestamp_seconds=0.1)),
            ("unknown camera", lambda s: s["objects"][0]["visibility"][0].update(camera_id="missing")),
            ("unknown source frame", lambda s: s["objects"][0].update(source_frames=[99999])),
            ("unverified geometry", lambda s: s["reconstruction_metadata"].update(geometry_contract_verified=False)),
            ("unobserved geometry", lambda s: s["objects"][0]["geometry_source"]["confidence"].update(provenance="unobserved")),
            ("moving static object", lambda s: s["objects"][person_index].update(is_static=True)),
            ("missing motion", lambda s: s["objects"][person_index].update(motion_track=[])),
            ("bbox proxy", lambda s: s["objects"][0]["observations"][0]["mask"].update(kind="bbox_proxy")),
        ]
        for name, change in cases:
            with self.subTest(case=name):
                scene = deepcopy(self.scene)
                change(scene)
                with self.assertRaises(SceneValidationError):
                    validate_scene(scene)

    def test_multiple_camera_tracks_supported(self):
        scene = deepcopy(self.scene)
        camera = deepcopy(scene["cameras"][0])
        camera["camera_id"] = "camera_secondary"
        scene["cameras"].append(camera)
        extra = deepcopy(scene["timeline"]["frames"])
        for row in extra:
            row["camera_id"] = "camera_secondary"
        scene["timeline"]["frames"].extend(extra)
        self.assertEqual(validate_scene(scene)["cameras"], 2)


class AnnotationAndCliTests(unittest.TestCase):
    def test_synthetic_cli_and_generic_metric_replay(self):
        from scripts.understand_scene import main
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--synthetic", "--output", str(folder/"demo")]), 0)
                self.assertEqual(main(["--input-manifest", str(folder/"demo/synthetic_input_manifest.json"),
                    "--annotations", str(folder/"demo/synthetic_annotations.json"), "--scene-id", "generic-replay", "--fps", "60",
                    "--min-depth-m", "0.1", "--max-depth-m", "35", "--output", str(folder/"replay")]), 0)
            report = json.loads((folder/"replay/validation_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["object_count"], 5)
            self.assertEqual(report["dynamic"], 1)
            self.assertEqual(report["static"], 4)

    def test_polygon_raster_and_mask_bbox_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"a.json"
            document = dict(schema="stage_c.rgb_annotations/1", evidence="SYNTHETIC assisted test", frames=[dict(frame_index=1, raster_size=[10, 10], objects=[dict(key="one", category="rock", score=0.8, mask=dict(kind="polygon", vertices=[[1, 1],[4, 1],[4, 4],[1, 4]]))])])
            json_write(path, document)
            provider = AnnotationProvider(path)
            rgb = np.zeros((10, 10, 3), dtype=np.uint8)
            self.assertEqual(provider.detect(rgb, 1)[0].bbox, [1, 1, 5, 5])
            with self.assertRaisesRegex(ValueError, "raster"):
                provider.detect(rgb[:5], 1)

    def test_cli_missing_depth_limits_and_unknown_schema_exit_one(self):
        from scripts.understand_scene import main
        from scripts.validate_scene import main as validate_main
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--input-manifest", "not-read.json", "--scene-id", "test", "--fps", "60", "--annotations", "not-read.json"]), 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"bad.json"
            json_write(path, dict(schema_version="bad"))
            with redirect_stderr(io.StringIO()):
                self.assertEqual(validate_main([str(path)]), 1)

    def test_real_annotation_contract_is_assisted_not_gt(self):
        path = Path(__file__).resolve().parents[1]/"config/sample_66ad_annotations.json"
        provider = AnnotationProvider(path)
        d = provider.detect(np.zeros((1080, 1920, 3), dtype=np.uint8), 0)
        self.assertEqual({o.category for o in d}, set(SyntheticColorProvider.palette))
        self.assertTrue(all(o.confidence["provenance"] == "inferred" for o in d))


if __name__ == "__main__":
    unittest.main()
