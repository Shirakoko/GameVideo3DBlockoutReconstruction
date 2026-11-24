"""Stage E/F contracts plus real Blender background integration (synthetic)."""
from copy import deepcopy
from contextlib import redirect_stdout
from fractions import Fraction
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from agents.object_detection import PRIMITIVES, SyntheticColorProvider
from agents.scene_builder import build_scene
from agents.synthetic_scene import make_scene_fixture
from blender.media import command, probe_video
from blender.pipeline import blender_path, cached_outputs_match, json_write, main, run_pipeline
from blender.planner import load_camera_poses, prepare_plan, source_scene_snapshot
from blender.timeline import frame_mapping, matrix_quaternion, sample_camera, slerp
from reconstruction.coordinate_transform import ue_rotator_matrix
from scripts.understand_scene import rgb_records
from stage_a.media import ffmpeg_path


def video_spec(count=241, fps="60", size=(240, 140)):
    return dict(path="synthetic.mp4", width=size[0], height=size[1], fps=fps, start_frame=0,
                first_pts_seconds=0., frame_count=count, duration_seconds=count/float(Fraction(fps)))


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)
        frames, _ = make_scene_fixture()
        cls.scene, _ = build_scene(rgb_records(frames), SyntheticColorProvider(), cls.folder,
            scene_id="synthetic-ef-tests", data_source="SYNTHETIC ONLY", fps=60,
            metric_frames=frames, min_depth_m=.1, max_depth_m=35)
        cls.scene_path = cls.folder/"scene_schema.json"

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def scene_file(self, scene, name="modified.json"):
        path = self.folder/name
        json_write(path, scene)
        return path

    def test_exact_timeline_nonzero_source_indices(self):
        mapping = frame_mapping(self.scene["timeline"], video_spec(), "camera_main")
        self.assertEqual(len(mapping), 241)  # inclusive last PTS, not 240 or 242
        self.assertEqual(mapping[0]["source_frame_index"], 2000)
        self.assertEqual(mapping[-1]["source_frame_index"], 2240)
        self.assertEqual(mapping[-1]["blender_frame"], 241)
        self.assertEqual(mapping[-1]["timestamp_seconds"], 4.)
        self.assertIsNone(mapping[1]["original_tick"])
        self.assertEqual(mapping[0]["original_tick"], 8000)

    def test_time_origin_is_not_frame_index_over_fps(self):
        timeline = deepcopy(self.scene["timeline"])
        timeline["start_seconds"] += 12
        timeline["end_seconds"] += 12
        for row in timeline["frames"]:
            row["timestamp_seconds"] += 12
        mapping = frame_mapping(timeline, video_spec(), "camera_main")
        self.assertEqual(mapping[0]["timestamp_seconds"], 12)
        self.assertEqual(mapping[0]["source_frame_index"], 2000)

    def test_fps_and_duration_mismatch_refused(self):
        for spec in [video_spec(fps="30"), video_spec(count=242), video_spec(count=240)]:
            with self.assertRaises(ValueError):
                frame_mapping(self.scene["timeline"], spec, "camera_main")

    def test_source_index_not_guessed(self):
        timeline = deepcopy(self.scene["timeline"])
        timeline["frames"][1]["frame_index"] += 1
        with self.assertRaisesRegex(ValueError, "lattice"):
            frame_mapping(timeline, video_spec(), "camera_main")

    def test_quaternion_half_turn_and_shortest_arc(self):
        t = np.eye(4)
        t[:3, :3] = ue_rotator_matrix(pitch_deg=0, yaw_deg=180, roll_deg=0)
        q = matrix_quaternion(t)
        np.testing.assert_allclose(np.abs(q), [0, 0, 1, 0], atol=1e-8)
        np.testing.assert_allclose(slerp(q, -q, .5), q, atol=1e-8)
        q = slerp([0, 0, 0, 1], [0, 0, 1, 0], .5)
        np.testing.assert_allclose(q, [0, 0, 2**-.5, 2**-.5], atol=1e-8)

    def test_camera_samples_and_no_extrapolation(self):
        rows = self.scene["cameras"][0]["track"]
        sample = sample_camera(rows, .75)
        np.testing.assert_allclose(sample["position"], [0.05, .16, 2.2])
        with self.assertRaisesRegex(ValueError, "extrapolation"):
            sample_camera(rows, 5)

    def test_plan_per_frame_static_dynamic_reference(self):
        plan = prepare_plan(self.scene_path, video_spec(), resolution=(120, 70))
        self.assertEqual(plan["frame_end"], 241)
        self.assertEqual(len(plan["cameras"][0]["samples"]), 241)
        self.assertEqual(plan["cameras"][0]["samples"][0]["intrinsics"][0][0], 67.5)
        self.assertEqual(len(plan["reference_points"]) > 0, True)
        self.assertLessEqual(len(plan["reference_points"]), 50000)
        for o in plan["objects"]:
            self.assertEqual(len(o["samples"]), 0 if o["is_static"] else 241)
        person = next(o for o in plan["objects"] if o["category"] == "person")
        self.assertGreater(person["samples"][-1]["transform"]["position"][0]-person["samples"][0]["transform"]["position"][0], 1.4)

    def test_unverified_geometry_never_falls_back(self):
        scene = deepcopy(self.scene)
        scene["reconstruction_metadata"]["geometry_contract_verified"] = False
        with self.assertRaises(ValueError):
            prepare_plan(self.scene_file(scene), video_spec())

    def test_invalid_camera_and_render_parameters(self):
        scene = deepcopy(self.scene)
        scene["cameras"][0]["track"][0]["intrinsics"][0][1] = .2
        with self.subTest(case="skew"), self.assertRaisesRegex(ValueError, "skew"):
            prepare_plan(self.scene_file(scene), video_spec())
        with self.subTest(case="native raster"), self.assertRaisesRegex(ValueError, "native raster"):
            prepare_plan(self.scene_path, video_spec(size=(120, 70)))
        for value in (True, 1.5, 0, "16"):
            with self.subTest(samples=value), self.assertRaisesRegex(ValueError, "Positive integer"):
                prepare_plan(self.scene_path, video_spec(), samples=value)
        for value, error in (([], "exactly"), ([96], "exactly"), ([96, 56, 8], "exactly"),
                             ("96x56", "exactly"), ({}, "exactly"), ((121, 70), "even")):
            with self.subTest(resolution=value), self.assertRaisesRegex(ValueError, error):
                prepare_plan(self.scene_path, video_spec(), resolution=value)

    def test_raw_telemetry_manifest_is_not_pose_contract(self):
        raw = self.scene_file(dict(coordinate_system="lh_z_up", frames=[]), "raw.json")
        with self.assertRaisesRegex(ValueError, "raw telemetry refused"):
            load_camera_poses(raw)

    def test_reviewed_ue_contract_converts_units_axes(self):
        document = dict(schema="stage_e.camera_poses/1", verified=True, evidence="synthetic known UE contract and sync",
            pose_format="ue_frotator_degrees", coordinate_system="ue_lh_z_up", world_units_per_meter=100,
            samples=[dict(camera_id="camera_main", frame_index=2000, timestamp_seconds=0,
                pose=dict(position=[100, 200, 300], pitch=0, yaw=0, roll=0), intrinsics=[[10, 0, 5], [0, 10, 5], [0, 0, 1]])])
        rows = load_camera_poses(self.scene_file(document, "ue.json"))["camera_main"]
        np.testing.assert_allclose(np.array(rows[0]["world_from_camera"])[:3, 3], [1, -2, 3])
        np.testing.assert_allclose(np.linalg.det(np.array(rows[0]["world_from_camera"])[:3, :3]), 1)

    def test_missing_cloud_fails_instead_of_unmeasured_proxy(self):
        scene = deepcopy(self.scene)
        scene["objects"][0]["geometry_source"]["point_cloud_path"] = "missing.ply"
        with self.assertRaisesRegex(ValueError, "Missing geometric source"):
            prepare_plan(self.scene_file(scene), video_spec())

    def test_dense_original_tick_and_pose_are_preserved(self):
        rows = deepcopy(self.scene["cameras"][0]["track"])
        for row in rows:
            row.pop("confidence")
            row["camera_id"] = "camera_main"
            row["original_tick"] = 8000+row["frame_index"]-2000
        # Add a render-only original pose/tick that is absent from Scene's
        # sparse timeline; it must replace interpolation at its exact frame.
        middle = deepcopy(rows[0])
        middle.update(frame_index=2001, timestamp_seconds=1/60, original_tick=8001)
        middle["world_from_camera"][0][3] = .02
        rows.insert(1, middle)
        document = dict(schema="stage_e.camera_poses/1", verified=True, evidence="SYNTHETIC reviewed dense pose", pose_format="world_from_optical_camera",
                        coordinate_system="blender_rh_z_up", translation_unit="m", samples=rows)
        path = self.scene_file(document, "dense.json")
        plan = prepare_plan(self.scene_path, video_spec(), camera_poses=path)
        self.assertEqual(plan["frame_mapping"][1]["original_tick"], 8001)
        self.assertEqual(plan["cameras"][0]["samples"][1]["position"][0], .02)
        document["samples"][0]["original_tick"] += 1
        with self.assertRaisesRegex(ValueError, "tick disagrees"):
            prepare_plan(self.scene_path, video_spec(), camera_poses=self.scene_file(document, "dense_wrong.json"))

    def test_multiple_cameras_and_explicit_active_camera(self):
        scene = deepcopy(self.scene)
        secondary = deepcopy(scene["cameras"][0])
        secondary["camera_id"] = "secondary"
        for row in secondary["track"]:
            row["frame_index"] += 5000
        scene["cameras"].append(secondary)
        for row in deepcopy(scene["timeline"]["frames"]):
            row["camera_id"] = "secondary"
            row["frame_index"] += 5000
            scene["timeline"]["frames"].append(row)
        plan = prepare_plan(self.scene_file(scene, "multi.json"), video_spec(), active_camera="secondary")
        self.assertEqual(len(plan["cameras"]), 2)
        self.assertEqual(plan["active_camera"], "secondary")
        self.assertEqual(plan["frame_mapping"][0]["source_frame_index"], 7000)

    def test_reviewed_pose_schema_rejects_boolean_ticks(self):
        rows = deepcopy(self.scene["cameras"][0]["track"])
        for row in rows:
            row.pop("confidence")
            row["camera_id"] = "camera_main"
            row["original_tick"] = True
        path = self.scene_file(dict(schema="stage_e.camera_poses/1", verified=True, evidence="SYNTHETIC", pose_format="world_from_optical_camera",
                          coordinate_system="blender_rh_z_up", translation_unit="m", samples=rows), "boolean_ticks.json")
        with self.assertRaisesRegex(ValueError, "Invalid reviewed Camera Pose"):
            load_camera_poses(path)

    def test_architectural_category_contract(self):
        scene = deepcopy(self.scene)
        replacements = {"ground": "road", "barrier": "wall", "rock": "building"}
        for obj in scene["objects"]:
            if obj["category"] in replacements:
                obj["category"] = replacements[obj["category"]]
                obj["primitive_type"] = PRIMITIVES[obj["category"]]
        plan = prepare_plan(self.scene_file(scene, "architecture.json"), video_spec())
        types = {o["category"]: o["primitive_type"] for o in plan["objects"]}
        self.assertEqual({k: types[k] for k in replacements.values()}, dict(road="plane", wall="box", building="box"))

    def test_source_snapshot_can_be_read_from_another_directory(self):
        snapshot = source_scene_snapshot(self.scene, self.scene_path)
        self.assertFalse(Path(self.scene["objects"][0]["geometry_source"]["point_cloud_path"]).is_absolute())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"scene_schema.json"
            json_write(path, snapshot)
            plan = prepare_plan(path, video_spec())
            self.assertEqual(plan["frame_end"], 241)
            for obj in snapshot["objects"]:
                self.assertTrue(Path(obj["geometry_source"]["point_cloud_path"]).is_file())
                self.assertTrue(Path(obj["observations"][0]["mask"]["path"]).is_file())

    def test_incomplete_or_unsafe_receipt_never_reuses(self):
        self.assertFalse(cached_outputs_match(self.folder, {"artifact_hashes": {}}, generate_only=False))
        self.assertFalse(cached_outputs_match(self.folder, {"artifact_hashes": {"../unrelated": "0"*64}}, generate_only=True))

    def test_invalid_batch_and_malformed_jobs_return_failure(self):
        path = self.scene_file(dict(schema="wrong", jobs=[]), "batch.json")
        with self.subTest(case="invalid batch schema"), redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--batch", str(path)]), 1)
        path = self.scene_file(dict(schema="stage_ef.batch/1", jobs=[
            [], dict(scene_json=True, original_video="in.mp4", output="bad"),
            dict(scene_json="in.json", original_video="in.mp4", output="bad", generate_only="false"),
            dict(scene_json="in.json", original_video="in.mp4", output="good", generate_only=True)]), "malformed_batch.json")
        output = io.StringIO()
        with patch("blender.pipeline.run_pipeline", return_value={}) as run, redirect_stdout(output):
            self.assertEqual(main(["--batch", str(path)]), 1)
        self.assertEqual(run.call_count, 1)
        self.assertEqual([r["status"] for r in json.loads(output.getvalue())], ["failed"]*3+["passed"])


class MediaTests(unittest.TestCase):
    def test_invalid_window_is_rejected_before_decode(self):
        for start, count in ((True, 1), (.5, 1), (0, False), (0, 2.5), (-1, 1)):
            with self.subTest(start=start, count=count), self.assertRaisesRegex(ValueError, "frame window"):
                probe_video("unused.mp4", start_frame=start, frame_count=count)

    def test_command_timeout_preserves_partial_diagnostics(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder)/"timeout.log"
            with patch("blender.media.subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 2, output=b"partial decoder log")):
                with self.assertRaisesRegex(RuntimeError, "timed out"):
                    command(["ffmpeg"], log, timeout=2)
            self.assertIn("partial decoder log", log.read_text())
            self.assertIn("2 seconds", log.read_text())

    def test_rational_fps_frame_count_pts_window(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"fractional.mp4"
            command([ffmpeg_path(), "-y", "-nostdin", "-f", "lavfi", "-i", "testsrc=size=96x56:rate=60000/1001",
                     "-frames:v", "8", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)])
            video = probe_video(path, start_frame=2, frame_count=4)
            self.assertEqual(Fraction(video["fps"]), Fraction(60000, 1001))
            self.assertEqual(video["frame_count"], 4)
            self.assertAlmostEqual(video["first_pts_seconds"], 2*1001/60000)
            with self.assertRaises(ValueError):
                probe_video(path, frame_count=9)

    def test_vfr_refused_without_frame_duplication(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"vfr.mp4"
            command([ffmpeg_path(), "-y", "-nostdin", "-f", "lavfi", "-i", "testsrc=size=96x56:rate=60",
                     "-frames:v", "5", "-vf", "setpts=N*N/(60*TB)", "-fps_mode", "vfr",
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)])
            with self.assertRaisesRegex(ValueError, "VFR"):
                probe_video(path)


class BackgroundIntegrationTests(unittest.TestCase):
    def test_real_blender_repeat_batch_and_protection(self):
        try:
            executable = blender_path()
        except ValueError:
            self.skipTest("Blender unavailable; configure STAGE_EF_BLENDER to run background integration")
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            # Keep all five primitive categories and nonzero source IDs, but
            # compress ONLY this synthetic fixture to eight test frames.
            frames, _ = make_scene_fixture()
            scene, _ = build_scene(rgb_records(frames), SyntheticColorProvider(), base,
                scene_id="synthetic-8-frame-background", data_source="SYNTHETIC ONLY", fps=60,
                metric_frames=frames, min_depth_m=.1, max_depth_m=35)
            index_map = dict(zip([2000, 2030, 2060, 2120, 2180, 2240], [4000, 4001, 4002, 4003, 4005, 4007]))
            def remap(value):
                if isinstance(value, dict):
                    if "frame_index" in value:
                        value["frame_index"] = index_map[value["frame_index"]]
                        value["timestamp_seconds"] = (value["frame_index"]-4000)/60
                    for v in value.values():
                        remap(v)
                elif isinstance(value, list):
                    for v in value:
                        remap(v)
            remap(scene)
            scene["timeline"]["end_seconds"] = 7/60
            for o in scene["objects"]:
                o["source_frames"] = [index_map[i] for i in o["source_frames"]]
            # Test nonidentity object orientation/scale and organizational parent.
            scene["objects"][1]["transform"]["rotation"] = [0, 0, 2**-.5, 2**-.5]
            scene["objects"][1]["transform"]["scale"] = [1.2, .8, 1.1]
            scene["objects"][1]["parent_id"] = scene["objects"][0]["object_id"]
            # Off-centre K and unequal fx/fy exercise real Blender projection.
            for row in scene["cameras"][0]["track"]:
                row["intrinsics"] = [[135, 0, 115], [0, 150, 73], [0, 0, 1]]
            scene_path = base/"scene_schema.json"
            json_write(scene_path, scene)
            video_path = base/"original.mp4"
            command([ffmpeg_path(), "-y", "-nostdin", "-f", "lavfi", "-i", "testsrc=size=240x140:rate=60",
                     "-frames:v", "8", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video_path)])
            output = base/"output"
            first = run_pipeline(scene_path, video_path, output, blender=executable, resolution=(96, 56))
            self.assertEqual(first["scene_verification"]["objects"], 5)
            self.assertTrue(first["scene_verification"]["reopened_saved_blend"])
            for name in ["scene.blend", "blockout_video.mp4", "original_vs_blockout.mp4"]:
                self.assertTrue((output/name).is_file())
            snapshot = json.loads((output/"scene_schema.json").read_text(encoding="utf-8"))
            self.assertEqual(prepare_plan(output/"scene_schema.json", video_spec(count=8))["frame_end"], 8)
            self.assertEqual(snapshot["scene_id"], scene["scene_id"])
            for report in first["media"].values():
                self.assertEqual(report["frame_count"], 8)
                self.assertEqual(report["fps"], "60")
            (output/"user_notes.txt").write_text("preserve me", encoding="utf-8")
            second = run_pipeline(scene_path, video_path, output, blender=executable, resolution=(96, 56))
            self.assertEqual(second["scene_verification"]["objects"], 5)
            self.assertEqual((output/"user_notes.txt").read_text(), "preserve me")
            cached = run_pipeline(scene_path, video_path, output, blender=executable, resolution=(96, 56), reuse=True)
            self.assertEqual(cached["created_at"], second["created_at"])
            # Changed output bytes must invalidate --reuse, even with same input.
            (output/"blockout_video.mp4").write_bytes(b"tampered")
            restored = run_pipeline(scene_path, video_path, output, blender=executable, resolution=(96, 56), reuse=True)
            self.assertEqual(restored["media"]["blockout_video.mp4"]["frame_count"], 8)
            # Empty hashes previously satisfied all([]) and skipped generation.
            broken_receipt = deepcopy(restored)
            broken_receipt["artifact_hashes"] = {}
            json_write(output/"pipeline_report.json", broken_receipt)
            repaired = run_pipeline(scene_path, video_path, output, blender=executable, resolution=(96, 56), reuse=True)
            self.assertIn("scene_schema.json", repaired["artifact_hashes"])
            # Blender also saves/reopens/renders the architectural materials.
            architecture = deepcopy(scene)
            for obj in architecture["objects"]:
                category = {"ground": "road", "barrier": "wall", "rock": "building"}.get(obj["category"])
                if category:
                    obj.update(category=category, primitive_type=PRIMITIVES[category])
            architecture_path = base/"architecture.json"
            json_write(architecture_path, architecture)
            architectural_report = run_pipeline(architecture_path, video_path, base/"architecture_output", blender=executable, resolution=(96, 56))
            self.assertEqual(architectural_report["stage_f"], "passed")
            # End-to-end fractional FPS, not just a media probe: Blender's
            # fps_base, animation timestamps and H.264 outputs must agree.
            fractional = deepcopy(scene)
            fractional["timeline"]["fps"] = 60000/1001
            fractional["timeline"]["end_seconds"] = 7*1001/60000
            def fractional_times(value):
                if isinstance(value, dict):
                    if "frame_index" in value:
                        value["timestamp_seconds"] = (value["frame_index"]-4000)*1001/60000
                    for v in value.values():
                        fractional_times(v)
                elif isinstance(value, list):
                    for v in value:
                        fractional_times(v)
            fractional_times(fractional)
            fractional_scene = base/"fractional_scene.json"
            fractional_video = base/"fractional_video.mp4"
            json_write(fractional_scene, fractional)
            command([ffmpeg_path(), "-y", "-nostdin", "-f", "lavfi", "-i", "testsrc=size=240x140:rate=60000/1001",
                     "-frames:v", "8", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(fractional_video)])
            fractional_report = run_pipeline(fractional_scene, fractional_video, base/"fractional_output", blender=executable, resolution=(96, 56))
            for report in fractional_report["media"].values():
                self.assertEqual(report["frame_count"], 8)
                self.assertEqual(report["fps"], "60000/1001")
            unmanaged = base/"unmanaged"
            unmanaged.mkdir()
            (unmanaged/"scene.blend").write_bytes(b"user data")
            with self.assertRaisesRegex(ValueError, "unmanaged"):
                run_pipeline(scene_path, video_path, unmanaged, blender=executable)
            self.assertEqual((unmanaged/"scene.blend").read_bytes(), b"user data")
            manifest = base/"batch.json"
            json_write(manifest, dict(schema="stage_ef.batch/1", jobs=[
                dict(scene_json="scene_schema.json", original_video="original.mp4", output="batch_good", resolution=[96, 56], generate_only=True),
                dict(scene_json="missing.json", original_video="original.mp4", output="batch_bad")]))
            self.assertEqual(main(["--batch", str(manifest), "--blender", executable]), 1)
            self.assertTrue((base/"batch_good/scene.blend").is_file())


if __name__ == "__main__":
    unittest.main()
