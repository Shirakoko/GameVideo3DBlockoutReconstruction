"""Stage G analytical correctness, evidence boundaries and pipeline recovery."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from evaluation.evaluator import evaluate, evaluate_existing_sample
from evaluation.geometry_metrics import geometry_metrics
from evaluation.report import add_comparison, add_metric, new_report, write_json, write_reports
from evaluation.resources import ResourceMonitor
from evaluation.semantic_metrics import semantic_metrics
from evaluation.temporal_metrics import camera_metrics, synchronization_metrics
from evaluation.visual_metrics import ssim
from pipeline.config import resolve_settings
from pipeline.runner import run_steps


def pose(index, position=(0, 0, 0), degrees=0, camera="camera_main", time=None):
    matrix = np.eye(4)
    matrix[:3, 3] = position
    matrix[:3, :3] = Rotation.from_euler("z", degrees, degrees=True).as_matrix()
    return dict(frame_index=index, timestamp_seconds=float(index if time is None else time), camera_id=camera, world_from_camera=matrix.tolist())


def obj(frame=0, identity="one", category="tree", static=True, box=(0, 0, 10, 10)):
    return dict(frame_index=frame, object_id=identity, category=category, is_static=static, bbox_xyxy=list(box))


class GeometryTests(unittest.TestCase):
    def test_analytical_distance_definitions_and_tolerances(self):
        result = geometry_metrics([[0, 0, 0]], [[1, 0, 0]], thresholds_m=[.5, 1.])
        values = result["values"]
        self.assertEqual(values["chamfer_distance_m"], 1.)
        self.assertEqual(values["chamfer_distance_squared_m2"], 1.)
        self.assertEqual(values["f_score_at_0.5m"], 0.)
        self.assertEqual(values["f_score_at_1m"], 1.)

    def test_partial_coverage_not_visual_accuracy(self):
        result = geometry_metrics([[0, 0, 0]], [[0, 0, 0], [2, 0, 0]], thresholds_m=[.1])["values"]
        self.assertEqual(result["chamfer_distance_m"], .5)
        self.assertEqual(result["point_cloud_coverage_at_0.1m"], .5)
        self.assertAlmostEqual(result["f_score_at_0.1m"], 2/3)

    def test_empty_prediction_is_not_zero_error(self):
        result = geometry_metrics([], [[0, 0, 0]])["values"]
        self.assertIsNone(result["chamfer_distance_m"])
        self.assertIsNone(result["point_precision_at_0.05m"])
        self.assertEqual(result["point_cloud_coverage_at_0.05m"], 0.)
        self.assertEqual(result["f_score_at_0.05m"], 0.)

    def test_invalid_cloud_and_configuration(self):
        for name, predicted, gt, options in [
            ("empty_gt", [[0, 0, 0]], [], {}), ("nan", [[np.nan, 0, 0]], [[0, 0, 0]], {}),
            ("shape", [[0, 0]], [[0, 0, 0]], {}), ("threshold", [[0, 0, 0]], [[0, 0, 0]], {"thresholds_m": [0]}),
            ("duplicate_threshold", [[0, 0, 0]], [[0, 0, 0]], {"thresholds_m": [.1, .1]}),
            ("budget", [[0, 0, 0]], [[0, 0, 0]], {"max_points": True})]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                geometry_metrics(predicted, gt, **options)

    def test_sampling_reproducible_and_reported(self):
        points = np.arange(300).reshape(-1, 3)
        first = geometry_metrics(points, points, max_points=20, seed=17)
        self.assertEqual(first, geometry_metrics(points, points, max_points=20, seed=17))
        self.assertEqual(first["details"]["evaluated_ground_truth_points"], 20)
        self.assertEqual(first["details"]["ground_truth_points"], 100)


class TemporalTests(unittest.TestCase):
    def test_pose_translation_and_so3_rotation(self):
        result = camera_metrics([pose(0, (1, 0, 0), 90)], [pose(0)])["values"]
        self.assertAlmostEqual(result["camera_translation_error_m_rmse"], 1.)
        self.assertAlmostEqual(result["camera_rotation_error_deg_rmse"], 90.)
        self.assertIsNone(result["trajectory_rpe_translation_m_rmse"])

    def test_ate_constant_offset_and_relative_pose(self):
        truth = [pose(i, (i, 0, 0)) for i in range(3)]
        pred = [pose(i, (i+1, 0, 0)) for i in range(3)]
        result = camera_metrics(pred, truth)["values"]
        self.assertEqual(result["trajectory_ate_m_rmse"], 1.)
        self.assertEqual(result["trajectory_rpe_translation_m_rmse"], 0.)

    def test_rpe_measures_drift(self):
        truth = [pose(i, (i, 0, 0)) for i in range(4)]
        pred = [pose(i, (i*1.1, 0, 0)) for i in range(4)]
        result = camera_metrics(pred, truth)["values"]
        self.assertAlmostEqual(result["trajectory_rpe_translation_m_rmse"], .1)

    def test_missing_frames_never_shorten_rpe_delta(self):
        truth = [pose(i) for i in [100, 101, 102]]
        pred = [pose(i) for i in [100, 102]]
        result = camera_metrics(pred, truth)["values"]
        self.assertEqual(result["camera_pose_coverage"], 2/3)
        self.assertEqual(result["camera_pose_missing_frames"], 1)
        self.assertEqual(result["rpe_pair_count"], 0)
        self.assertEqual(result["rpe_expected_pair_count"], 2)

    def test_multicamera_join_not_crossed(self):
        result = camera_metrics([pose(100, camera="left")], [pose(100, camera="left"), pose(100, camera="right")])["values"]
        self.assertEqual(result["camera_pose_coverage"], .5)

    def test_empty_prediction_has_missing_metrics(self):
        result = camera_metrics([], [pose(100)])["values"]
        self.assertIsNone(result["trajectory_ate_m_rmse"])
        self.assertEqual(result["camera_pose_coverage"], 0.)

    def test_se3_alignment_explicit_no_scale(self):
        positions = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
        gt = [pose(i, xyz) for i, xyz in enumerate(positions)]
        pred = [pose(i, (2-xyz[1], 4+xyz[0], 0)) for i, xyz in enumerate(positions)]
        result = camera_metrics(pred, gt, alignment="se3")["values"]
        self.assertLess(result["trajectory_ate_m_rmse"], 1e-10)
        self.assertGreater(result["camera_translation_error_m_rmse"], 1.)
        with self.assertRaises(ValueError):
            camera_metrics([pose(0), pose(1)], [pose(0), pose(1)], alignment="se3")

    def test_sync_offset_drift_missing_and_nonzero_time_origin(self):
        rgb = [dict(frame_index=100+i, timestamp_seconds=10.+i) for i in range(4)]
        depth = [dict(frame_index=100+i, timestamp_seconds=10.+i+.01+i*.0005) for i in range(3)]
        values = synchronization_metrics(dict(rgb=rgb, depth=depth))["depth"]["values"]
        self.assertAlmostEqual(values["sync_error_s_max"], .011)
        self.assertAlmostEqual(values["sync_clock_drift_ppm"], 500.)
        self.assertEqual(values["sync_missing_frames"], 1)

    def test_invalid_pose_sync_and_source_indices(self):
        reflected = pose(0)
        reflected["world_from_camera"][0][0] = -1
        for name, pred, options in [("reflected", [reflected], {}), ("duplicate", [pose(0), pose(0)], {}),
            ("order", [pose(1), pose(0)], {}), ("delta", [pose(0)], {"rpe_delta_frames": 0}),
            ("sim3", [pose(0)], {"alignment": "sim3"})]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                camera_metrics(pred, [pose(0)], **options)
        with self.assertRaises(ValueError):
            synchronization_metrics(dict(rgb=[]))


class SemanticTests(unittest.TestCase):
    def test_assignment_maximizes_valid_match_count_before_iou(self):
        with patch("evaluation.semantic_metrics.box_iou", side_effect=[.99, .6, .6, .49]*2):
            result = semantic_metrics([obj(identity="p1"), obj(identity="p2")], [obj(identity="g1"), obj(identity="g2")], evaluated_frames=[0])["values"]
        self.assertEqual(result["detection_true_positives"], 2)

    def test_wrong_category_counts_as_error(self):
        result = semantic_metrics([obj(category="rock")], [obj()], evaluated_frames=[0])["values"]
        self.assertEqual(result["detection_precision"], 0.)
        self.assertEqual(result["detection_recall"], 0.)
        self.assertEqual(result["category_accuracy"], 0.)

    def test_duplicate_localization_is_one_false_positive(self):
        result = semantic_metrics([obj(), obj(identity="two")], [obj()], evaluated_frames=[0])["values"]
        self.assertEqual(result["detection_true_positives"], 1)
        self.assertEqual(result["detection_false_positives"], 1)
        self.assertEqual(result["detection_precision"], .5)

    def test_id_switch_across_missed_annotation(self):
        gt = [obj(frame=i) for i in range(3)]
        pred = [obj(frame=0), obj(frame=2, identity="changed")]
        result = semantic_metrics(pred, gt, evaluated_frames=[0, 1, 2])["values"]
        self.assertEqual(result["object_tracking_id_switches"], 1)
        self.assertEqual(result["tracking_fragmentations"], 1)
        result = semantic_metrics(pred, gt, evaluated_frames=[0, 1, 2], max_id_gap_frames=1)["values"]
        self.assertEqual(result["object_tracking_id_switches"], 0)

    def test_unknown_motion_prediction_not_excluded(self):
        result = semantic_metrics([obj(static=None)], [obj()], evaluated_frames=[0])["values"]
        self.assertEqual(result["static_dynamic_accuracy"], 0.)
        self.assertEqual(result["static_dynamic_match_count"], 1)

    def test_exhaustive_empty_frame_penalizes_false_positive(self):
        result = semantic_metrics([obj(frame=1)], [obj(frame=0)], evaluated_frames=[0, 1])["values"]
        self.assertEqual(result["detection_false_positives"], 1)
        self.assertEqual(result["detection_false_negatives"], 1)

    def test_no_gt_and_duplicate_identity(self):
        result = semantic_metrics([], [], evaluated_frames=[0])["values"]
        self.assertIsNone(result["category_accuracy"])
        self.assertIsNone(result["object_tracking_id_switches"])
        with self.assertRaises(ValueError):
            semantic_metrics([obj(), obj()], [obj()], evaluated_frames=[0])


class EvaluationTests(unittest.TestCase):
    def test_metric_units_do_not_confuse_mean_and_metres(self):
        report = new_report("test")
        add_comparison(report, "camera", dict(values=dict(camera_timestamp_error_s_mean=.1, camera_translation_error_m_max=1., sync_error_s_median=.01)))
        self.assertEqual([r["unit"] for r in report["metrics"]], ["s", "m", "s"])

    def test_ssim_identical_different_and_invalid(self):
        rgb = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)
        self.assertAlmostEqual(ssim(rgb, rgb), 1.)
        self.assertLess(ssim(rgb, 255-rgb), .5)
        with self.assertRaises(ValueError):
            ssim(rgb, rgb[:8])

    def test_html_escapes_and_csv_agrees_with_strict_json(self):
        with tempfile.TemporaryDirectory() as folder:
            report = new_report("<script>alert('bad')</script>")
            add_metric(report, "geometry", "distance", .25, unit="m")
            add_metric(report, "camera", "missing", None, reason="<broken>")
            write_reports(folder, report)
            html = (Path(folder)/"evaluation_report.html").read_text(encoding="utf-8")
            self.assertNotIn("<script>", html)
            self.assertIn("&lt;script&gt;", html)
            with (Path(folder)/"metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(float(csv_rows[0]["value"]), .25)
            self.assertEqual(csv_rows[1]["value"], "")
            self.assertIsNone(json.loads((Path(folder)/"evaluation_report.json").read_text(encoding="utf-8"))["metrics"][1]["value"])
            with self.assertRaises(ValueError):
                write_json(Path(folder)/"nan.json", dict(value=float("nan")))

    def test_inferred_depth_never_gt(self):
        with tempfile.TemporaryDirectory() as folder:
            result = evaluate(dict(schema="stage_g.config/1", data_source="test", ground_truth=dict(provenance="inferred", verified=True, evidence="guess", metric_manifest="unknown.json")), folder)
            self.assertEqual(result["status"], "partial")
            self.assertTrue(result["failures"])
            self.assertTrue(all(m["value"] is None for m in result["metrics"] if m["group"] == "geometry"))

    def test_dataset_audit_is_not_fake_geometry_accuracy(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder)/"sample", Path(folder)/"report"
            write_json(source/"validation_report.json", dict(timestamp_max_rgb_depth_meta_error_seconds=0., camera_table_pose_mismatch_frames=[]))
            result = evaluate_existing_sample(source, output)
            self.assertEqual(result["status"], "partial")
            self.assertFalse(result["evidence"]["download_performed"])
            self.assertTrue(all(m["value"] is None for m in result["metrics"] if m["group"] in ("geometry", "camera")))

    def test_known_depth_truth_and_shifted_camera_prediction(self):
        from agents.synthetic_scene import make_scene_fixture
        from scripts.reconstruct_scene import save_synthetic_inputs
        from reconstruction.backprojection import backproject
        from reconstruction.coordinate_transform import transform_points
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            frames, _ = make_scene_fixture()
            frames = [frames[0]]
            save_synthetic_inputs(folder, frames, dict(data_source="SYNTHETIC test"))
            f = frames[0]
            bp = backproject(f.depth, f.intrinsics, spec=f.depth_spec, valid_mask=f.valid_mask, rgb=f.rgb, min_depth_m=.1, max_depth_m=35, stride=4)
            points = transform_points(bp.points_camera_m, f.world_from_camera)
            np.savez_compressed(folder/"prediction.npz", points_world_m=points)
            candidate = pose(f.frame_index, time=f.timestamp_seconds)
            candidate["world_from_camera"] = f.world_from_camera.copy().tolist()
            candidate["world_from_camera"][0][3] += .5
            write_json(folder/"camera.json", dict(schema="stage_g.camera_prediction/1", coordinate_system="blender_rh_z_up", translation_unit="m", pose_direction="world_from_optical_camera", samples=[candidate]))
            config = dict(schema="stage_g.config/1", data_source="SYNTHETIC ONLY", ground_truth=dict(provenance="ground_truth", verified=True, evidence="fixture oracle", metric_manifest="synthetic_input_manifest.json"),
                prediction=dict(provenance="predicted", coordinate_system="blender_rh_z_up", unit="m", point_cloud="prediction.npz", camera_poses="camera.json"), geometry=dict(min_depth_m=.1, max_depth_m=35, stride=4))
            result = evaluate(config, folder/"report", base=folder)
            values = {m["metric"]: m["value"] for m in result["metrics"]}
            self.assertEqual(values["chamfer_distance_m"], 0.)
            self.assertAlmostEqual(values["camera_translation_error_m_rmse"], .5)
            self.assertIsNone(values["detection_precision"])
            config["ground_truth"]["verified"] = False
            result = evaluate(config, folder/"bad", base=folder)
            self.assertTrue(result["failures"])

    def test_resource_probe_has_explicit_gpu_absence(self):
        monitor = ResourceMonitor(interval=.01)
        monitor.nvidia = None
        monitor.start()
        result = monitor.finish()
        self.assertGreater(result["sample_count"], 0)
        self.assertGreater(result["process_tree_rss_peak_bytes"], 0)
        self.assertEqual(result["gpu"]["status"], "unavailable")
        self.assertIsNone(result["gpu"]["memory_peak_mb"])


class PipelineTests(unittest.TestCase):
    def test_raw_preparation_uses_isolated_attempt_and_no_stale_fallback(self):
        from pipeline.stages import execute_stage
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            config = dict(mode="existing_sample", prepare_raw_sample=True, sample_source=str(folder/"source"))
            def prepare(*, source, output):
                write_json(Path(output)/"validation_report.json", dict(timestamp_max_rgb_depth_meta_error_seconds=0., camera_table_pose_mismatch_frames=[]))
            with patch("scripts.prepare_sample.main", side_effect=prepare) as prepared, patch("reconstruction.validation.audit_sample", return_value=dict(candidate_poses=[], blockers=[dict(id="metric_depth", reason="unverified")])):
                self.assertEqual(execute_stage("A", config, {}, folder/"A"), 2)
                self.assertEqual(prepared.call_args.kwargs["output"], (folder/"A/prepared_sample").resolve())
            stages = dict(A=dict(status="blocked", last_attempt_output_directory=str(folder/"A")))
            self.assertEqual(execute_stage("G", config, stages, folder/"G"), 0)
            stages["A"]["last_attempt_output_directory"] = str(folder/"missing")
            self.assertEqual(execute_stage("G", config, stages, folder/"G_missing"), 0)
            report = json.loads((folder/"G_missing/evaluation_report.json").read_text(encoding="utf-8"))
            self.assertIn("no stale sample", report["data_source"])

    def test_one_click_blender_export_cache_and_reviewed_mode(self):
        from scripts.run_pipeline import main
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            configuration = dict(schema="pipeline.config/1", mode="synthetic", output="synthetic", retries=0, timeout_seconds=120,
                synthetic=dict(fps=1, frame_count=4, keyframe_stride=1), reconstruction=dict(min_depth_m=.1, max_depth_m=35, voxel_size_m=.08, stride=4),
                render=dict(enabled=True, resolution=[240, 140]))
            write_json(folder/"config.json", configuration)
            self.assertEqual(main(["--config", str(folder/"config.json")]), 0)
            report = json.loads((folder/"synthetic/evaluation_report.json").read_text(encoding="utf-8"))
            stages = {s["id"]: Path(s["output_directory"]) for s in report["stages"]}
            exported = json.loads((stages["F"]/"camera_pose_evaluation.json").read_text(encoding="utf-8"))
            self.assertEqual(len(exported["samples"]), 4)
            self.assertEqual(exported["samples"][0]["frame_index"], 7000)
            self.assertIn("reopened scene.blend", exported["source"])
            self.assertEqual(report["comparisons"]["camera"]["values"]["camera_pose_match_count"], 4)
            self.assertEqual(main(["--config", str(folder/"config.json")]), 0)
            cached = json.loads((folder/"synthetic/pipeline_report.json").read_text(encoding="utf-8"))
            self.assertTrue(all(s["status"] == "cached" for s in cached["stages"]))
            scene = json.loads((stages["C"]/"scene_schema.json").read_text(encoding="utf-8"))
            rows = []
            for frame in scene["timeline"]["frames"]:
                items = []
                for item in scene["objects"]:
                    observation = next(row for row in item["observations"] if row["frame_index"] == frame["frame_index"])
                    items.append(dict(key=item["object_id"], category=item["category"], score=1.,
                        mask=dict(kind="file", path=str(stages["C"]/observation["mask"]["path"]), score=1.)))
                rows.append(dict(frame_index=frame["frame_index"], raster_size=[240, 140], objects=items))
            write_json(folder/"annotations.json", dict(schema="stage_c.rgb_annotations/1", evidence="Synthetic replay test, not real ground truth", frames=rows))
            configuration.update(mode="reviewed", output="reviewed", input_manifest=str(stages["A"]/"synthetic_input_manifest.json"),
                original_video=str(stages["A"]/"original_video.mp4"), annotations="annotations.json", render=dict(enabled=False),
                evaluation=dict(ground_truth=dict(provenance="ground_truth", verified=True, evidence="synthetic fixture", metric_manifest=str(stages["A"]/"synthetic_input_manifest.json"))))
            configuration.pop("synthetic")
            write_json(folder/"reviewed.json", configuration)
            self.assertEqual(main(["--config", str(folder/"reviewed.json")]), 0)
            reviewed = json.loads((folder/"reviewed/pipeline_report.json").read_text(encoding="utf-8"))
            self.assertEqual([s["id"] for s in reviewed["stages"]], list("ABCDG"))

    def step(self, identity="A", depends=None, code=None, inputs=None):
        code = code or "from pathlib import Path; import sys; Path(sys.argv[1],'done.json').write_text('ok')"
        return dict(id=identity, depends_on=depends or [], command=["{python}", "-c", code, "{output}"], inputs=inputs or [], outputs=["done.json"], timeout_seconds=10, retries=0)

    def test_cache_hashes_outputs_and_preserves_attempts(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"run"
            first = run_steps([self.step()], output)
            second = run_steps([self.step()], output)
            self.assertEqual(second["stages"][0]["status"], "cached")
            artifact = Path(first["stages"][0]["output_directory"])/"done.json"
            artifact.write_text("corrupted")
            third = run_steps([self.step()], output)
            self.assertEqual(third["stages"][0]["status"], "completed")
            self.assertEqual(len(third["stages"][0]["attempts"]), 2)
            self.assertTrue(artifact.exists())

    def test_changed_inputs_invalidate_descendants(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder)/"input.txt", Path(folder)/"run"
            source.write_text("first")
            code = f"from pathlib import Path; import sys; Path(sys.argv[1],'done.json').write_text(Path({str(source)!r}).read_text())"
            steps = [self.step(code=code, inputs=[str(source)]), self.step(identity="B", depends=["A"])]
            run_steps(steps, output)
            source.write_text("changed")
            result = run_steps(steps, output)
            self.assertEqual([s["status"] for s in result["stages"]], ["completed", "completed"])
            self.assertTrue(all(len(s["attempts"]) == 2 for s in result["stages"]))

    def test_failed_attempt_retry_and_upstream_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            code = ("from pathlib import Path; import sys; p=Path(sys.argv[1]); marker=p.parent.parent/'once'; "
                "existed=marker.exists(); marker.write_text('1'); "
                "p.joinpath('done.json').write_text('ok') if existed else None; sys.exit(0 if existed else 1)")
            step = self.step(identity="B", depends=["A"], code=code)
            step["retries"] = 1
            result = run_steps([self.step(), step], Path(folder)/"run")
            self.assertEqual(result["status"], "completed")
            attempts = result["stages"][1]["attempts"]
            self.assertEqual([a["status"] for a in attempts], ["failed", "completed"])
            self.assertNotIn("error", result["stages"][1])
            self.assertTrue(all(Path(a["log"]).exists() for a in attempts))

    def test_blocked_stage_not_retried_and_reporting_still_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            blocked = self.step(code="import sys; sys.exit(2)")
            blocked["retries"] = 2
            report = self.step(identity="G", depends=["A", "B"])
            report["always_run"] = True
            result = run_steps([blocked, self.step(identity="B", depends=["A"]), report], Path(folder)/"run")
            self.assertEqual(result["status"], "blocked")
            self.assertEqual([s["status"] for s in result["stages"]], ["blocked", "skipped", "completed"])
            self.assertEqual(len(result["stages"][0]["attempts"]), 1)

    def test_timeout_terminates_and_logs_failed_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            step = self.step(code="import time; time.sleep(20)")
            step["timeout_seconds"] = 1
            result = run_steps([step], Path(folder)/"run")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["stages"][0]["attempts"][0]["exit_code"], 124)
            self.assertTrue(result["stages"][0]["attempts"][0]["resources"])

    def test_no_resume_keeps_prior_outputs_and_reexecutes(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"run"
            first = run_steps([self.step()], output)
            second = run_steps([self.step()], output, resume=False)
            self.assertEqual(second["stages"][0]["status"], "completed")
            self.assertNotEqual(first["stages"][0]["output_directory"], second["stages"][0]["output_directory"])

    def test_successful_exit_without_outputs_is_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run_steps([self.step(code="pass")], Path(folder)/"run")
            self.assertEqual(result["status"], "failed")
            self.assertIn("required artifacts", result["stages"][0]["error"])

    def test_output_protection_and_dag_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder/"user.txt").write_text("keep")
            with self.assertRaises(ValueError):
                run_steps([self.step()], folder)
            for name, steps in [("cycle", [self.step(depends=["B"])]), ("traversal", [self.step(identity="../A")]), ("duplicate", [self.step(), self.step()])]:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    run_steps(steps, folder/"invalid")
            self.assertEqual((folder/"user.txt").read_text(), "keep")

    def test_stale_lock_and_interrupted_checkpoint_recovered(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"run"
            result = run_steps([self.step()], output)
            state = json.loads((output/"pipeline_state.json").read_text(encoding="utf-8"))
            state["stages"]["A"]["status"] = "running"
            state["stages"]["A"]["attempts"][0]["status"] = "running"
            write_json(output/"pipeline_state.json", state)
            write_json(output/".pipeline.lock", dict(pid=2000000000, created=0))
            resumed = run_steps([self.step()], output)
            self.assertEqual(resumed["stages"][0]["status"], "completed")
            self.assertEqual(resumed["stages"][0]["attempts"][0]["status"], "interrupted")

    def test_high_level_config_is_explicit_and_strict(self):
        config = dict(schema="pipeline.config/1", mode="reviewed")
        with self.assertRaises(ValueError):
            resolve_settings(config, Path("."))
        config = dict(schema="pipeline.config/1", mode="synthetic", reconstruction=dict(min_depth_m=.1, max_depth_m=35), typo=True)
        with self.assertRaises(ValueError):
            resolve_settings(config, Path("."))


if __name__ == "__main__":
    unittest.main()
