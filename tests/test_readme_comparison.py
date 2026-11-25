import importlib.util
from fractions import Fraction
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


spec = importlib.util.spec_from_file_location(
    "readme_comparison", Path(__file__).resolve().parents[1] / "docs/generate_comparison.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ReadmeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.report = {
            "status": "complete_source_indexed_approximation",
            "temporal_pass": True, "source_mapping_pass": True,
            "source": "source.mp4", "source_sha256": "source-hash",
            "output_sha256": "video-hash",
            "probe": {"width": 2, "height": 1, "frames": 720, "fps": "60/1",
                      "duration_seconds": "12",
                      "pts_seconds": [str(Fraction(i, 60)) for i in range(720)]},
            "mapping": [{"source_index": i, "padding": False,
                         "source_pts_seconds": str(Fraction(i, 60))} for i in range(720)],
        }
        self.source_hashes = {"source.mp4": "source-hash"}
        self.pixels = b"".join(bytes(color) * 2 for color in ((255, 0, 0), (0, 255, 0), (0, 0, 255)))
        self.decoder = self.enterContext(patch.object(
            comparison.subprocess, "run", return_value=SimpleNamespace(stdout=self.pixels)))
        self.enterContext(patch.object(comparison.imageio_ffmpeg, "get_ffmpeg_exe", return_value="ffmpeg"))
        self.hash_mock = self.enterContext(patch.object(comparison, "digest", return_value="video-hash"))
        self.enterContext(patch.object(comparison, "read_json", side_effect=self.read_receipt))

    def read_receipt(self, path):
        return self.report if path.name == "report.json" else {"source_hashes": self.source_hashes}

    def test_extracts_final_source_indices_without_timestamp_seeking(self):
        frames = comparison.load_h3_frames((0, 360, 719))
        self.assertEqual(list(frames), [0, 360, 719])
        self.assertEqual([frame.getpixel((0, 0)) for frame in frames.values()],
                         [(255, 0, 0), (0, 255, 0), (0, 0, 255)])
        command = self.decoder.call_args.args[0]
        self.assertEqual(command[command.index("-vf") + 1], "select='eq(n,0)+eq(n,360)+eq(n,719)'")
        self.assertNotIn("-ss", command)

    def test_rejects_part_local_index_or_padding(self):
        self.report["mapping"][360]["source_index"] = 122
        with self.assertRaisesRegex(ValueError, "source mapping"):
            comparison.load_h3_frames((0, 360, 719))
        self.report["mapping"][360]["source_index"] = 360
        self.report["mapping"][719]["padding"] = True
        with self.assertRaisesRegex(ValueError, "source mapping"):
            comparison.load_h3_frames((0, 360, 719))
        self.decoder.assert_not_called()

    def test_rejects_unverified_delivery(self):
        self.report["temporal_pass"] = False
        with self.assertRaisesRegex(ValueError, "verified complete"):
            comparison.load_h3_frames((0, 360, 719))
        self.decoder.assert_not_called()

    def test_rejects_wrong_frame_count_fps_duration_or_pts(self):
        probe = self.report["probe"]
        for field, wrong_value in (("frames", 719), ("fps", "24"),
                                   ("duration_seconds", "30"), ("pts_seconds", ["0"])):
            original = probe[field]
            probe[field] = wrong_value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "media time lattice"):
                comparison.load_h3_frames((0, 360, 719))
            probe[field] = original
        probe["pts_seconds"][360] = "6.001"
        with self.assertRaisesRegex(ValueError, "media time lattice"):
            comparison.load_h3_frames((0, 360, 719))
        self.decoder.assert_not_called()

    def test_rejects_different_source_or_modified_video(self):
        self.source_hashes["source.mp4"] = "different-source"
        with self.assertRaisesRegex(ValueError, "RGB source"):
            comparison.load_h3_frames((0, 360, 719))
        self.source_hashes["source.mp4"] = "source-hash"
        self.hash_mock.return_value = "modified-video"
        with self.assertRaisesRegex(ValueError, "verified delivery"):
            comparison.load_h3_frames((0, 360, 719))
        self.decoder.assert_not_called()

    def test_rejects_incomplete_decode(self):
        self.decoder.return_value.stdout = self.pixels[:-6]
        with self.assertRaisesRegex(ValueError, "all requested native frames"):
            comparison.load_h3_frames((0, 360, 719))

    def test_rejects_invalid_requested_indices(self):
        for indices in ((), (360, 0), (0, 0), (-1,), (720,)):
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, "unique, increasing"):
                comparison.load_h3_frames(indices)
        self.decoder.assert_not_called()


class ThreeRouteMetricTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.routes = {name: self.root / name for name in ("voxel", "primitive")}
        self.enterContext(patch.object(comparison, "ROOT", self.root))
        self.enterContext(patch.object(comparison, "ROUTES", self.routes))
        self.enterContext(patch.object(comparison, "digest", return_value="fixture-hash"))
        self.enterContext(patch.object(comparison, "last_render_seconds", return_value=1))
        self.indices = list(comparison.METRIC_FRAME_INDICES)
        self.rgb = np.full((24, 24, 3), 90, dtype=np.uint8)
        self.codes = np.ones((24, 24), dtype=np.uint16)
        self.labels = np.full((24, 24), comparison.CATEGORIES.index("environment"), dtype=np.uint8)
        self.labels[0, 0] = comparison.CATEGORIES.index("person")
        self.codes[0, 1] = 0
        self.codes[0, 2] = 1020
        self.mask = (self.codes > 0) & (self.codes < 1020) & (self.labels != comparison.CATEGORIES.index("person"))
        self.h3 = {i: Image.fromarray(np.full_like(self.rgb, 120)) for i in self.indices}
        self.loader = self.enterContext(patch.object(comparison, "load_h3_frames", return_value=self.h3))
        self.reports = {}
        for route, folder in self.routes.items():
            (folder / "frames").mkdir(parents=True)
            (folder / "scene.blend").touch()
            (folder / "blockout_video.mp4").touch()
            predicted = self.rgb if route == "voxel" else np.full_like(self.rgb, 100)
            previews = []
            for index in self.indices:
                Image.fromarray(predicted).save(folder / f"frames/frame_{index+1:06d}.png")
                previews.append(dict(frame_index=index,
                                     masked_ssim_informational=comparison.masked_ssim(self.rgb, predicted, self.mask),
                                     masked_luminance_mae=comparison.masked_luminance_mae(self.rgb, predicted, self.mask)))
            self.reports[folder / ("proxy_run.json" if route == "voxel" else "primitive_run.json")] = dict(
                status="passed", full_video_rendered=True, source_hashes={"source.mp4": "same-source"},
                preview_frames=previews)
            self.reports[folder / "scene_schema.json"] = dict(cameras=[], objects=[dict(
                observations=[], parent_id=None, category="environment")])
            self.reports[folder / "pipeline_report.json"] = dict(
                parameters={"engine": "BLENDER_EEVEE"}, media={"video": dict(
                    frame_count=720, fps="60", pts_verified=True)})
        for index in self.indices:
            np.savez(self.cache / f"frame_{index:06d}.npz", rgb=self.rgb, codes=self.codes, labels=self.labels)
        self.reports[self.root / "outputs/stage_ef/real_refined/refinement_report.json"] = {"frame_cache": str(self.cache)}
        self.enterContext(patch.object(comparison, "read_json", side_effect=lambda path: self.reports[path]))

    def test_collects_same_nine_frames_and_mask_for_all_three_routes(self):
        scorer = self.enterContext(patch.object(comparison, "score_frame", wraps=comparison.score_frame))
        rows, summary, hashes, cache = comparison.collect()
        self.loader.assert_called_once_with(self.indices)
        self.assertEqual(cache, self.cache)
        self.assertEqual(len(rows), 27)
        self.assertEqual(list(summary), ["voxel", "primitive", "h3"])
        for index in self.indices:
            current = [row for row in rows if row["frame_index"] == index]
            self.assertEqual([row["route"] for row in current], ["voxel", "primitive", "h3"])
            self.assertEqual({row["evaluated_pixels"] for row in current}, {573})
            self.assertEqual({row["timestamp_seconds"] for row in current}, {index / 60})
            calls = [call for call in scorer.call_args_list if call.args[1] == index]
            self.assertTrue(all(call.args[4] is calls[0].args[4] for call in calls))
            np.testing.assert_array_equal(calls[0].args[4], self.mask)
            self.assertIn(f"h3_rgb24_frame_{index:06d}", hashes)
        self.assertIn("h3_delivery_video", hashes)
        self.assertIn("h3_delivery_report", hashes)
        self.assertEqual(summary["voxel"]["mean_masked_ssim"], 1.0)
        self.assertEqual(summary["voxel"]["mean_rgb_mean_mae_255"], 0.0)
        self.assertEqual(summary["primitive"]["mean_rgb_mean_mae_255"], 10.0)
        self.assertEqual(summary["h3"]["mean_rgb_mean_mae_255"], 30.0)
        self.assertAlmostEqual(summary["h3"]["mean_masked_ssim"],
                               comparison.masked_ssim(self.rgb, np.asarray(self.h3[0]), self.mask))
        self.assertEqual(set(summary["h3"]), {"mean_masked_ssim", "mean_rgb_mean_mae_255"})

    def test_rejects_different_h3_raster_without_resizing(self):
        self.h3[0] = self.h3[0].resize((12, 12))
        with self.assertRaisesRegex(ValueError, "equal image shapes"):
            comparison.collect()

    def test_rejects_changed_geometry_metric(self):
        self.reports[self.routes["voxel"] / "proxy_run.json"]["preview_frames"][0]["masked_ssim_informational"] = 0
        with self.assertRaisesRegex(ValueError, "Existing preview metric"):
            comparison.collect()

    def test_rejects_unreviewed_frame_set(self):
        for route, folder in self.routes.items():
            report = self.reports[folder / ("proxy_run.json" if route == "voxel" else "primitive_run.json")]
            report["preview_frames"] = report["preview_frames"][:-1]
        with self.assertRaisesRegex(ValueError, "reviewed nine"):
            comparison.collect()
        self.loader.assert_not_called()

    def test_rejects_empty_ssim_mask(self):
        with self.assertRaisesRegex(ValueError, "Empty evaluation mask"):
            comparison.score_frame("h3", 0, self.rgb, self.rgb, np.zeros_like(self.mask))

    def test_means_are_per_frame_not_pixel_weighted(self):
        rows = [dict(route="h3", masked_ssim=score, rgb_mean_mae_255=mae, evaluated_pixels=pixels)
                for score, mae, pixels in ((0.2, 10, 1), (0.8, 30, 100))]
        self.assertEqual(comparison.metric_means(rows, "h3"),
                         dict(mean_masked_ssim=0.5, mean_rgb_mean_mae_255=20))

    def test_plot_contains_three_routes_and_does_not_clip_mae(self):
        rows = [dict(route=route, timestamp_seconds=index / 60, masked_ssim=0.8, rgb_mean_mae_255=110)
                for index in self.indices for route in comparison.METRIC_ROUTES]
        summary = {route: comparison.metric_means(rows, route) for route in comparison.METRIC_ROUTES}
        real_close = comparison.plt.close
        close = self.enterContext(patch.object(comparison.plt, "close"))
        self.enterContext(patch.object(comparison.plt.Figure, "savefig"))
        comparison.save_metrics(self.root, rows, summary)
        figure = close.call_args.args[0]
        for axis in figure.axes:
            self.assertEqual(len(axis.lines), 3)
            self.assertTrue(axis.lines[2].get_label().startswith("H3"))
        self.assertGreater(figure.axes[1].get_ylim()[1], 110)
        real_close(figure)


if __name__ == "__main__":
    unittest.main()
