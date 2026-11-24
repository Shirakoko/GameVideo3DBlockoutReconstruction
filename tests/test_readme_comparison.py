import importlib.util
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


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
            "output_sha256": "video-hash", "probe": {"width": 2, "height": 1},
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


if __name__ == "__main__":
    unittest.main()
