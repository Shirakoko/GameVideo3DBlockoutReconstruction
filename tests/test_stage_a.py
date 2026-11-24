"""Synthetic tests only. No HF payload and no claim of real-session validation."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from stage_a.access import Client, InspectionError, REQUIRED, rank_samples
from stage_a.depth import (UnverifiedContractError, metric_depth, relative_log_depth,
                           validate_intrinsics, validate_near_plane_cm, decode_v3_raw)
from stage_a.mp4 import boxes, video_prefix_end, relocate_chunks
from stage_a.media import DecodeError, DuplicateTracker, ffmpeg_path, iter_frames, probe_local
from stage_a.sync import SyncError, compare_timestamps, join_frames, read_jsonl


class DepthTests(unittest.TestCase):
    def test_relative_endpoints(self):
        np.testing.assert_allclose(relative_log_depth(np.array([0, 1]), log_constant=4000), [0, 1])

    def test_relative_roundtrip(self):
        values = np.linspace(0, 1, 1024)
        q = np.log1p(4000*values) / np.log1p(4000)
        np.testing.assert_allclose(relative_log_depth(q, log_constant=4000), values, atol=1e-12)

    def test_invalid_q(self):
        for q in (-1, 1.1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                relative_log_depth(np.array([q]), log_constant=4000)

    def test_no_default_constant(self):
        with self.assertRaises(TypeError):
            relative_log_depth(np.array([0.5]))

    def test_bad_constant(self):
        for value in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                relative_log_depth(np.array([0.5]), log_constant=value)

    def test_metric_is_blocked(self):
        with self.assertRaises(UnverifiedContractError):
            metric_depth(np.array([1023], dtype=np.uint16), near_plane_cm=10)

    def test_near_plane_required(self):
        for value in (None, 0, -1, float("nan"), True, "10"):
            with self.assertRaises(ValueError):
                validate_near_plane_cm(value)
        self.assertEqual(validate_near_plane_cm(10), 10.0)

    def test_intrinsics(self):
        result = validate_intrinsics([[100, 0, 16], [0, 100, 16], [0, 0, 1]], width=32, height=32)
        self.assertEqual(result["width"], 32)
        for k in ([[0, 0, 16], [0, 100, 16], [0, 0, 1]],
                  [[100, 0, 100], [0, 100, 16], [0, 0, 1]],
                  [[100, 0, 16], [0, 100, 16], [0, 0, 2]]):
            with self.assertRaises(ValueError):
                validate_intrinsics(k, width=32, height=32)

    def test_v3_raw_masks_reserved_codes_without_metric_conversion(self):
        contract = {"schema": "originlab.depth.decode_contract/1", "depth_transform": "log_sentinel_v3",
                    "vui_full_range": True, "near_plane_cm": 10,
                    "transform_constants": {"K": 4000, "far_plane_encode": 100000, "sentinel_code10": 1023, "max_valid_code10": 1020}}
        codes = np.array([[0, 1020, 1021, 1022, 1023]], dtype=np.uint16)
        raw, mask = decode_v3_raw(codes, contract)
        np.testing.assert_array_equal(raw, codes)
        np.testing.assert_array_equal(mask, [[True, True, False, False, False]])
        with self.assertRaises(ValueError):
            decode_v3_raw(codes.astype(np.float32), contract)
        contract["vui_full_range"] = False
        with self.assertRaises(UnverifiedContractError):
            decode_v3_raw(codes, contract)


class MP4Tests(unittest.TestCase):
    def fixture(self):
        import struct
        def box(kind, payload): return struct.pack(">I4s", len(payload)+8, kind)+payload
        co = box(b"stco", struct.pack(">IIII", 0, 2, 1000, 2000))
        sc = box(b"stsc", struct.pack(">IIIII", 0, 1, 1, 2, 1))
        sz = box(b"stsz", struct.pack(">IIIIIII", 0, 0, 4, 10, 20, 30, 40))
        stbl = box(b"stbl", co+sc+sz)
        handler = box(b"hdlr", b"\0"*8+b"vide"+b"\0"*12)
        return box(b"moov", box(b"trak", box(b"mdia", handler+box(b"minf", stbl))))

    def test_sample_chunk_prefix(self):
        end, info = video_prefix_end(self.fixture(), 3)
        self.assertEqual(end, 2070)
        self.assertEqual(info["prefix_video_samples_including_preroll"], 4)

    def test_relocate_all_offsets(self):
        end, _ = video_prefix_end(relocate_chunks(self.fixture(), 100), 3)
        self.assertEqual(end, 2170)

    def test_truncated_box_rejected(self):
        with self.assertRaises(InspectionError):
            list(boxes(b"\0\0\0\x10moov"))

    def test_sample_out_of_bounds(self):
        with self.assertRaises(InspectionError):
            video_prefix_end(self.fixture(), 5)


class SyncTests(unittest.TestCase):
    def test_nonzero_frame_index_not_row_number(self):
        rows, report = join_frames([100, 101], [100, 101], [{"frame": 100}, {"frame": 101}], camera_frame_key="frame")
        self.assertEqual(rows[0]["frame_index"], 100)
        self.assertEqual(rows[0]["depth_clip_ordinal"], 0)
        self.assertTrue(report["index_join_complete"])
        self.assertFalse(report["mapping_provenance_verified"])

    def test_duplicate_camera_samples_preserved(self):
        records = [{"frame": 10, "qpc": 100}, {"frame": 10, "qpc": 101}]
        rows, report = join_frames([10], [10], records, camera_frame_key="frame")
        self.assertEqual(rows[0]["camera_samples"], records)
        self.assertEqual(report["multiple_camera_sample_frame_indices"], [10])
        self.assertFalse(report["single_pose_rule_verified"])

    def test_gaps_and_missing_pose(self):
        _, report = join_frames([10, 12], [10, 11], [{"frame": 10}], camera_frame_key="frame")
        self.assertEqual(report["missing_depth_frame_indices"], [12])
        self.assertEqual(report["extra_depth_frame_indices"], [11])
        self.assertEqual(report["missing_camera_frame_indices"], [12])
        self.assertEqual(report["rgb_index_gap_edges"], [[10, 12]])
        self.assertFalse(report["index_join_complete"])

    def test_reject_duplicate_or_backward_frame_indices(self):
        for indices in ([0, 0], [1, 0], [True], [-1], [0.0]):
            with self.assertRaises(SyncError):
                join_frames(indices, [0], [], camera_frame_key="frame")

    def test_no_camera_timestamp_fallback(self):
        with self.assertRaises(SyncError):
            join_frames([0], [0], [{"qpc": 123}], camera_frame_key="frame")

    def test_timestamp_tolerance(self):
        self.assertTrue(compare_timestamps([0, 1/60], [0.0001, 1/60+0.0001], tolerance_seconds=0.001)["within_tolerance"])
        self.assertFalse(compare_timestamps([0, 1/60], [0.01, 1/60+0.01], tolerance_seconds=0.001)["within_tolerance"])

    def test_bad_timestamps(self):
        for timestamps in ([0, 0], [1, 0], [0, float("nan")]):
            with self.assertRaises(SyncError):
                compare_timestamps(timestamps, [0, 1], tolerance_seconds=0.001)

    def test_jsonl_strict_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.jsonl"
            path.write_text('{"frame":10,"qpc":123,"position":[1,2,3]}\n', encoding="utf-8")
            self.assertEqual(list(read_jsonl(path))[0]["frame"], 10)
            with self.assertRaises(SyncError):
                list(read_jsonl(path, max_line_bytes=5))
            path.write_text('{"frame":NaN}\n', encoding="utf-8")
            with self.assertRaises(SyncError):
                list(read_jsonl(path))


class AccessTests(unittest.TestCase):
    def test_rank_by_required_size_not_uuid(self):
        entries = [{"type": "file", "path": f"samples/{session}/{name}", "size": size}
                   for session, size in (("large", 20), ("small", 10)) for name in REQUIRED]
        self.assertEqual(rank_samples(entries)[0]["session_id"], "small")

    def test_missing_paths_not_complete(self):
        result = rank_samples([{"type": "file", "path": "samples/one/session.json", "size": 10}])[0]
        self.assertIn("depth/decode_contract.json", result["missing_required_paths"])

    def test_client_rejects_untrusted_url(self):
        for url in ("https://example.com/file", "http://huggingface.co/file"):
            with self.assertRaises(InspectionError):
                Client().request(url)

    def test_client_byte_cap(self):
        class FakeResponse:
            code = 200
            headers = {}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, size): return b"x"*size
        client = Client(max_total_bytes=4)
        with patch.object(client.opener, "open", return_value=FakeResponse()):
            with self.assertRaises(InspectionError):
                client.request("https://huggingface.co/api/test", max_bytes=8)
        self.assertEqual(client.bytes_read, 5)  # bounded single-byte overrun detector


class MediaIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.depth_path = cls.root / "synthetic_depth.hevc"
        cls.rgb_path = cls.root / "synthetic_rgb.mp4"
        cls.count = 720  # 12 seconds @ 60 FPS, deliberately tiny raster.
        cls.raster = 32
        # All 1024 codes including 0, 64, 940, 1023: detects unwanted range scaling.
        base = np.arange(1024, dtype="<u2").reshape(32, 32)
        cls.expected = [np.roll(base, index//2, axis=1) for index in range(cls.count)]
        chroma = np.full(32*32//2, 512, dtype="<u2").tobytes()
        depth_raw = b"".join(frame.tobytes()+chroma for frame in cls.expected)
        depth_cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                     "-pix_fmt", "yuv420p10le", "-s", "32x32", "-r", "60", "-i", "pipe:0",
                     "-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "lossless=1:log-level=error:pools=1:frame-threads=1",
                     "-f", "hevc", str(cls.depth_path)]
        result = subprocess.run(depth_cmd, input=depth_raw, capture_output=True, timeout=60)
        if result.returncode:
            raise RuntimeError("Synthetic HEVC encode failed: " + result.stderr.decode(errors="replace"))
        rgb_raw = b"".join(np.full((32, 32, 3), [(index//2)%256, 80, 160], dtype=np.uint8).tobytes() for index in range(cls.count))
        rgb_cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", "32x32", "-r", "60", "-i", "pipe:0", "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p", str(cls.rgb_path)]
        result = subprocess.run(rgb_cmd, input=rgb_raw, capture_output=True, timeout=60)
        if result.returncode:
            raise RuntimeError("Synthetic RGB encode failed: " + result.stderr.decode(errors="replace"))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_native_ten_bit_codes_and_duplicates_12_seconds(self):
        tracker = DuplicateTracker()
        decoded = list(iter_frames(self.depth_path, start=0, count=self.count, depth=True, max_decoded_bytes=3*1024**2))
        self.assertEqual(len(decoded), self.count)
        for (ordinal, actual), expected in zip(decoded, self.expected):
            self.assertEqual(actual.dtype, np.uint16)
            np.testing.assert_array_equal(actual, expected)
            tracker.add(actual)
        self.assertEqual(tracker.adjacent_equal, 360)

    def test_nonzero_clip_start(self):
        frames = list(iter_frames(self.depth_path, start=100, count=3, depth=True))
        self.assertEqual([ordinal for ordinal, _ in frames], [100, 101, 102])
        np.testing.assert_array_equal(frames[0][1], self.expected[100])

    def test_rgb_12_seconds(self):
        spec = probe_local(self.rgb_path)
        self.assertEqual((spec.width, spec.height), (32, 32))
        frames = list(iter_frames(self.rgb_path, start=0, count=self.count, max_decoded_bytes=3*1024**2))
        self.assertEqual(len(frames), self.count)
        self.assertEqual(frames[0][1].shape, (32, 32, 3))
        self.assertEqual(frames[0][1].dtype, np.uint8)

    def test_short_stream_fails(self):
        with self.assertRaises(DecodeError):
            list(iter_frames(self.depth_path, start=719, count=2, depth=True))

    def test_byte_budget_fails_before_decode(self):
        with self.assertRaises(DecodeError):
            list(iter_frames(self.depth_path, start=0, count=720, depth=True, max_decoded_bytes=100))

    def test_depth_rejects_rgb_format(self):
        # Supplying a regular RGB video as depth must fail, not silently convert it.
        with self.assertRaises(DecodeError):
            list(iter_frames(self.rgb_path, start=0, count=1, depth=True))

    def test_synthetic_sync_fixture_and_diagnostic(self):
        original_indices = list(range(1000, 1000+self.count))
        cameras = [{"frame": index, "qpc": 10_000_000+ordinal*166667,
                    "position": [ordinal*0.01, 0, 0], "rotation": [0, 0, 0]}
                   for ordinal, index in enumerate(original_indices)]
        rows, report = join_frames(original_indices, original_indices, cameras, camera_frame_key="frame")
        self.assertTrue(report["index_join_complete"])
        self.assertEqual(len(rows), 720)
        output = Path("outputs/stage_a/tests")
        output.mkdir(parents=True, exist_ok=True)
        details = {"data_source": "SYNTHETIC ONLY — NOT HUGGING FACE", "duration_seconds": 12,
                   "fps": 60, "raster": [32, 32], "frames": 720,
                   "native_10bit_codes": "lossless all-code comparison; no luma range scaling",
                   "expected_adjacent_depth_duplicates": 360, "index_join": report,
                   "session_metric_depth_validated": False, "ffmpeg": ffmpeg_path()}
        (output / "synthetic_fixture_report.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
        picture = Image.new("RGB", (720, 220), "white")
        draw = ImageDraw.Draw(picture)
        draw.text((15, 10), "SYNTHETIC TEST ONLY / NOT A DATASET SAMPLE", fill="black")
        for column, ordinal in enumerate((0, 120, 360, 719)):
            code = self.expected[ordinal]
            gray = (code/1023*255).astype(np.uint8)
            tile = Image.fromarray(gray).resize((140, 140), Image.Resampling.NEAREST).convert("RGB")
            picture.paste(tile, (15+column*180, 40))
            draw.text((15+column*180, 190), f"index={1000+ordinal} t={ordinal/60:.2f}s", fill="black")
        picture.save(output / "synthetic_code_diagnostic.png")


if __name__ == "__main__":
    unittest.main()
