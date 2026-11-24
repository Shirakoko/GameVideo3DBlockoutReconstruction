import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from blender.pipeline import json_write, sha256
from blender.planner import prepare_plan
from reconstruction.heuristic_proxy import AUTHORIZATION
from reconstruction.primitive_blockout import (boxes_from_cells, contains_camera,
                                               fit_primitives_from_points, merge_boxes, observed_points)
from scene_schema import SceneValidationError, validate_scene
from tests.test_refined import mesh_fixture


def covered(boxes):
    cells = set()
    for x0, x1, y0, y1, z0, z1 in boxes:
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                for z in range(z0, z1 + 1):
                    cells.add((x, y, z))
    return cells


class PrimitiveFitTests(unittest.TestCase):
    def test_merge_covers_each_solid_cell_once_and_keeps_holes(self):
        solid = np.array([[x, y, z] for x in range(2) for y in range(2) for z in range(2)])
        self.assertEqual(len(merge_boxes(solid)), 1)
        self.assertEqual(covered(merge_boxes(solid)), {tuple(cell) for cell in solid.tolist()})
        hole = np.array([[0, 0, 0], [2, 0, 0]])
        boxes = merge_boxes(hole)
        self.assertEqual(len(boxes), 2)
        self.assertNotIn((1, 0, 0), covered(boxes))

    def test_coarse_box_that_contains_a_camera_is_split(self):
        cells = np.array([[0, 0, 0], [3, 0, 0]])
        poses = np.repeat(np.eye(4)[None], 1, axis=0)
        poses[0, :3, 3] = [1.5, 0.5, 0.5]
        bounds = boxes_from_cells(cells, 1.0, poses, 4)
        self.assertGreaterEqual(len(bounds), 2)
        for low, high in bounds:
            self.assertFalse(np.all(poses[0, :3, 3] > low + 0.05) and np.all(poses[0, :3, 3] < high - 0.05))

    def test_observed_points_require_two_frame_support(self):
        from unittest import mock
        poses = np.repeat(np.eye(4)[None], 2, axis=0)
        poses[1, 0, 3] = 1.0
        k = np.eye(3) * 100
        k[0, 2] = k[1, 2] = 16
        lut = np.full(1024, 8.0)

        def raw(index):
            rgb = np.zeros((32, 32, 3), np.uint8)
            codes = np.full((32, 32), 200, np.uint16)
            labels = np.full((32, 32), 1, np.int32)
            if index == 0:
                labels[8:12, 8:12] = 2
            else:
                labels[20:24, 20:24] = 2
            return rgb, codes, labels

        with mock.patch("reconstruction.primitive_blockout.proxy_lut", return_value=(lut, dict(inferred=True))):
            points, categories, report = observed_points(raw, [0, 1], poses, k, stride=4, min_votes=2)
        self.assertFalse(report["column_fill"])
        self.assertFalse(report["floor_completion"])
        self.assertFalse(np.any(categories == 2))
        self.assertGreater(len(points), 0)

    def test_ground_tiles_keep_separate_heights_and_trees_use_point_bounds(self):
        low = np.array([[x, y, 0.2] for x in np.linspace(0, 1, 8) for y in np.linspace(0, 1, 8)])
        high = np.array([[x, y, 2.0] for x in np.linspace(10, 11, 8) for y in np.linspace(0, 1, 8)])
        trunk = np.array([[5 + dx, 5 + dy, z] for z in np.linspace(0.2, 1.2, 6)
                          for dx in (-0.05, 0.05) for dy in (-0.05, 0.05)])
        canopy = np.array([[5 + dx, 5 + dy, 1.6 + dz] for dx in np.linspace(-0.6, 0.6, 5)
                           for dy in np.linspace(-0.6, 0.6, 5) for dz in np.linspace(0, 0.8, 3)])
        points = np.vstack((low, high, trunk, canopy))
        categories = np.array([1] * len(low) + [1] * len(high) + [3] * (len(trunk) + len(canopy)))
        poses = np.repeat(np.eye(4)[None], 1, axis=0)
        poses[0, :3, 3] = [2, 2, 4]
        specs, counts = fit_primitives_from_points(points, categories, poses)
        grounds = [item for item in specs if item["category"] == "ground"]
        self.assertGreaterEqual(len(grounds), 2)
        heights = sorted(item["position"][2] for item in grounds)
        self.assertGreater(heights[-1] - heights[0], 1.0)
        canopy = next(item for item in specs if item["primitive_type"] == "sphere")
        self.assertGreater(canopy["dimensions"][0], 0.8)
        self.assertTrue(canopy["parent"].endswith("_trunk"))
        self.assertIn("cylinder", counts)

    def test_holes_are_not_filled_into_solid_walls(self):
        # Barrier uses 0.5 m cells; a one-cell gap must stay empty.
        points = np.array([[0.25, 0.25, 0.25], [1.25, 0.25, 0.25]])
        categories = np.array([4, 4])
        poses = np.repeat(np.eye(4)[None], 1, axis=0)
        poses[0, :3, 3] = [10, 10, 10]
        specs, _ = fit_primitives_from_points(points, categories, poses)
        self.assertEqual(len(specs), 2)
        for item in specs:
            self.assertLessEqual(max(item["dimensions"]), 0.5 + 1e-6)

    def test_tree_becomes_trunk_and_canopy_ground_stays_a_slab(self):
        ground = np.array([[x, y, 0] for x in range(4) for y in range(4)])
        tree = np.array([[10, 10, z] for z in range(8)])
        cells = np.vstack((ground, tree))
        categories = np.array([1] * len(ground) + [3] * len(tree))
        poses = np.repeat(np.eye(4)[None], 1, axis=0)
        poses[0, :3, 3] = [2, 2, 5]
        specs, counts = fit_primitives_from_points((cells + 0.5) * 0.5, categories, poses)
        self.assertGreaterEqual(counts.get("box", 0), 1)
        self.assertEqual(counts.get("cylinder", 0), 1)
        self.assertEqual(counts.get("sphere", 0), 1)
        canopy = next(item for item in specs if item["primitive_type"] == "sphere")
        self.assertTrue(canopy["parent"].endswith("_trunk"))
        self.assertTrue(all(not contains_camera(item, poses) for item in specs))

    def test_primitive_scene_is_a_separate_contract_from_voxels(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            scene = mesh_fixture(folder)
            confidence = scene["objects"][0]["confidence"]
            scene["objects"] = [dict(object_id="ground_0001", category="ground", primitive_type="box",
                transform=dict(position=[0, 0, 0], rotation=[0, 0, 0, 1], scale=[1, 1, 1]), dimensions=[4, 4, 0.5],
                parent_id=None, is_static=True, classification_confidence=confidence, confidence=confidence,
                tracking_confidence=confidence,
                geometry_source=dict(kind="heuristic_primitive", point_cloud_path="reference.ply",
                                     bounds_type="inferred_proxy_aabb", confidence=confidence),
                source_frames=[], observations=[], visibility=[], motion_track=[])]
            scene["appearance"]["texture_mode"] = "category_primitive"
            camera_hash = hashlib.sha256(json.dumps(scene["cameras"], sort_keys=True).encode()).hexdigest()
            proxy = dict(schema="heuristic.proxy/1", status="passed", authorization=AUTHORIZATION,
                         geometry_accuracy_validated=False, camera_inside_frame_count=0, coverage_status="passed",
                         camera_sha256=camera_hash)
            fit = dict(schema="heuristic.primitive/1", status="passed", geometry_accuracy_validated=False,
                       camera_inside_frame_count=0, camera_sha256=camera_hash)
            json_write(folder / "proxy.json", proxy)
            json_write(folder / "fit.json", fit)
            scene["reconstruction_metadata"]["heuristic_proxy"] = dict(path="proxy.json", sha256=sha256(folder / "proxy.json"))
            scene["reconstruction_metadata"]["primitive_fit"] = dict(path="fit.json", sha256=sha256(folder / "fit.json"))
            validate_scene(scene)
            json_write(folder / "scene.json", scene)
            video = dict(fps="60", frame_count=3, start_frame=0, first_pts_seconds=0, width=64, height=64)
            with self.assertRaisesRegex(ValueError, "allow-heuristic"):
                prepare_plan(folder / "scene.json", video, engine="BLENDER_EEVEE", allow_inferred=True)
            plan = prepare_plan(folder / "scene.json", video, engine="BLENDER_EEVEE", allow_inferred=True, allow_heuristic=True)
            self.assertEqual(plan["objects"][0]["primitive_type"], "box")
            self.assertEqual(plan["appearance"]["texture_mode"], "category_primitive")
            scene["objects"].append(dict(scene["objects"][0], object_id="voxel_block",
                                         geometry_source=dict(scene["objects"][0]["geometry_source"], kind="heuristic_voxel")))
            with self.assertRaisesRegex(SceneValidationError, "separate outputs"):
                validate_scene(scene)
            scene["objects"].pop()
            scene["reconstruction_metadata"].pop("primitive_fit")
            with self.assertRaisesRegex(SceneValidationError, "fit receipt"):
                validate_scene(scene)


if __name__ == "__main__":
    unittest.main()
