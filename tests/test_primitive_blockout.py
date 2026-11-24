import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from blender.pipeline import json_write, sha256
from blender.planner import prepare_plan
from reconstruction.heuristic_proxy import AUTHORIZATION
from reconstruction.primitive_blockout import boxes_from_cells, contains_camera, fit_primitives, merge_boxes
from scene_schema import SceneValidationError, validate_scene
from tests.test_refined import mesh_fixture


def covered(boxes):
    cells = set()
    for x0, x1, y0, y1, z0, z1 in boxes:
        for x in range(x0, x1+1):
            for y in range(y0, y1+1):
                for z in range(z0, z1+1):
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
            self.assertFalse(np.all(poses[0, :3, 3] > low+0.05) and np.all(poses[0, :3, 3] < high-0.05))

    def test_tree_becomes_trunk_and_canopy_ground_stays_a_slab(self):
        ground = np.array([[x, y, 0] for x in range(4) for y in range(4)])
        tree = np.array([[10, 10, z] for z in range(8)])
        cells = np.vstack((ground, tree))
        categories = np.array([1]*len(ground)+[3]*len(tree))
        poses = np.repeat(np.eye(4)[None], 1, axis=0)
        poses[0, :3, 3] = [2, 2, 5]
        specs, counts = fit_primitives(cells, categories, 0.5, poses)
        self.assertGreaterEqual(counts["box"], 1)
        self.assertEqual(counts["cylinder"], 1)
        self.assertEqual(counts["sphere"], 1)
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
                parent_id=None, is_static=True, classification_confidence=confidence, confidence=confidence, tracking_confidence=confidence,
                geometry_source=dict(kind="heuristic_primitive", point_cloud_path="reference.ply", bounds_type="inferred_proxy_aabb", confidence=confidence),
                source_frames=[], observations=[], visibility=[], motion_track=[])]
            scene["appearance"]["texture_mode"] = "clay_primitive"
            camera_hash = hashlib.sha256(json.dumps(scene["cameras"], sort_keys=True).encode()).hexdigest()
            proxy = dict(schema="heuristic.proxy/1", status="passed", authorization=AUTHORIZATION,
                         geometry_accuracy_validated=False, camera_inside_frame_count=0, coverage_status="passed", camera_sha256=camera_hash)
            fit = dict(schema="heuristic.primitive/1", status="passed", geometry_accuracy_validated=False,
                       camera_inside_frame_count=0, camera_sha256=camera_hash)
            json_write(folder/"proxy.json", proxy)
            json_write(folder/"fit.json", fit)
            scene["reconstruction_metadata"]["heuristic_proxy"] = dict(path="proxy.json", sha256=sha256(folder/"proxy.json"))
            scene["reconstruction_metadata"]["primitive_fit"] = dict(path="fit.json", sha256=sha256(folder/"fit.json"))
            validate_scene(scene)
            json_write(folder/"scene.json", scene)
            video = dict(fps="60", frame_count=3, start_frame=0, first_pts_seconds=0, width=64, height=64)
            with self.assertRaisesRegex(ValueError, "allow-heuristic"):
                prepare_plan(folder/"scene.json", video, engine="BLENDER_EEVEE", allow_inferred=True)
            plan = prepare_plan(folder/"scene.json", video, engine="BLENDER_EEVEE", allow_inferred=True, allow_heuristic=True)
            self.assertEqual(plan["objects"][0]["primitive_type"], "box")
            self.assertIn("clay primitives", plan["reference_evidence"])
            scene["objects"].append(dict(scene["objects"][0], object_id="voxel_block", geometry_source=dict(scene["objects"][0]["geometry_source"], kind="heuristic_voxel")))
            with self.assertRaisesRegex(SceneValidationError, "separate outputs"):
                validate_scene(scene)
            scene["objects"].pop()
            scene["reconstruction_metadata"].pop("primitive_fit")
            with self.assertRaisesRegex(SceneValidationError, "fit receipt"):
                validate_scene(scene)


if __name__ == "__main__":
    unittest.main()
