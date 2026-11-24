import unittest
from pathlib import Path
import tempfile
import json

import numpy as np

from reconstruction.rgb_hull import carve_envelope, ground_plane, envelope_raycast_scene, envelope_depth, volume_camera_check
from reconstruction.refined_voxels import voxel_shell
from scripts.render_rgb_blockout import render


class HullTests(unittest.TestCase):
    def fixture(self):
        cells=np.array([[0,0,4]])
        poses=[np.eye(4)]*5
        k=np.array([[20.,0,16],[0,20.,16],[0,0,1.]])
        rgb=np.full((32,32,3),128,np.uint8)
        codes=np.full((32,32),150,np.uint16)
        labels=np.full((32,32),2,np.uint8)
        return cells,poses,k,rgb,codes,labels

    def test_envelope_needs_three_supported_training_views(self):
        cells,poses,k,rgb,codes,labels=self.fixture()
        raw=lambda i:(rgb,codes,labels)
        result,colors,votes=carve_envelope(cells,1.,"rock",[0,1,2],raw,poses,k)
        np.testing.assert_array_equal(result,cells)
        np.testing.assert_allclose(colors,128/255)
        self.assertEqual(votes.tolist(),[3])
        result,_,_=carve_envelope(cells,1.,"rock",[0,1],raw,poses,k)
        self.assertEqual(len(result),0)

    def test_clear_sky_and_person_masks_do_not_create_solids(self):
        cells,poses,k,rgb,codes,labels=self.fixture()
        def raw(i): return rgb,np.full_like(codes,1023) if i<2 else codes,labels
        result,_,_=carve_envelope(cells,1.,"rock",range(5),raw,poses,k)
        self.assertEqual(len(result),0)
        raw=lambda i:(rgb,codes,np.full_like(labels,8))
        result,_,_=carve_envelope(cells,1.,"rock",range(5),raw,poses,k)
        self.assertEqual(len(result),0)

    def test_ground_plane_stays_inside_measured_xy_footprint(self):
        x,y=np.meshgrid(np.linspace(-2,2,12),np.linspace(-2,2,12))
        points=np.column_stack((x.ravel(),y.ravel(),np.full(x.size,1.5)))
        plane,hull,count=ground_plane(points)
        self.assertGreater(count,100)
        self.assertGreaterEqual(hull.find_simplex([[0.,0.]])[0],0)
        self.assertLess(hull.find_simplex([[100.,100.]])[0],0)
        self.assertLess(abs(np.dot(plane[:3],[0,0,1.5])+plane[3]),1e-5)

    def test_render_wrapper_never_bypasses_failed_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            (output/"refinement_report.json").write_text(json.dumps(dict(schema="refined.run/1",geometry_method="rgb_three_view",
                calibration=dict(status="passed"),quality=dict(status="failed"))))
            with self.assertRaisesRegex(ValueError,"preview must both pass"):
                render(output)

    def test_raycast_faces_fill_between_projected_voxel_centres(self):
        mesh=voxel_shell(np.array([[0,0,4]]),np.ones((1,3)),1.)
        scene=envelope_raycast_scene({"rock":mesh})
        k=np.array([[20.,0,16],[0,20.,16],[0,0,1.]])
        depth,valid=envelope_depth(scene,np.eye(4),k,32,32)
        self.assertTrue(valid[17,17])
        self.assertTrue(valid[19,19])
        self.assertAlmostEqual(depth[17,17],4.,places=5)
        self.assertFalse(valid[0,0])

    def test_raycast_depth_is_optical_z_not_euclidean_range(self):
        mesh=voxel_shell(np.array([[0,0,4]]),np.ones((1,3)),1.)
        pose=np.eye(4); pose[0,3]=-.25
        k=np.array([[20.,0,16],[0,20.,16],[0,0,1.]])
        depth,valid=envelope_depth(envelope_raycast_scene({"rock":mesh}),pose,k,32,32)
        self.assertTrue(valid[19,20])
        self.assertAlmostEqual(depth[19,20],4.,places=5)

    def test_volume_cannot_enclose_the_camera_path(self):
        pose=np.eye(4); pose[:3,3]=[.5,.5,.5]
        bad=volume_camera_check(np.array([[0,0,0]]),1.,[pose])
        self.assertEqual(bad["status"],"failed")
        self.assertEqual(bad["camera_inside_frames"],[0])
        good=volume_camera_check(np.array([[0,0,4]]),1.,[pose])
        self.assertEqual(good["status"],"passed")

    def test_passing_image_score_cannot_bypass_camera_intersections(self):
        from blender.pipeline import sha256
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory); cache=output/"solid_voxels.npz"
            np.savez_compressed(cache,cells=np.array([[0,0,0]]))
            pose=np.eye(4); pose[:3,3]=[.5,.5,.5]
            (output/"scene_schema.json").write_text(json.dumps(dict(cameras=[dict(track=[dict(world_from_camera=pose.tolist())])])))
            report=dict(schema="refined.run/1",geometry_method="rgb_three_view",volume_envelope_inferred=True,
                calibration=dict(status="passed"),quality=dict(status="passed"),block_size_metres=1.,
                voxel_cache_sha256=sha256(cache))
            (output/"refinement_report.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,"enclose the original camera"):
                render(output)
            saved=json.loads((output/"refinement_report.json").read_text())
            self.assertEqual(saved["status"],"quality_rejected")
            self.assertFalse(saved["full_video_rendered"])
            self.assertEqual(saved["quality"]["image_agreement_status"],"passed")
            self.assertEqual(saved["quality"]["status"],"failed")


if __name__=="__main__": unittest.main()
