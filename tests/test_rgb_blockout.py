"""Multi-view RGB reconstruction does not depend on native depth units."""
import unittest

import cv2
import numpy as np

from reconstruction.refined_calibration import HOLDOUT, project, pose_track
from reconstruction.rgb_blockout import dense_triplet, rasterize_points, static_tracks, validate_tracks


class RGBTracksTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(42)
        k=np.array([[80.,0,64],[0,80,48],[0,0,1.]])
        uv=rng.uniform([35,20],[100,76],(200,2))
        rays=np.column_stack((uv,np.ones(len(uv))))@np.linalg.inv(k).T
        points=rays*rng.uniform(1.,2.,len(uv))[:,None]
        descriptors=rng.normal(size=(len(uv),128)).astype(np.float32)
        poses=[np.eye(4) for _ in range(720)]
        for i,p in enumerate(poses): p[0,3]=i*.001
        def feature(i):
            return project(points,poses[i],k)[0],descriptors
        return k,poses,feature

    def test_train_only_points_pass_heldout_projection(self):
        k,poses,feature=self.fixture()
        training=[0,12,24,36,48]
        tracks,lookup=static_tracks(training,feature,poses,k)
        self.assertTrue(all(node[0] not in HOLDOUT for _,nodes in tracks for node in nodes))
        report=validate_tracks(training,feature,tracks,lookup,poses,k)
        self.assertEqual(report["status"],"passed")
        self.assertLess(report["p90_pixels"],1e-8)
        self.assertGreaterEqual(report["sample_count"],100)

    def test_bad_holdouts_are_not_fit_or_filtered_to_pass(self):
        k,poses,feature=self.fixture()
        training=[0,12,24,36,48]
        tracks,lookup=static_tracks(training,feature,poses,k)
        def disturbed(i):
            uv,desc=feature(i)
            return (uv+[12,12] if i in HOLDOUT else uv),desc
        report=validate_tracks(training,disturbed,tracks,lookup,poses,k)
        self.assertEqual(report["status"],"failed")
        self.assertGreater(report["median_pixels"],10)

    def test_dense_three_view_plane_recovers_metric_shape(self):
        rng=np.random.default_rng(3)
        image=cv2.GaussianBlur(rng.integers(0,256,(96,128),dtype=np.uint8),(3,3),.7)
        k=np.array([[80.,0,64],[0,80,48],[0,0,1.]])
        poses=[np.eye(4) for _ in range(3)]
        images=[]
        for i,p in enumerate(poses):
            p[0,3]=i*.06
            shift=-80*p[0,3]/3
            view=cv2.warpAffine(image,np.array([[1.,0,shift],[0,1.,0]]),(128,96))
            images.append(np.repeat(view[:,:,None],3,axis=2))
        codes=np.full((96,128),150,np.uint16)  # Arbitrary code; never decoded.
        labels=np.ones((96,128),np.uint8)
        def raw(i): return images[i],codes,labels
        points,pixels=dense_triplet(0,1,2,raw,poses,k,stride=4)
        self.assertGreater(len(points),100)
        self.assertAlmostEqual(float(np.median(points[:,2])),3.,delta=.1)
        self.assertEqual(len(pixels[0]),len(points))

    def test_dense_person_mask_yields_no_environment_geometry(self):
        rgb=np.zeros((32,32,3),np.uint8)
        def raw(i): return rgb,np.full((32,32),150,np.uint16),np.full((32,32),8,np.uint8)
        points,pixels=dense_triplet(0,1,2,raw,[np.eye(4)]*3,np.eye(3))
        self.assertEqual(len(points),0)

    def test_raster_z_buffer_keeps_nearest_positive_surface(self):
        raster=np.full((4,4),np.inf,np.float32)
        rasterize_points(raster,np.array([[1.,1.],[1.,1.],[-1.,1.],[2.,2.]]),np.array([3.,2.,1.,-1.]))
        self.assertEqual(raster[1,1],2.)
        self.assertEqual(np.isfinite(raster).sum(),1)

    def test_pose_offset_preserves_track_length_and_explicit_boundary(self):
        rows=[dict(camera_raw=dict(px=i*100.,py=0.,pz=0.,pitch=0.,yaw=0.,roll=0.)) for i in range(3)]
        poses=pose_track(rows,100,frame_offset=1)
        self.assertEqual(len(poses),3)
        np.testing.assert_allclose([p[0,3] for p in poses],[1.,2.,2.])


if __name__=="__main__": unittest.main()
