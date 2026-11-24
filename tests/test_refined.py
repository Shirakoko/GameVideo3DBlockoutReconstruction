"""Independent numerical, asset, provenance and background mesh tests."""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import open3d as o3d
from PIL import Image

from agents.object_detection import confidence
from blender.media import command
from blender.pipeline import blender_path, json_write, run_pipeline, sha256
from blender.planner import prepare_plan, source_scene_snapshot
from reconstruction.refined_calibration import (HOLDOUT, decode_empirical, epipolar_errors,
    fit_curve, project, selected_frames, triangulate, unproject_inferred, validate_predictions)
from reconstruction.refined_geometry import depth_surface, inspect_glb, write_glb
from scene_schema import validate_scene, SceneValidationError
from stage_a.media import ffmpeg_path


def mesh_fixture(folder):
    vertices = np.array([[x, y, z] for x, y, z in
        [(-1, -1.5, -2), (1, -1.5, -2), (1, 1.5, -2), (-1, 1.5, -2),
         (-1, -1.5, 2), (1, -1.5, 2), (1, 1.5, 2), (-1, 1.5, 2)]], dtype=float)
    faces = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                      [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]], dtype=np.uint32)
    uv = np.array([[0,0],[1,0],[1,1],[0,1]]*2, float)
    png = BytesIO()
    Image.new("RGB", (8, 8), (180, 70, 40)).save(png, format="PNG")
    write_glb(folder/"mesh.glb", vertices, faces, uv, png.getvalue())
    o3d.io.write_point_cloud(str(folder/"reference.ply"), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices+[0, 0, 8])))
    Image.new("L", (64, 64), 255).save(folder/"mask.png")
    json_write(folder/"calibration.json", dict(schema="refined.calibration/1", status="passed", physical_contract_verified=False))
    c = confidence(.6, "inferred", "synthetic_mesh_test", "SYNTHETIC analytic known mesh, not real Session validation")
    frames = [dict(frame_index=1000+i, timestamp_seconds=i/60, original_tick=8000+i,
                   camera_id="main", source_observation_id=str(i)) for i in range(3)]
    transform = dict(position=[0,0,8], rotation=[0,0,0,1], scale=[1,1,1])
    obj = dict(object_id="cube", category="rock", primitive_type="mesh", mesh_asset=dict(path="mesh.glb", sha256=sha256(folder/"mesh.glb"), format="glb"),
        transform=transform, dimensions=[2,3,4], parent_id=None, is_static=True,
        classification_confidence=c, confidence=c, tracking_confidence=c,
        geometry_source=dict(kind="surface_mesh", point_cloud_path="reference.ply", bounds_type="observed_mesh_aabb", confidence=c),
        source_frames=[1000], motion_track=[], visibility=[dict(frame_index=1000,timestamp_seconds=0,camera_id="main",state="visible")],
        observations=[dict(frame_index=1000,timestamp_seconds=0,camera_id="main",bbox_xyxy=[0,0,64,64],raster_size=[64,64],
            mask=dict(path="mask.png",kind="instance",confidence=c),position=[0,0,8],dimensions=[2,3,4],geometry_confidence=c,point_count=8)])
    scene = dict(schema_version="1.1.0", scene_id="SYNTHETIC-refined-test", scene_type=dict(value="test", confidence=c),
        created_at="2026-11-24T00:00:00+08:00",coordinate_system="blender_rh_z_up",unit_scale=1.,objects=[obj],
        cameras=[dict(camera_id="main", local_axes="optical_x_right_y_down_z_forward", raw_source=None,
            track=[dict(frame_index=f["frame_index"], timestamp_seconds=f["timestamp_seconds"], world_from_camera=np.eye(4).tolist(),
                intrinsics=[[50,0,32],[0,50,32],[0,0,1]],confidence=c) for f in frames])],
        timeline=dict(fps=60,start_seconds=0,end_seconds=2/60,time_origin="synthetic seconds",frames=frames), relationships=[],
        appearance=dict(texture_mode="baked_rgb",background_color=[.2,.2,.2]),
        reconstruction_metadata=dict(status="complete_inferred",data_source="SYNTHETIC ONLY",geometry_contract_verified=False,
            reconstruction_mode="inferred", empirical_validation=dict(path="calibration.json", sha256=sha256(folder/"calibration.json")),
            provider="analytic_test",limitations=[]))
    json_write(folder/"scene.json", scene)
    return scene


class CalibrationTests(unittest.TestCase):
    def test_sampling_and_holdout_separation(self):
        frames = selected_frames(720)
        self.assertEqual(frames[0], 0)
        self.assertEqual(frames[-1], 719)
        self.assertEqual(len(frames), 61)
        self.assertFalse(set(HOLDOUT) & set(i for i in frames if i not in HOLDOUT))

    def test_monotonic_fit_and_no_extrapolation(self):
        codes = np.arange(100, 900)
        curve = fit_curve(codes, np.exp(codes/250))
        depth, valid = decode_empirical(np.array([0, 99, 300, 700, 1023]), curve)
        self.assertEqual(valid.tolist(), [False,False,True,True,False])
        self.assertGreater(depth[3], depth[2])
        self.assertFalse(curve["physical_contract_verified"])
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            fit_curve([10], [1])

    def test_triangulation_epipolar_and_cheirality(self):
        a, b = np.eye(4), np.eye(4)
        b[0,3] = 1.
        k = np.array([[200,0,100],[0,200,100],[0,0,1.]])
        p = np.array([[0,0,5],[.5,.2,7],[-.2,-.3,9]])
        ua, _ = project(p,a,k)
        ub, _ = project(p,b,k)
        np.testing.assert_allclose(epipolar_errors(ua,ub,a,b,k),0,atol=1e-10)
        recovered, keep = triangulate(ua,ub,a,b,k)
        self.assertTrue(keep.all())
        np.testing.assert_allclose(recovered,p,atol=1e-9)

    def test_inferred_unprojection_does_not_require_or_fabricate_verified_spec(self):
        d = np.full((4,4),2.,np.float32)
        p,uv = unproject_inferred(d,np.eye(3),np.ones((4,4),bool),stride=2)
        self.assertEqual(len(p),4)
        np.testing.assert_allclose(p[1],[4,0,2])

    def test_insufficient_holdout_never_passes(self):
        curve=fit_curve(np.arange(100,900),np.exp(np.arange(100,900)/250))
        report=validate_predictions([],curve,[np.eye(4)],np.eye(3),lambda i: None)
        self.assertEqual(report["status"],"failed")

    def test_surface_does_not_bridge_depth_discontinuity(self):
        depth=np.ones((8,8),np.float32)
        depth[:,4:]=10
        mesh=depth_surface(depth,np.ones((8,8),bool),np.eye(3),np.eye(4),stride=1)
        triangles=np.asarray(mesh.triangles)
        z=np.asarray(mesh.vertices)[triangles,2]
        self.assertTrue((np.ptp(z,axis=1)==0).all())
        empty=depth_surface(depth,np.zeros((8,8),bool),np.eye(3),np.eye(4),stride=1)
        self.assertEqual(len(empty.triangles),0)


class RefinedContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.folder=Path(self.temp.name)
        self.scene=mesh_fixture(self.folder)
        self.video=dict(fps="60",frame_count=3,start_frame=0,first_pts_seconds=0,width=64,height=64)

    def tearDown(self):
        self.temp.cleanup()

    def test_explicit_opt_in_and_source_snapshot(self):
        validate_scene(self.scene)
        with self.assertRaisesRegex(ValueError,"allow-inferred"):
            prepare_plan(self.folder/"scene.json",self.video,engine="CYCLES")
        plan=prepare_plan(self.folder/"scene.json",self.video,engine="CYCLES",allow_inferred=True)
        self.assertEqual(plan["reconstruction_mode"],"inferred")
        self.assertTrue(Path(plan["objects"][0]["mesh_asset"]["path"]).is_absolute())
        snapshot=source_scene_snapshot(self.scene,self.folder/"scene.json")
        self.assertTrue(Path(snapshot["objects"][0]["mesh_asset"]["path"]).is_absolute())

    def test_no_verified_flag_upgrade_and_no_1_0_bypass(self):
        modified=deepcopy(self.scene)
        modified["reconstruction_metadata"]["geometry_contract_verified"]=True
        with self.assertRaises(SceneValidationError): validate_scene(modified)
        modified["schema_version"]="1.0.0"
        with self.assertRaises(SceneValidationError): validate_scene(modified)

    def test_mesh_and_calibration_hashes_are_checked(self):
        (self.folder/"mesh.glb").write_bytes(b"tamper")
        with self.assertRaisesRegex(ValueError,"mesh asset"):
            prepare_plan(self.folder/"scene.json",self.video,engine="CYCLES",allow_inferred=True)
        (self.folder/"calibration.json").write_text("{}")
        with self.assertRaisesRegex(ValueError,"calibration receipt"):
            prepare_plan(self.folder/"scene.json",self.video,engine="CYCLES",allow_inferred=True)

    def test_glb_bounds_and_truncation(self):
        self.assertEqual(inspect_glb(self.folder/"mesh.glb")["faces"],12)
        (self.folder/"mesh.glb").write_bytes(b"glTF")
        with self.assertRaisesRegex(ValueError,"Truncated"):
            inspect_glb(self.folder/"mesh.glb")

    def test_background_textured_mesh_and_packed_texture(self):
        try: executable=blender_path()
        except ValueError: self.skipTest("Configure Blender for refined integration")
        original=self.folder/"original.mp4"
        command([ffmpeg_path(),"-y","-nostdin","-f","lavfi","-i","testsrc=size=64x64:rate=60",
            "-frames:v","3","-c:v","libx264","-pix_fmt","yuv420p",str(original)])
        output=self.folder/"output"
        report=run_pipeline(self.folder/"scene.json",original,output,blender=executable,
            engine="CYCLES",samples=1,allow_inferred=True)
        self.assertEqual(report["scene_verification"]["objects"],1)
        self.assertTrue(report["scene_verification"]["reopened_saved_blend"])
        self.assertEqual(report["media"]["blockout_video.mp4"]["frame_count"],3)
        cached=run_pipeline(self.folder/"scene.json",original,output,blender=executable,
            engine="CYCLES",samples=1,allow_inferred=True,reuse=True)
        self.assertEqual(cached["created_at"],report["created_at"])


if __name__=="__main__": unittest.main()
