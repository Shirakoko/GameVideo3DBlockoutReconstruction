"""Low-poly geometry, alignment, cache and inference guard regressions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from blender.pipeline import json_write, sha256, blender_path, run_pipeline
from blender.media import command
from reconstruction.refine import refinement_lock, verified_frame_cache, make_scene, render_previews
from reconstruction.refined_calibration import align_depth, decode_empirical, fit_smooth_curve, sample_depth_curve, calibrate, project
from reconstruction.refined_geometry import split_chunks, inspect_glb, write_glb
from reconstruction.refined_voxels import VoxelFusion, palette_mesh, voxel_shell
from scene_schema import validate_scene
from stage_a.media import ffmpeg_path
from tests.test_refined import mesh_fixture


class VoxelTests(unittest.TestCase):
    def test_internal_faces_culled_and_bounds_preserved(self):
        mesh = voxel_shell([[0,0,0], [1,0,0]], [[.5,.3,.2], [.2,.8,.1]], .2)
        self.assertEqual(len(mesh.triangles), 20)  # 12 + 12 - 4 shared triangles
        np.testing.assert_allclose(mesh.get_min_bound(), [0,0,0])
        np.testing.assert_allclose(mesh.get_max_bound(), [.4,.2,.2])
        self.assertTrue(mesh.has_vertex_colors())

    def test_palette_uniform_per_face_and_glb(self):
        mesh = voxel_shell([[0,0,0]], [[.6,.4,.2]], .2)
        points, faces, uv, png = palette_mesh(mesh)
        self.assertTrue((np.ptp(uv.reshape(-1,3,2), axis=1) == 0).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"voxel.glb"
            write_glb(path, points, faces, uv, png)
            self.assertEqual(inspect_glb(path)["faces"], 12)
            self.assertLess(path.stat().st_size, 10000)

    def test_chunking_keeps_palette_and_cubes(self):
        mesh = voxel_shell([[0,0,0], [50,0,0]], [[.1,.4,.2], [.5,.4,.3]], .2)
        chunks = list(split_chunks(mesh, max_faces=4, preserve_voxels=True))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(sum(len(m.triangles) for _,m in chunks), 24)
        self.assertTrue(all(m.has_vertex_colors() for _,m in chunks))

    def test_dynamic_exclusion_and_independent_frame_votes(self):
        k = np.array([[10.,0,2],[0,10,2],[0,0,1.]])
        rgb = np.full((4,4,3), 120, np.uint8)
        depth = np.full((4,4), 3., np.float32)
        valid = np.ones((4,4), bool)
        labels = np.ones((4,4), np.uint8)
        labels[:,2:] = 8  # person
        fusion = VoxelFusion(k, .2, stride=1)
        fusion.integrate(rgb,depth,valid,labels,np.eye(4),0)
        fusion.integrate(rgb,depth,valid,labels,np.eye(4),1)
        with self.assertRaisesRegex(ValueError, "multiple training views"):
            fusion.extract()
        rgb2 = rgb.copy(); rgb2[0,0] += 1
        fusion.integrate(rgb2,depth,valid,labels,np.eye(4),2)
        meshes = fusion.extract()
        self.assertEqual(set(meshes), {"ground"})
        self.assertEqual(fusion.report["duplicate_frame_aliases"], {1:0})

    def test_cell_budget_is_a_hard_gate(self):
        fusion = VoxelFusion(np.eye(3), .2, stride=1, max_cells=1)
        for i in range(2):
            fusion.integrate(np.full((2,2,3),i,np.uint8),np.ones((2,2),np.float32),
                np.ones((2,2),bool),np.ones((2,2),np.uint8),np.eye(4),i)
        with self.assertRaisesRegex(ValueError, "budget exceeded"):
            fusion.extract()

    def test_voxel_scene_contract_and_separate_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            mesh = voxel_shell([[-1,-1,16]],[[.4,.3,.2]],.5)
            rgb=np.full((64,64,3),120,np.uint8)
            depth=np.full((64,64),8.,np.float32)
            valid=np.ones((64,64),bool)
            labels=np.ones((64,64),np.uint8)
            rows=[dict(timestamp_seconds=i/60,original_tick=100+i) for i in range(3)]
            k=np.array([[50.,0,32],[0,50,32],[0,0,1.]])
            video=dict(width=64,height=64,fps="60")
            scene=make_scene(folder,{"ground":mesh},[0,1,2],lambda i:(rgb,depth,valid,labels),
                [np.eye(4)]*3,dict(schema="refined.calibration/1",status="passed",physical_contract_verified=False),
                rows,k,video,dict(poses=folder/"synthetic-poses.json"),style="voxel",voxel_size=.5)
            validate_scene(scene)
            self.assertEqual(len(scene["objects"]),1)
            obj=scene["objects"][0]
            self.assertEqual(obj["source_frames"],[0])
            self.assertEqual(obj["dimensions"],[.5,.5,.5])
            self.assertEqual(inspect_glb(folder/obj["mesh_asset"]["path"])["faces"],12)
            provenance=json.loads(next((folder/"assets").glob("*.provenance.json")).read_text())
            self.assertEqual(provenance["geometry_training_frames"],[0,1,2])
            self.assertEqual(provenance["appearance_source_frames"],[0])

    def test_background_voxel_palette_pixels_and_reopen(self):
        try: executable=blender_path()
        except ValueError: self.skipTest("Configure Blender for voxel integration")
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            scene=mesh_fixture(folder)  # explicitly labelled synthetic known test
            cells=np.array([(x,y,z) for x in range(-4,4) for y in range(-6,6) for z in range(-8,8)])
            mesh=voxel_shell(cells,np.tile([.6,.4,.2],(len(cells),1)),.25)
            points,faces,uv,png=palette_mesh(mesh)
            write_glb(folder/"mesh.glb",points,faces,uv,png)
            scene["objects"][0]["mesh_asset"]["sha256"]=sha256(folder/"mesh.glb")
            json_write(folder/"scene.json",scene)
            original=folder/"original.mp4"
            command([ffmpeg_path(),"-y","-nostdin","-f","lavfi","-i","testsrc=size=64x64:rate=60",
                "-frames:v","3","-c:v","libx264","-pix_fmt","yuv420p",str(original)])
            output=folder/"output"
            report=run_pipeline(folder/"scene.json",original,output,blender=executable,
                engine="CYCLES",samples=1,allow_inferred=True)
            self.assertTrue(report["scene_verification"]["reopened_saved_blend"])
            self.assertEqual(report["media"]["blockout_video.mp4"]["frame_count"],3)
            render_previews(output,[0],executable=executable)
            pixels=np.array(Image.open(output/"frames/frame_000001.png").convert("RGB"))
            # Exact flat face color after the front-face shade and quantization.
            np.testing.assert_allclose(pixels[30:34,30:34].mean((0,1)),[102,68,34],atol=3)


class AlignmentAndSafetyTests(unittest.TestCase):
    def test_z_buffer_handles_shift_and_occlusion(self):
        depth = np.array([[1.,2.,0.]],np.float32)
        valid = depth > 0
        target = np.eye(4); target[0,3] = 2.
        aligned, mask = align_depth(depth,valid,np.eye(4),target,np.eye(3))
        self.assertEqual(mask.tolist(), [[True,False,False]])
        self.assertEqual(aligned[0,0], 2.)
        # Two pixels collide under translation: retain the nearer surface.
        target[0,3] = -2.
        depth = np.array([[1.,2.,0.,0.]],np.float32)
        aligned, mask = align_depth(depth,depth>0,np.eye(4),target,np.eye(3))
        self.assertEqual(aligned[0,2], 1.)

    def test_sampling_never_clamps_offscreen_points(self):
        raw = np.full((8,8),200,np.uint16)
        codes, z = sample_depth_curve(np.array([[3.,3.,1.],[-4.,3.,1.],[99.,0.,1.]]),raw,np.eye(4),np.eye(3))
        self.assertEqual(codes.tolist(),[200])
        self.assertEqual(z.tolist(),[1.])

    def test_smooth_empirical_fit_labels_and_limits(self):
        codes = np.arange(100,400)
        depth = 2.5*np.expm1(codes*.008)
        curve = fit_smooth_curve(codes,depth)
        z, valid = decode_empirical(np.array([0,100,200,399,1023]),curve)
        np.testing.assert_allclose(z[1:4], 2.5*np.expm1(np.array([100,200,399])*.008),rtol=.001)
        self.assertEqual(valid.tolist(),[False,True,True,True,False])
        self.assertFalse(curve["physical_contract_verified"])
        self.assertFalse(curve["extrapolation"])

    def test_cache_verifies_exact_frame_set_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder/"frame_000000.npz"; path.write_bytes(b"test fixture")
            json_write(folder/"frames.json",dict(hashes={path.name:sha256(path)}))
            self.assertIsNotNone(verified_frame_cache(folder,[0]))
            self.assertIsNone(verified_frame_cache(folder,[0,1]))
            path.write_bytes(b"changed")
            self.assertIsNone(verified_frame_cache(folder,[0]))

    def test_failure_is_persisted_and_lock_released(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.assertRaises(RuntimeError):
                with refinement_lock(folder):
                    raise RuntimeError("redacted private value")
            report = json.loads((folder/"refinement_report.json").read_text())
            self.assertEqual(report["status"],"failed")
            self.assertFalse(report["full_video_rendered"])
            self.assertNotIn("redacted private",json.dumps(report))
            self.assertFalse((folder/".refined.lock").exists())

    def test_training_split_and_holdouts_are_isolated(self):
        rng = np.random.default_rng(4)
        k = np.array([[80.,0,80],[0,80,60],[0,0,1.]])
        uv0 = rng.uniform([25,20],[120,100],(400,2))
        rays = np.column_stack((uv0,np.ones(400)))@np.linalg.inv(k).T
        points = rays*(15/(1-.7*rays[:,0]-.5*rays[:,1]))[:,None]
        descriptors = rng.normal(size=(400,128)).astype(np.float32)
        poses = [np.eye(4) for _ in range(96)]
        for i,p in enumerate(poses): p[0,3] = i*.02
        def feature_loader(i):
            uv,z = project(points,poses[i],k)
            keep = (uv[:,0]>2)&(uv[:,0]<158)&(uv[:,1]>2)&(uv[:,1]<118)
            return uv[keep],descriptors[keep]
        def depth_loader(i):
            y,x = np.mgrid[:120,:160]
            denominator = 1-.7*(x-80)/80-.5*(y-60)/80
            z = (15+.7*poses[i][0,3])/np.maximum(denominator,.01)
            return np.rint(100*np.log(np.clip(z,.1,100))).astype(np.uint16)
        indices = sorted(set(range(0,96,12))|{95})
        with patch("reconstruction.refined_calibration.pose_track",return_value=poses):
            _,_,report = calibrate([{}]*96,100,k,indices,feature_loader,depth_loader)
        self.assertNotIn(60,report["training_frames"])
        self.assertNotIn(60,report["selection_frames"])
        self.assertEqual(report["status"],"passed")


if __name__ == "__main__": unittest.main()
