import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import subprocess

import numpy as np

from blender.pipeline import ROOT, blender_path, json_write, run_pipeline, sha256
from blender.planner import prepare_plan, source_scene_snapshot
from reconstruction.heuristic_proxy import AUTHORIZATION, exclude_camera_corridor, proxy_lut, solid_columns
from scene_schema import validate_scene
from scene_schema import SceneValidationError
from tests.test_refined import mesh_fixture


class ProxyTests(unittest.TestCase):
    def test_proxy_lut_is_monotone_and_explicitly_not_physical(self):
        lut,receipt=proxy_lut(np.arange(100,900),np.exp(np.arange(100,900)/250))
        self.assertTrue((np.diff(lut)>=0).all())
        self.assertGreaterEqual(lut.min(),2.5)
        self.assertLessEqual(lut.max(),70)
        self.assertFalse(receipt["physical_inverse_used"])
        self.assertFalse(receipt["geometry_accuracy_validated"])

    def test_all_camera_poses_get_clearance_not_only_preview_frames(self):
        poses=np.tile(np.eye(4),(3,1,1)); poses[1,:3,3]=[10.,0.,0.]
        cells=np.array([[0,0,0],[10,0,0],[20,0,0],[0,0,-3]])
        keep=exclude_camera_corridor(cells,np.array([2,2,2,1]),1.,poses)
        self.assertEqual(keep.tolist(),[False,False,True,True])

    def test_rock_columns_are_grounded_canopies_not_filled_to_floor(self):
        cells=np.array([[0,0,3],[0,0,5],[2,2,4],[2,2,6]])
        result,_,categories=solid_columns(cells,np.ones((4,3)),np.array([2,2,3,3]),0)
        self.assertEqual(result[categories==2,2].tolist(),[1,2,3,4,5])
        self.assertEqual(result[categories==3,2].tolist(),[4,5,6])

    def test_proxy_requires_separate_authorization_receipt_and_camera_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory); scene=mesh_fixture(folder)
            obj=scene["objects"][0]
            obj["geometry_source"].update(kind="heuristic_voxel",bounds_type="inferred_proxy_aabb")
            receipt=dict(schema="heuristic.proxy/1",status="passed",authorization=AUTHORIZATION,
                geometry_accuracy_validated=False,camera_inside_frame_count=0,coverage_status="passed",
                camera_sha256=hashlib.sha256(json.dumps(scene["cameras"],sort_keys=True).encode()).hexdigest())
            json_write(folder/"proxy.json",receipt)
            scene["reconstruction_metadata"]["heuristic_proxy"]=dict(path="proxy.json",sha256=sha256(folder/"proxy.json"))
            validate_scene(scene); json_write(folder/"scene.json",scene)
            video=dict(fps="60",frame_count=3,start_frame=0,first_pts_seconds=0,width=64,height=64)
            with self.assertRaisesRegex(ValueError,"allow-heuristic"):
                prepare_plan(folder/"scene.json",video,engine="CYCLES",allow_inferred=True)
            plan=prepare_plan(folder/"scene.json",video,engine="CYCLES",allow_inferred=True,allow_heuristic=True)
            self.assertIn("NOT measured",plan["reference_evidence"])
            snapshot=source_scene_snapshot(scene,folder/"scene.json")
            self.assertTrue(Path(snapshot["reconstruction_metadata"]["heuristic_proxy"]["path"]).is_absolute())
            scene["cameras"][0]["track"][0]["world_from_camera"][0][3]+=1
            json_write(folder/"scene.json",scene)
            with self.assertRaisesRegex(ValueError,"camera changed"):
                prepare_plan(folder/"scene.json",video,engine="CYCLES",allow_inferred=True,allow_heuristic=True)

    def test_only_explicit_proxy_completion_can_have_no_source_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            scene=mesh_fixture(Path(directory)); obj=scene["objects"][0]
            obj.update(source_frames=[],observations=[],visibility=[])
            with self.assertRaises(SceneValidationError): validate_scene(scene)
            obj["geometry_source"].update(kind="heuristic_voxel",bounds_type="inferred_proxy_aabb")
            scene["reconstruction_metadata"]["heuristic_proxy"]=dict(path="proxy.json",sha256="0"*64)
            validate_scene(scene)

    def test_background_static_driver_is_rejected_by_optimized_verification(self):
        from blender.media import command
        from stage_a.media import ffmpeg_path
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory); mesh_fixture(folder)
            original=folder/"original.mp4"
            command([ffmpeg_path(),"-y","-nostdin","-f","lavfi","-i","testsrc=size=64x64:rate=60",
                "-frames:v","3","-c:v","libx264","-pix_fmt","yuv420p",str(original)])
            output=folder/"output"; executable=blender_path()
            run_pipeline(folder/"scene.json",original,output,blender=executable,engine="CYCLES",samples=1,
                allow_inferred=True,generate_only=True)
            expr="\n".join(["import bpy,json,sys",f"sys.path.insert(0,{str(ROOT)!r})",
                "from blender.verification import verify_scene",f"plan=json.load(open({str(output/'build_plan.json')!r}))",
                "obj=next(o for o in bpy.context.scene.objects if o.get('source_object_id'))",
                "obj.driver_add('location',0).driver.expression='0'", "try:", "    verify_scene(plan)",
                "except ValueError as error:", "    assert 'unexpected animation/deformation' in str(error)",
                "    print('STATIC_DRIVER_REJECTED')", "else:", "    raise RuntimeError('Static driver was not rejected')"])
            result=subprocess.run([executable,"--background","--factory-startup",str(output/"scene.blend"),
                "--python-exit-code","1","--python-expr",expr],capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("STATIC_DRIVER_REJECTED",result.stdout)


if __name__=="__main__": unittest.main()
