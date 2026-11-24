import importlib.util
import unittest
from fractions import Fraction
from pathlib import Path


spec = importlib.util.spec_from_file_location("h3_restyle", Path(__file__).resolve().parents[1] / "scripts/run_h3_restyle.py")
restyle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restyle)


class H3RestyleTests(unittest.TestCase):
    def setUp(self):
        self.config = {"turbo": True, "prompt": restyle.PROMPT, "width": 768, "height": 432,
                       "length": 124, "control": "control.mp4", "strength": .85,
                       "steps": 8, "seed": 1}

    def test_native_control_and_source_frame_count(self):
        graph = restyle.graph(self.config, "test")
        self.assertEqual(graph["6"]["inputs"]["length"], 124)
        self.assertEqual(graph["10"]["inputs"]["control_video"], ["8", 0])

    def test_guide_padding_does_not_change_target_canvas(self):
        self.config["guide"] = "guide.mp4"
        graph = restyle.graph(self.config, "test")
        self.assertEqual(graph["6"]["inputs"]["height"],432)
        self.assertEqual(graph["22"]["inputs"]["height"],448)
        self.assertEqual(graph["23"]["inputs"]["bottom"],16)
        self.assertEqual(graph["23"]["inputs"]["top"],0)
        self.assertEqual(graph["21"]["inputs"]["latent"],["22",0])
        self.assertEqual(graph["15"]["inputs"]["latent_image"],["6",1])
        self.assertEqual(graph["17"]["inputs"]["fps"], 60)
        self.assertEqual(graph["14"]["inputs"]["model"], ["10", 0])
        self.assertNotIn("first_frame", graph["6"]["inputs"])
        self.assertFalse(any("Aligned" in node["class_type"] for node in graph.values()))

    def test_no_turbo_uses_base_directly(self):
        self.config["turbo"] = False
        graph = restyle.graph(self.config, "test")
        self.assertNotIn("2", graph)
        self.assertEqual(graph["3"]["inputs"]["model"], ["1", 0])

    def test_overlap_guide_is_conditioning_not_control(self):
        self.config["guide"] = "guide.mp4"
        graph = restyle.graph(self.config, "test")
        self.assertEqual(graph["21"]["inputs"]["frame_idx"], 0)
        self.assertEqual(graph["14"]["inputs"]["conditioning"], ["21", 0])
        self.assertEqual(graph["10"]["inputs"]["control_video"], ["8", 0])

    def test_all_720_indices_without_padding_or_overlap_duplicates(self):
        configs = []
        for start in (0,238,476):
            items = [{"source_index":min(start+i,719), "source_pts_seconds":str(Fraction(min(start+i,719),60)),
                      "generated_index":i,"padding":start+i>=720} for i in range(260)]
            configs.append({"mapping":items,"prompt_id":str(start)})
        selected = restyle.selected_mapping(configs)
        self.assertEqual([item["source_index"] for item in selected],list(range(720)))
        self.assertEqual(selected[-1]["generated_index"],243)
        self.assertEqual(selected[260]["generated_index"],22)
        self.assertFalse(any(item["padding"] for item in selected))

    def test_source_gap_is_rejected(self):
        with self.assertRaises(ValueError):
            restyle.selected_mapping([{"mapping":[{"source_index":1,"source_pts_seconds":"1/60","padding":False}],"prompt_id":"test"}])

    def test_rational_fps_representation_is_compared_numerically(self):
        probe={"frames":3,"fps":"60/1","duration_seconds":"1/20","pts_seconds":["0","1/60","1/30"]}
        self.assertTrue(restyle.exact_timeline(probe,3))
        probe["fps"]="24"
        self.assertFalse(restyle.exact_timeline(probe,3))


if __name__ == "__main__":
    unittest.main()
