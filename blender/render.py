"""Reopen scene.blend, verify it, then render a complete PNG frame sequence."""
from pathlib import Path
import json
import sys

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blender.verification import verify_scene


def main():
    args = sys.argv[sys.argv.index("--")+1:]
    plan_path, workspace = Path(args[0]), Path(args[1])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    report = verify_scene(plan)
    report["reopened_saved_blend"] = True
    (workspace/"saved_scene_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if "--verify-only" not in args:
        (workspace/"frames").mkdir(exist_ok=True)
        scene = bpy.context.scene
        # Blend stores //frames; absolute render path avoids any cwd dependence.
        scene.render.filepath = str((workspace/"frames/frame_######").resolve())
        bpy.ops.render.render(animation=True)
        actual = {p.name for p in (workspace/"frames").glob("*.png")}
        expected = {f"frame_{i:06d}.png" for i in range(1, plan["frame_end"]+1)}
        if actual != expected:
            raise ValueError("Background render produced missing/extra frames")


if __name__ == "__main__":
    main()
