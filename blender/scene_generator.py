"""Execute via blender --background --factory-startup --python, never UI."""
from pathlib import Path
import json
import sys

import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blender.animation_generator import animate_cameras, animate_objects
from blender.camera_generator import create_camera
from blender.object_generator import create_object, material
from blender.verification import verify_scene


def generate(plan, workspace):
    workspace = Path(workspace)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = plan["scene_id"]
    scene.unit_settings.system, scene.unit_settings.scale_length = "METRIC", 1.
    scene.render.resolution_x, scene.render.resolution_y = plan["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x, scene.render.pixel_aspect_y = plan["pixel_aspect"]
    scene.render.fps, scene.render.fps_base = plan["fps_integer"], plan["fps_base"]
    scene.frame_start, scene.frame_end = plan["frame_start"], plan["frame_end"]
    scene.render.engine = plan["engine"]
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.render.use_motion_blur = False
    scene.render.filepath = "//frames/frame_######"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure, scene.view_settings.gamma = 0., 1.
    if plan["engine"] == "CYCLES":
        scene.cycles.device, scene.cycles.samples, scene.cycles.seed = "CPU", plan["samples"], 0
        scene.cycles.use_denoising = True
    elif plan["engine"] == "BLENDER_EEVEE" and hasattr(scene, "eevee"):
        if hasattr(scene.eevee, "taa_render_samples"):
            scene.eevee.taa_render_samples = plan["samples"]
    else:
        shade = scene.display.shading
        shade.light, shade.color_type = "STUDIO", "MATERIAL"
        shade.show_shadows, shade.show_cavity, shade.show_specular_highlight = True, True, False
        shade.background_type, shade.background_color = "WORLD", (0.12, 0.14, 0.17)
        scene.display.render_aa = "8"
    scene.world = bpy.data.worlds.new("BlockoutWorld")
    scene.world.color = (0.12, 0.14, 0.17)
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.12, 0.14, 0.17, 1)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = .7
    if plan.get("appearance"):
        color = plan["appearance"]["background_color"]
        linear = [v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4 for v in color]
        scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (*linear, 1)
        clay_world = plan["appearance"].get("texture_mode") == "clay_primitive"
        scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.55 if clay_world else 1
    collections = {}
    for name in ["Static", "Dynamic", "UnknownMotion", "Cameras", "Reference", "Lighting", "Hierarchy"]:
        c = bpy.data.collections.new(name)
        scene.collection.children.link(c)
        collections[name] = c
    clay = (plan.get("appearance") or {}).get("texture_mode") == "clay_primitive"
    mats = {category: material(category, clay=clay) for category in sorted({o["category"] for o in plan["objects"]})}
    objects = {}
    for item in plan["objects"]:
        c = collections["Static" if item["is_static"] is True else "Dynamic" if item["is_static"] is False else "UnknownMotion"]
        objects[item["object_id"]] = create_object(item, c, mats[item["category"]])
    # JSON parents are organizational, not relative-transform overrides. Nested
    # collections express the hierarchy without double-transforming children.
    groups = {}
    for item in plan["objects"]:
        c = bpy.data.collections.new("Group_"+item["object_id"])
        groups[item["object_id"]] = c
        c.objects.link(objects[item["object_id"]])
    for item in plan["objects"]:
        parent = groups[item["parent_id"]] if item["parent_id"] else collections["Hierarchy"]
        parent.children.link(groups[item["object_id"]])
    reference_mesh = bpy.data.meshes.new("ReferencePoints")
    reference_mesh.from_pydata(plan["reference_points"], [], [])
    reference_mesh.update()
    reference = bpy.data.objects.new("PointCloudReference", reference_mesh)
    collections["Reference"].objects.link(reference)
    reference.hide_render, reference.display_type = True, "WIRE"
    reference["evidence"] = plan.get("reference_evidence", "Measured point reference; NOT complete geometry; may include temporal object samples")
    clay = (plan.get("appearance") or {}).get("texture_mode") == "clay_primitive"
    if clay:
        lights = [("Sun", "SUN", 4.5, (0.85, 0.2, 0.45)), ("Fill", "SUN", 2.6, (-0.55, -0.8, 0.2))]
    else:
        lights = [("Sun", "SUN", 2., (0.4, -0.5, -0.3)), ("Fill", "AREA", 700., (0.4, -0.5, -0.3))]
    for name, kind, energy, rotation in lights:
        data = bpy.data.lights.new(name, kind)
        data.energy = energy
        obj = bpy.data.objects.new(name, data)
        collections["Lighting"].objects.link(obj)
        obj.rotation_euler = rotation
        if kind == "AREA":
            obj.location = (4, -4, 10)
            data.size = 8
    cameras = {c["camera_id"]: create_camera(c, collections["Cameras"], scene) for c in plan["cameras"]}
    scene.camera = cameras[plan["active_camera"]]
    animate_objects(plan["objects"], objects)
    animate_cameras(plan["cameras"], cameras, scene)
    for name, value in [("SourceScene.json", plan["source_scene_json"]), ("FrameMapping.json", plan["frame_mapping"]), ("Conventions.json", plan["conventions"])]:
        bpy.data.texts.new(name).write(json.dumps(value, ensure_ascii=False, indent=2))
    scene["source_scene"], scene["source_fps_rational"] = plan["source_scene"], plan["fps"]
    scene["data_source"] = plan["data_source"]
    report = verify_scene(plan)
    scene.frame_set(plan["frame_start"])
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str((workspace/"scene.blend").resolve()), check_existing=False, compress=False)
    (workspace/"build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    args = sys.argv[sys.argv.index("--")+1:]
    plan = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    generate(plan, Path(args[1]))


if __name__ == "__main__":
    main()
