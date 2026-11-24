"""JSON configuration and explicit local source discovery, without downloads."""
from copy import deepcopy
from pathlib import Path

from evaluation.evaluator import read_json
from pipeline.runner import ROOT

PATH_KEYS = {"input_manifest", "original_video", "camera_poses", "annotations", "local_detector", "local_sam", "blender",
             "prepared_sample", "sample_source", "metric_manifest", "point_cloud", "scene_json", "streams",
             "lpips_backbone_weights", "lpips_linear_weights", "predicted", "reference"}


def resolve_settings(document, base):
    settings = deepcopy(document)
    allowed = {"schema", "mode", "output", "cache", "resume", "retries", "timeout_seconds", "scene_id", "synthetic", "reconstruction", "render", "evaluation",
               "input_manifest", "original_video", "camera_poses", "annotations", "local_detector", "local_sam", "blender", "prepared_sample", "sample_source", "prepare_raw_sample", "video_start_frame", "video_frame_count"}
    if settings.get("schema") != "pipeline.config/1" or set(settings)-allowed:
        raise ValueError("pipeline.config/1 required, with no unknown top-level options")
    if settings.get("mode") not in ("synthetic", "reviewed", "existing_sample"):
        raise ValueError("Choose explicit synthetic, reviewed or existing_sample mode; no fallback")
    def resolve(value):
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, dict):
            return {key: str((base/item).resolve()) if key in PATH_KEYS and isinstance(item, str) else resolve(item) for key, item in value.items()}
        return value
    settings = resolve(settings)
    for key in ("cache", "resume", "prepare_raw_sample"):
        if not isinstance(settings.get(key, key != "prepare_raw_sample"), bool):
            raise ValueError(f"{key} must be boolean")
    if settings.get("prepare_raw_sample") and settings["mode"] != "existing_sample":
        raise ValueError("prepare_raw_sample is only for the existing local sample adapter")
    render = settings.get("render", {})
    if not isinstance(render, dict) or set(render)-{"enabled", "generate_only", "resolution", "engine", "samples"}:
        raise ValueError("Unknown render options")
    for key in ("enabled", "generate_only"):
        if not isinstance(render.get(key, key == "enabled"), bool):
            raise ValueError(f"render.{key} must be boolean")
    if settings["mode"] == "reviewed":
        if not settings.get("input_manifest") or not settings.get("original_video"):
            raise ValueError("Reviewed mode requires a metric input manifest and original_video")
        if bool(settings.get("annotations")) == bool(settings.get("local_detector")):
            raise ValueError("Reviewed mode requires annotations OR local_detector")
    if settings["mode"] == "existing_sample":
        if not settings.get("prepare_raw_sample"):
            settings.setdefault("prepared_sample", str(ROOT/"outputs/stage_a/sample_66ad"))
        settings.setdefault("sample_source", str(ROOT/"data/input/samples/66ad9535-583f-4a25-b655-6b1697c83a40"))
    else:
        reconstruction = settings.get("reconstruction", {})
        if "min_depth_m" not in reconstruction or "max_depth_m" not in reconstruction:
            raise ValueError("Explicit min_depth_m/max_depth_m required; no assumed real clip planes")
        if set(reconstruction)-{"min_depth_m", "max_depth_m", "voxel_size_m", "stride"}:
            raise ValueError("Unknown reconstruction options")
    settings["output"] = str((base/settings.get("output", "../outputs/pipeline/run")).resolve())
    return settings


def source_inputs(settings):
    sources = set()
    def collect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in PATH_KEYS and isinstance(item, str):
                    sources.add(Path(item).resolve())
                else:
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
    collect(settings)
    if settings.get("render", {}).get("enabled", True) and settings["mode"] != "existing_sample":
        try:
            from blender.pipeline import blender_path
            sources.add(Path(blender_path(settings.get("blender"))).resolve())
        except ValueError:
            pass  # Stage E records missing Blender; A-D can still make progress.
    from stage_a.media import ffmpeg_path
    sources.add(Path(ffmpeg_path()).resolve())
    # Resolve indirect local assets in reviewed manifests and annotations too.
    expanded = set(sources)
    for source in sources:
        if source.is_file() and source.suffix.lower() == ".json":
            document = read_json(source)
            def references(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key in {"data_npz", "point_cloud_npz", "rgb_path", "path"} and isinstance(item, str):
                            expanded.add((source.parent/item).resolve())
                        else:
                            references(item)
                elif isinstance(value, list):
                    for item in value:
                        references(item)
            references(document)
    output = Path(settings["output"])
    if any(source == output or output.is_relative_to(source) or source.is_relative_to(output) for source in expanded):
        raise ValueError("Pipeline output must not overlap any source input; choose a dedicated sibling directory")
    return sorted(map(str, expanded))


def make_steps(settings):
    inputs = source_inputs(settings)
    render = settings.get("render", {})
    ids = list("ABCD")
    if render.get("enabled", True):
        ids.append("E")
        if not render.get("generate_only", False):
            ids.append("F")
    ids.append("G")
    dependencies = {"A": [], "B": ["A"], "C": ["A", "B"], "D": ["C", "A"], "E": ["A", "B", "C", "D"], "F": ["A", "E"]}
    expected = dict(A=["prepared.json"], B=["scene_pointcloud.ply", "reconstruction_metadata.json"], C=["scene_schema.json", "validation_report.json"],
        D=["schema_validation.json"], E=["scene.blend", "build_plan.json", "pipeline_report.json"],
        F=["blockout_video.mp4", "original_vs_blockout.mp4", "camera_pose_evaluation.json", "render_report.json"], G=["evaluation_report.json", "evaluation_report.html", "metrics.csv"])
    return [dict(id=identity, depends_on=dependencies.get(identity, [key for key in ids if key != "G"]),
        command=["{python}", "{root}/scripts/pipeline_stage.py", identity, "--context", "{context}", "--output", "{output}"],
        inputs=inputs, outputs=expected[identity], timeout_seconds=settings.get("timeout_seconds", 3600), retries=settings.get("retries", 1),
        retry_exit_codes=[1, 124], always_run=identity == "G") for identity in ids]
