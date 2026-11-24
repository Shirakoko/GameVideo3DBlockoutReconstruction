"""One command for Stage A-G: config, verified cache, resume, bounded retries."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.evaluator import read_json
from evaluation.report import attach_stages, new_report, write_json, write_reports
from pipeline.config import make_steps, resolve_settings
from pipeline.runner import run_steps


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = resolve_settings(read_json(args.config), args.config.resolve().parent)
        if args.output:
            settings["output"] = str(args.output.resolve())
        output = Path(settings["output"])
        result = run_steps(make_steps(settings), output, settings=settings,
            cache=settings.get("cache", True) and not args.no_cache, resume=settings.get("resume", True) and not args.no_resume)
        stage_g = next(s for s in result["stages"] if s["id"] == "G")
        report = read_json(Path(stage_g["output_directory"])/"evaluation_report.json") if stage_g.get("output_directory") else new_report("Pipeline evaluation unavailable")
        report["pipeline_status"] = result["status"]
        report["artifacts"] = {}
        for stage in result["stages"]:
            if stage.get("output_directory"):
                for name in ("scene.blend", "scene_schema.json", "scene_pointcloud.ply", "original_video.mp4", "blockout_video.mp4", "original_vs_blockout.mp4"):
                    target = Path(stage["output_directory"])/name
                    if target.is_file():
                        report["artifacts"][stage["id"]+"/"+name] = str(target)
        write_json(output/"artifact_manifest.json", report["artifacts"])
        for stage in result["stages"]:
            for attempt in stage["attempts"]:
                if attempt["status"] not in ("completed",):
                    report["failures"].append(dict(stage=stage["id"], attempt=attempt["number"], error=attempt.get("error", ""), log=attempt.get("log")))
            if stage["status"] not in ("completed", "cached"):
                report["failures"].append(dict(stage=stage["id"], status=stage["status"], error=stage.get("error", "")))
        attach_stages(report, result["stages"])
        write_reports(output, report)
        print(f"Pipeline {result['status']}; evaluation {report['status']}: {output}")
        return 0 if result["status"] == "completed" else 2 if result["status"] == "blocked" else 1
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"Pipeline configuration failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
