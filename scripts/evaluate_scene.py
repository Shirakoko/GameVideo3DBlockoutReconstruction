"""Stage G: evaluate reviewed local GT, or report existing dataset limitations."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.evaluator import evaluate, evaluate_existing_sample, read_json
from evaluation.report import new_report, write_reports
from evaluation.resources import ResourceMonitor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--sample", type=Path, default=ROOT/"outputs/stage_a/sample_66ad")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/stage_g/sample_66ad")
    args = parser.parse_args(argv)
    monitor = ResourceMonitor().start()
    code = 0
    try:
        report = evaluate(read_json(args.config), args.output, base=args.config.resolve().parent) if args.config else evaluate_existing_sample(args.sample, args.output)
        code = 0 if report["status"] == "evaluated" else 2
    except (ValueError, KeyError, OSError, TypeError) as error:
        report = new_report("Stage G input failed")
        report["status"] = "failed"
        report["failures"].append(dict(group="input", error=str(error)))
        code = 1
    finally:
        resource = monitor.finish()
    report["resources"] = resource
    write_reports(args.output, report)
    print(f"Stage G {report['status']}: {args.output.resolve()}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
