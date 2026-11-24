"""Internal subprocess entrypoint; each stage writes only its attempt artifacts."""
import argparse
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.evaluator import read_json
from evaluation.report import write_json
from pipeline.stages import execute_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=list("ABCDEFG"))
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        context = read_json(args.context)
        return execute_stage(args.stage, context["settings"], context["stages"], args.output)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output/"failure.json", dict(stage=args.stage, type=type(error).__name__, error=str(error)))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
