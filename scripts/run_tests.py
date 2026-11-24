"""Run small synthetic tests and persist honest results, including failures."""
import argparse
import io
import json
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/stage_a/tests")
    parser.add_argument("--pattern", default="test_*.py", help="unittest file pattern, e.g. test_stage_b.py")
    args = parser.parse_args()
    log = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=args.pattern)
    start = time.perf_counter()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    summary = {"data_source": "SYNTHETIC ONLY — NOT HUGGING FACE", "tests_run": result.testsRun,
               "passed": result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
               "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
               "successful": result.wasSuccessful(), "seconds": time.perf_counter()-start,
               "real_session_validated": False, "test_pattern": args.pattern}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "test_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "test_log.txt").write_text(log.getvalue(), encoding="utf-8")
    print(log.getvalue())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
