"""Probe access and inventory without downloading any restricted file contents."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage_a.access import Client, InspectionError, credential, inspect_dataset, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/stage_a"))
    args = parser.parse_args()
    token, source = credential()
    try:
        report, card = inspect_dataset(Client(token), source)
        write_report(report, args.output, card)
    except (InspectionError, OSError, ValueError) as error:
        # Never stringify request objects, signed redirect URLs, headers or token.
        print(json.dumps({"status": "inspection_failed", "error_type": type(error).__name__}))
        return 1
    print(json.dumps({key: report[key] for key in ("status", "revision", "provisional_candidate", "dataset_payload_bytes_downloaded", "stage_a_complete")}, indent=2))
    return 0 if report["status"] == "metadata_accessible_not_validated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
