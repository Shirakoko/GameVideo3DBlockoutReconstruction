"""Validate a Stage D scene, returning exit 1 with actionable diagnostics."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scene_schema import validate_scene


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    args = parser.parse_args(argv)
    try:
        result = validate_scene(json.loads(args.scene.read_text(encoding="utf-8")))
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError) as error:
        print(f"Scene validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
