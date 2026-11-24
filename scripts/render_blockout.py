"""Build Scene JSON and render Stage E/F artifacts with no Blender UI."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blender.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
