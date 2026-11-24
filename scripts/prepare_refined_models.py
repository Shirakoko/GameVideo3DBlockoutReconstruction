"""Download only the explicitly selected public models, never session media."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blender.pipeline import json_write, sha256

MODELS = {
    "detector": ("IDEA-Research/grounding-dino-tiny", "a2bb814dd30d776dcf7e30523b00659f4f141c71"),
    "sam": ("facebook/sam-vit-base", "70c1a07f894ebb5b307fd9eaaee97b9dfc16068f"),
}


def prepare(output):
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    output = Path(output).resolve()
    records = {}
    for name, (repo, revision) in MODELS.items():
        info = api.model_info(repo, revision=revision)
        license_name = (info.card_data.to_dict() if info.card_data else {}).get("license")
        if license_name != "apache-2.0" or info.sha != revision:
            raise ValueError("Model revision/license differs from the reviewed Apache-2.0 selection")
        directory = output/name
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Preparing public local {name}: {repo}@{revision}", flush=True)
        snapshot_download(repo, revision=revision, local_dir=directory,
                          allow_patterns=["*.json", "*.txt", "*.safetensors", "README.md", "LICENSE*"],
                          max_workers=2)
        files = {str(p.relative_to(directory)): sha256(p) for p in directory.iterdir() if p.is_file()}
        if "model.safetensors" not in files:
            raise ValueError("Selected model has no safetensors weights")
        records[name] = dict(repo_id=repo, revision=revision, license=license_name,
                             path=str(directory), files=files)
    json_write(output/"models.lock.json", dict(schema="refined.models/1", models=records,
                remote_video_upload=False, dataset_downloaded=False))
    print("Model bytes and immutable revisions recorded", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT/"data/models/refined")
    args = p.parse_args()
    try:
        prepare(args.output)
        return 0
    except (OSError, ValueError, ImportError) as error:
        print(f"Local model preparation unavailable: {type(error).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
