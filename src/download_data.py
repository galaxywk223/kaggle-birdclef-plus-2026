from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

from src.config import KAGGLE_COMPETITION_SLUG, RAW_DATA_DIR, ensure_project_dirs


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "kaggle",
        "competitions",
        "download",
        "-c",
        args.competition,
        "-p",
        str(output_dir),
    ]
    if args.force:
        command.append("--force")
    subprocess.run(command, check=True)

    zip_path = output_dir / f"{args.competition}.zip"
    if args.unzip and zip_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(output_dir)
        print(f"[extracted] {zip_path} -> {output_dir}")
    print(f"[downloaded] competition={args.competition} output_dir={output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BirdCLEF+ 2026 data.")
    parser.add_argument("--competition", default=KAGGLE_COMPETITION_SLUG)
    parser.add_argument("--output-dir", default=str(RAW_DATA_DIR))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-unzip", dest="unzip", action="store_false")
    parser.set_defaults(unzip=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()

