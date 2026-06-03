from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_DIR
from src.data import find_primary_label_column, load_class_names


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.csv"
    sample_path = data_dir / "sample_submission.csv"
    taxonomy_path = data_dir / "taxonomy.csv"

    train_df = pd.read_csv(train_path)
    class_names = load_class_names(sample_path)
    primary_col = find_primary_label_column(train_df)

    print(f"train_rows={len(train_df)}")
    print(f"sample_classes={len(class_names)}")
    print(f"primary_label_column={primary_col}")
    print("top_primary_labels:")
    print(train_df[primary_col].value_counts().head(20).to_string())
    if taxonomy_path.exists():
        taxonomy = pd.read_csv(taxonomy_path)
        print(f"taxonomy_rows={len(taxonomy)}")
        print(f"taxonomy_columns={list(taxonomy.columns)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print BirdCLEF+ 2026 data overview.")
    parser.add_argument("--data-dir", default=str(RAW_DATA_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    main()

