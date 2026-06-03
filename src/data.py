from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_SECONDARY_LABEL_WEIGHT,
    ROW_ID_COL,
    SAMPLE_SUBMISSION_PATH,
    TAXONOMY_CSV_PATH,
    TRAIN_AUDIO_DIR,
    TRAIN_CSV_PATH,
)


PRIMARY_LABEL_CANDIDATES = (
    "primary_label",
    "label",
    "class_id",
    "species_id",
    "taxon_id",
    "target",
)
SECONDARY_LABEL_CANDIDATES = ("secondary_labels", "secondary_label", "secondary", "labels")
FILENAME_CANDIDATES = ("filename", "file", "filepath", "path", "audio_path")


@dataclass(frozen=True)
class RowId:
    soundscape_id: str
    end_second: int


def load_train_metadata(path: str | Path = TRAIN_CSV_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_taxonomy(path: str | Path = TAXONOMY_CSV_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_sample_submission(path: str | Path = SAMPLE_SUBMISSION_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_class_names(sample_submission_path: str | Path = SAMPLE_SUBMISSION_PATH) -> list[str]:
    sample = load_sample_submission(sample_submission_path)
    if ROW_ID_COL not in sample.columns:
        raise ValueError(f"{sample_submission_path} must contain a '{ROW_ID_COL}' column.")
    class_names = [str(col) for col in sample.columns if col != ROW_ID_COL]
    if not class_names:
        raise ValueError("sample_submission.csv does not contain class columns.")
    return class_names


def find_primary_label_column(df: pd.DataFrame) -> str:
    for column in PRIMARY_LABEL_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(
        "Unable to locate a primary label column. "
        f"Expected one of: {', '.join(PRIMARY_LABEL_CANDIDATES)}."
    )


def find_secondary_label_column(df: pd.DataFrame) -> str | None:
    for column in SECONDARY_LABEL_CANDIDATES:
        if column in df.columns:
            return column
    return None


def find_filename_column(df: pd.DataFrame) -> str:
    for column in FILENAME_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(
        "Unable to locate an audio filename column. "
        f"Expected one of: {', '.join(FILENAME_CANDIDATES)}."
    )


def resolve_audio_path(row: pd.Series, train_audio_dir: str | Path = TRAIN_AUDIO_DIR) -> Path:
    filename_column = find_filename_column(row.to_frame().T)
    filename = str(row[filename_column])
    path = Path(filename)
    if path.is_absolute():
        return path
    candidate = Path(train_audio_dir) / path
    if candidate.exists() or len(path.parts) > 1:
        return candidate

    primary_col = None
    for column in PRIMARY_LABEL_CANDIDATES:
        if column in row.index:
            primary_col = column
            break
    if primary_col is not None:
        nested = Path(train_audio_dir) / str(row[primary_col]) / filename
        return nested
    return candidate


def parse_secondary_labels(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    text = str(value).strip()
    if not text or text in {"[]", "nan", "None"}:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(item) for item in parsed if str(item)]
        if parsed is None:
            return []
        return [str(parsed)]
    except (SyntaxError, ValueError):
        pass
    return [part for part in re.split(r"[,;\s]+", text) if part]


def build_multilabel_target(
    primary_label: object,
    secondary_labels: Iterable[object],
    class_to_index: dict[str, int],
    *,
    secondary_weight: float = DEFAULT_SECONDARY_LABEL_WEIGHT,
) -> np.ndarray:
    target = np.zeros(len(class_to_index), dtype=np.float32)
    primary = str(primary_label)
    if primary in class_to_index:
        target[class_to_index[primary]] = 1.0
    for label in secondary_labels:
        key = str(label)
        if key in class_to_index and target[class_to_index[key]] < 1.0:
            target[class_to_index[key]] = secondary_weight
    return target


def build_targets(
    metadata: pd.DataFrame,
    class_names: list[str],
    *,
    secondary_weight: float = DEFAULT_SECONDARY_LABEL_WEIGHT,
) -> np.ndarray:
    primary_col = find_primary_label_column(metadata)
    secondary_col = find_secondary_label_column(metadata)
    class_to_index = {label: idx for idx, label in enumerate(class_names)}
    targets = []
    for _, row in metadata.iterrows():
        secondary = parse_secondary_labels(row[secondary_col]) if secondary_col else []
        targets.append(
            build_multilabel_target(
                row[primary_col],
                secondary,
                class_to_index,
                secondary_weight=secondary_weight,
            )
        )
    return np.vstack(targets) if targets else np.zeros((0, len(class_names)), dtype=np.float32)


def assign_folds(
    metadata: pd.DataFrame,
    *,
    n_splits: int,
    seed: int,
    primary_col: str | None = None,
) -> np.ndarray:
    if len(metadata) == 0:
        return np.array([], dtype=np.int64)
    if n_splits <= 1:
        return np.zeros(len(metadata), dtype=np.int64)
    primary_col = primary_col or find_primary_label_column(metadata)
    labels = metadata[primary_col].astype(str).to_numpy()
    counts = pd.Series(labels).value_counts()
    folds = np.full(len(metadata), -1, dtype=np.int64)
    if len(counts) >= n_splits and int(counts.min()) >= n_splits:
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (_, val_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
            folds[val_idx] = fold
        return folds

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(metadata))
    for position, index in enumerate(indices):
        folds[index] = position % n_splits
    return folds


def parse_row_id(row_id: str) -> RowId:
    match = re.match(r"^(?P<soundscape>.+)_(?P<end>\d+)$", str(row_id))
    if not match:
        raise ValueError(f"Invalid row_id format: {row_id}")
    return RowId(soundscape_id=match.group("soundscape"), end_second=int(match.group("end")))


def make_row_id(soundscape_id: str, end_second: int | float) -> str:
    end_value = int(round(float(end_second)))
    return f"{soundscape_id}_{end_value}"
