from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import (
    assign_folds,
    build_targets,
    load_class_names,
    make_row_id,
    parse_row_id,
    parse_secondary_labels,
    resolve_audio_path,
)


def test_load_class_names_preserves_sample_submission_order(tmp_path: Path) -> None:
    path = tmp_path / "sample_submission.csv"
    pd.DataFrame({"row_id": ["sound_5"], "b": [0.0], "a": [0.0]}).to_csv(path, index=False)

    assert load_class_names(path) == ["b", "a"]


def test_parse_row_id_uses_last_underscore_as_end_second() -> None:
    parsed = parse_row_id("site_123_15")

    assert parsed.soundscape_id == "site_123"
    assert parsed.end_second == 15
    assert make_row_id(parsed.soundscape_id, parsed.end_second) == "site_123_15"


def test_parse_secondary_labels_accepts_kaggle_list_strings() -> None:
    assert parse_secondary_labels("['a', 'b']") == ["a", "b"]
    assert parse_secondary_labels("a b") == ["a", "b"]
    assert parse_secondary_labels("[]") == []


def test_build_targets_sets_primary_and_weighted_secondary_labels() -> None:
    metadata = pd.DataFrame(
        {
            "primary_label": ["a", "b"],
            "secondary_labels": ["['b']", "['a', 'missing']"],
            "filename": ["x.ogg", "y.ogg"],
        }
    )

    targets = build_targets(metadata, ["a", "b"], secondary_weight=0.25)

    np.testing.assert_allclose(targets, np.array([[1.0, 0.25], [0.25, 1.0]], dtype=np.float32))


def test_resolve_audio_path_prefers_primary_label_subdirectory() -> None:
    row = pd.Series({"primary_label": "123", "filename": "clip.ogg"})

    assert resolve_audio_path(row, "train_audio") == Path("train_audio") / "123" / "clip.ogg"


def test_assign_folds_is_stable_when_stratification_is_not_possible() -> None:
    metadata = pd.DataFrame({"primary_label": ["a", "a", "b"], "filename": ["1", "2", "3"]})

    folds_a = assign_folds(metadata, n_splits=5, seed=42)
    folds_b = assign_folds(metadata, n_splits=5, seed=42)

    np.testing.assert_array_equal(folds_a, folds_b)
    assert set(folds_a.tolist()) <= set(range(5))

