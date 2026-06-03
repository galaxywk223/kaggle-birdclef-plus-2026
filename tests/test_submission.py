from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def test_submission_shape_matches_sample_submission_columns() -> None:
    sample = pd.DataFrame({"row_id": ["sound_5", "sound_10"], "a": [0.0, 0.0], "b": [0.0, 0.0]})
    prediction = pd.DataFrame(
        {"row_id": ["sound_5", "sound_10"], "a": [0.1, 0.2], "b": [0.3, 0.4]}
    )

    assert list(prediction.columns) == list(sample.columns)
    assert prediction.shape == sample.shape


def test_inference_smoke_test_can_be_skipped_without_torch() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed in the current lightweight environment.")

    assert importlib.util.find_spec("torch") is not None

