from __future__ import annotations

import numpy as np

from src.metrics import birdclef_macro_auc, sigmoid


def test_sigmoid_is_stable_for_large_logits() -> None:
    result = sigmoid(np.array([-1000.0, 0.0, 1000.0]))

    assert result[0] < 1e-6
    assert result[1] == 0.5
    assert result[2] > 1.0 - 1e-6


def test_birdclef_macro_auc_skips_classes_without_both_labels() -> None:
    y_true = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    y_score = np.array(
        [
            [0.9, 0.1, 0.1],
            [0.2, 0.2, 0.8],
            [0.8, 0.3, 0.2],
            [0.1, 0.4, 0.9],
        ]
    )

    metrics = birdclef_macro_auc(y_true, y_score)

    assert metrics["evaluated_classes"] == 2
    assert metrics["skipped_class_indices"] == [1]
    assert metrics["macro_auc"] == 1.0

