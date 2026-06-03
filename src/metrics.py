from __future__ import annotations

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))).astype(np.float32)


def birdclef_macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, object]:
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true) > 0
    y_score = np.asarray(y_score)
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape.")
    scores: list[float] = []
    skipped: list[int] = []
    for idx in range(y_true.shape[1]):
        class_truth = y_true[:, idx]
        if np.unique(class_truth).size < 2:
            skipped.append(idx)
            continue
        scores.append(float(roc_auc_score(class_truth, y_score[:, idx])))
    macro_auc = float(np.mean(scores)) if scores else float("nan")
    return {
        "macro_auc": macro_auc,
        "class_auc": scores,
        "skipped_class_indices": skipped,
        "evaluated_classes": len(scores),
    }
