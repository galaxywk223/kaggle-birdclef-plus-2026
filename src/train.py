from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_NAME,
    DEFAULT_DURATION_SECONDS,
    DEFAULT_EPOCHS,
    DEFAULT_FOLDS,
    DEFAULT_HOP_LENGTH,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MODEL_NAME,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SEED,
    MODELS_DIR,
    RAW_DATA_DIR,
    TRAIN_AUDIO_DIR,
    ensure_project_dirs,
)
from src.data import assign_folds, find_primary_label_column, load_class_names, load_train_metadata
from src.dataset import BirdClefDataset
from src.metrics import birdclef_macro_auc
from src.modeling import build_model, require_torch, save_checkpoint, save_ensemble_manifest


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    torch = require_torch()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    metadata = load_train_metadata(data_dir / "train.csv")
    if args.limit:
        metadata = metadata.head(args.limit).copy()
    class_names = load_class_names(data_dir / "sample_submission.csv")
    primary_col = find_primary_label_column(metadata)
    folds = assign_folds(metadata, n_splits=args.folds, seed=args.seed, primary_col=primary_col)
    metadata = metadata.copy()
    metadata["fold"] = folds

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_paths: list[str] = []
    fold_metrics: list[dict[str, Any]] = []

    for fold in sorted(np.unique(folds)):
        if args.debug and int(fold) > 0:
            continue
        train_df = metadata[metadata["fold"] != fold].reset_index(drop=True)
        val_df = metadata[metadata["fold"] == fold].reset_index(drop=True)
        if len(train_df) == 0 or len(val_df) == 0:
            continue

        train_dataset = BirdClefDataset(
            train_df,
            class_names,
            train_audio_dir=args.train_audio_dir,
            sample_rate=args.sample_rate,
            duration_seconds=args.duration_seconds,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            training=True,
            seed=args.seed + int(fold),
        )
        val_dataset = BirdClefDataset(
            val_df,
            class_names,
            train_audio_dir=args.train_audio_dir,
            sample_rate=args.sample_rate,
            duration_seconds=args.duration_seconds,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            training=False,
            seed=args.seed,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        model = build_model(args.model, num_classes=len(class_names), in_chans=1).to(device)
        resolved_model_name = getattr(model, "_birdclef_model_name", args.model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()

        best_auc = -float("inf")
        best_payload: dict[str, Any] = {}
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_payload = evaluate(model, val_loader, device)
            metrics = birdclef_macro_auc(val_payload["targets"], val_payload["scores"])
            macro_auc = float(metrics["macro_auc"])
            print(
                f"[fold {fold}] epoch={epoch} "
                f"train_loss={train_loss:.6f} macro_auc={macro_auc:.6f}"
            )
            if not best_payload or (not np.isnan(macro_auc) and macro_auc > best_auc):
                best_auc = macro_auc
                best_payload = {
                    "fold": int(fold),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "macro_auc": macro_auc,
                    "metrics": metrics,
                    "resolved_model_name": resolved_model_name,
                }
                checkpoint_path = MODELS_DIR / f"baseline_fold{int(fold)}.pt"
                save_checkpoint(
                    checkpoint_path,
                    model,
                    model_name=resolved_model_name,
                    class_names=class_names,
                    audio_config=audio_config_from_args(args),
                    metrics=best_payload,
                )
        if best_payload:
            checkpoint_paths.append(str(Path(f"baseline_fold{int(fold)}.pt")))
            fold_metrics.append(best_payload)

    if not checkpoint_paths:
        raise RuntimeError("No folds were trained. Check the input data and fold settings.")

    metrics = {
        "model": args.model,
        "resolved_model": fold_metrics[0].get("resolved_model_name", args.model),
        "folds": fold_metrics,
        "mean_macro_auc": float(np.nanmean([item.get("macro_auc", np.nan) for item in fold_metrics])),
        "class_count": len(class_names),
        "train_rows": len(metadata),
    }
    save_ensemble_manifest(
        MODELS_DIR / DEFAULT_CHECKPOINT_NAME,
        members=checkpoint_paths,
        model_name=fold_metrics[0].get("resolved_model_name", args.model),
        class_names=class_names,
        audio_config=audio_config_from_args(args),
        metrics=metrics,
    )
    metrics_path = MODELS_DIR / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[saved] checkpoint={MODELS_DIR / DEFAULT_CHECKPOINT_NAME}")
    print(f"[saved] metrics={metrics_path}")


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:  # type: ignore[no-untyped-def]
    torch = require_torch()
    model.train()
    total_loss = 0.0
    total_count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        batch_size = x.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size
    return total_loss / max(1, total_count)


def evaluate(model, loader, device) -> dict[str, np.ndarray]:  # type: ignore[no-untyped-def]
    torch = require_torch()
    model.eval()
    scores = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(y.numpy())
    return {
        "scores": np.vstack(scores) if scores else np.zeros((0, 0), dtype=np.float32),
        "targets": np.vstack(targets) if targets else np.zeros((0, 0), dtype=np.float32),
    }


def audio_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "sample_rate": args.sample_rate,
        "duration_seconds": args.duration_seconds,
        "n_mels": args.n_mels,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BirdCLEF+ 2026 baseline model.")
    parser.add_argument("--data-dir", default=str(RAW_DATA_DIR))
    parser.add_argument("--train-audio-dir", default=str(TRAIN_AUDIO_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--duration-seconds", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--n-mels", type=int, default=DEFAULT_N_MELS)
    parser.add_argument("--n-fft", type=int, default=DEFAULT_N_FFT)
    parser.add_argument("--hop-length", type=int, default=DEFAULT_HOP_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.debug:
        args.epochs = min(args.epochs, 1)
        if not args.limit:
            args.limit = 256
    return args


if __name__ == "__main__":
    main()
