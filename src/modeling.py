from __future__ import annotations

from pathlib import Path
from typing import Any


def require_torch():
    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for training and inference. "
            "Install the project environment from environment.yml first."
        ) from exc


def build_model(model_name: str, num_classes: int, *, in_chans: int = 1):
    require_torch()
    if model_name in {"simple_cnn", "simple_audio_cnn"}:
        model = build_simple_audio_cnn(num_classes=num_classes, in_chans=in_chans)
        setattr(model, "_birdclef_model_name", "simple_cnn")
        return model
    try:
        import timm

        model = timm.create_model(
            model_name,
            pretrained=False,
            in_chans=in_chans,
            num_classes=num_classes,
        )
        setattr(model, "_birdclef_model_name", model_name)
        return model
    except Exception:
        model = build_simple_audio_cnn(num_classes=num_classes, in_chans=in_chans)
        setattr(model, "_birdclef_model_name", "simple_cnn")
        return model


def build_simple_audio_cnn(num_classes: int, in_chans: int = 1):
    torch = require_torch()

    class SimpleAudioCnn(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(in_chans, 32, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(32),
                torch.nn.SiLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.SiLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.SiLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = torch.nn.Linear(128, num_classes)

        def forward(self, x):  # type: ignore[no-untyped-def]
            x = self.features(x)
            x = x.flatten(1)
            return self.classifier(x)

    return SimpleAudioCnn()


def save_checkpoint(
    path: str | Path,
    model,
    *,
    model_name: str,
    class_names: list[str],
    audio_config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> None:
    torch = require_torch()
    payload = {
        "checkpoint_type": "single",
        "model_name": model_name,
        "class_names": class_names,
        "audio_config": audio_config,
        "metrics": metrics or {},
        "state_dict": model.state_dict(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def save_ensemble_manifest(
    path: str | Path,
    *,
    members: list[str],
    model_name: str,
    class_names: list[str],
    audio_config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    torch = require_torch()
    payload = {
        "checkpoint_type": "ensemble",
        "members": members,
        "model_name": model_name,
        "class_names": class_names,
        "audio_config": audio_config,
        "metrics": metrics,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
    torch = require_torch()
    return torch.load(path, map_location=map_location)
