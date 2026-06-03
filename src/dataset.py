from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.audio import crop_or_pad, read_audio, waveform_to_mel
from src.config import (
    DEFAULT_DURATION_SECONDS,
    DEFAULT_HOP_LENGTH,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEFAULT_SAMPLE_RATE,
    TRAIN_AUDIO_DIR,
)
from src.data import build_targets, find_filename_column, resolve_audio_path


class BirdClefDataset:
    def __init__(
        self,
        metadata: pd.DataFrame,
        class_names: list[str],
        *,
        train_audio_dir: str | Path = TRAIN_AUDIO_DIR,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        duration_seconds: float = DEFAULT_DURATION_SECONDS,
        n_mels: int = DEFAULT_N_MELS,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        training: bool = False,
        seed: int = 42,
    ) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.class_names = class_names
        self.train_audio_dir = Path(train_audio_dir)
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.training = training
        self.rng = np.random.default_rng(seed)
        self.targets = build_targets(self.metadata, class_names)
        self.filename_column = find_filename_column(self.metadata)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int):
        torch = _require_torch()
        row = self.metadata.iloc[index]
        path = resolve_audio_path(row, self.train_audio_dir)
        waveform = read_audio(path, self.sample_rate)
        waveform = crop_or_pad(
            waveform,
            self.sample_rate,
            self.duration_seconds,
            mode="random" if self.training else "center",
            rng=self.rng,
        )
        mel = waveform_to_mel(
            waveform,
            self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        x = torch.from_numpy(mel[None, :, :])
        y = torch.from_numpy(self.targets[index])
        return x, y


def _require_torch():
    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required to use BirdClefDataset.") from exc

