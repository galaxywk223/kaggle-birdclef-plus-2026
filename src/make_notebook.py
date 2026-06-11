from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import DEFAULT_CHECKPOINT_NAME, NOTEBOOKS_DIR, ensure_project_dirs


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook(
        checkpoint_dataset=args.checkpoint_dataset,
        checkpoint_name=Path(args.checkpoint).name,
    )
    output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"[saved] notebook={output_path}")


def build_notebook(*, checkpoint_dataset: str, checkpoint_name: str) -> dict:
    source = f"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import soundfile as sf
except Exception:
    sf = None

DATA_DIR = Path('/kaggle/input/birdclef-2026')
MODEL_DIR = Path('/kaggle/input/{checkpoint_dataset}')
CHECKPOINT_PATH = MODEL_DIR / '{checkpoint_name}'
SAMPLE_SUBMISSION_PATH = DATA_DIR / 'sample_submission.csv'
TEST_SOUNDSCAPES_DIR = DATA_DIR / 'test_soundscapes'
OUTPUT_PATH = Path('/kaggle/working/submission.csv')

def read_audio(path, sample_rate):
    if sf is None:
        raise RuntimeError('soundfile is required in the Kaggle inference environment')
    waveform, native_rate = sf.read(path, always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1 if waveform.shape[0] > waveform.shape[1] else 0)
    if native_rate != sample_rate:
        target_length = max(1, int(round(waveform.shape[0] * sample_rate / native_rate)))
        old_positions = np.linspace(0.0, 1.0, num=waveform.shape[0], endpoint=True)
        new_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=True)
        waveform = np.interp(new_positions, old_positions, waveform).astype(np.float32)
    return np.nan_to_num(waveform, copy=False).astype(np.float32, copy=False)

def parse_row_id(row_id):
    soundscape_id, end_second = str(row_id).rsplit('_', 1)
    return soundscape_id, int(end_second)

def segment_waveform(waveform, sample_rate, end_second, duration_seconds):
    target_length = int(round(sample_rate * duration_seconds))
    end = int(round(float(end_second) * sample_rate))
    start = end - target_length
    output = np.zeros(target_length, dtype=np.float32)
    source_start = max(0, start)
    source_end = min(waveform.shape[0], end)
    if source_end <= source_start:
        return output
    dest_start = source_start - start
    dest_end = dest_start + (source_end - source_start)
    output[dest_start:dest_end] = waveform[source_start:source_end]
    return output

def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)

def mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax):
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center <= left:
            center = min(left + 1, filters.shape[1] - 1)
        if right <= center:
            right = min(center + 1, filters.shape[1])
        if center > left:
            filters[i, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            filters[i, center:right] = (right - np.arange(center, right)) / (right - center)
    return filters

def waveform_to_mel(waveform, sample_rate, n_mels=128, n_fft=2048, hop_length=512):
    if waveform.shape[0] < n_fft:
        waveform = np.pad(waveform, (0, n_fft - waveform.shape[0]))
    frames = np.lib.stride_tricks.sliding_window_view(waveform, n_fft)[::hop_length]
    if frames.shape[0] == 0:
        frames = waveform[:n_fft][None, :]
    window = np.hanning(n_fft).astype(np.float32)
    fft = np.fft.rfft(frames * window[None, :], n=n_fft, axis=1)
    power = (np.abs(fft) ** 2).T.astype(np.float32)
    filters = mel_filterbank(sample_rate, n_fft, n_mels, 20.0, sample_rate / 2)
    mel = np.log1p(np.maximum(filters @ power, 0.0))
    std = float(mel.std())
    mel = (mel - float(mel.mean())) / std if std > 1e-6 else mel - float(mel.mean())
    return mel.astype(np.float32)

class SimpleAudioCnn(nn.Module):
    def __init__(self, num_classes, in_chans=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_chans, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

def build_model(model_name, num_classes):
    if model_name in {'simple_cnn', 'simple_audio_cnn'}:
        return SimpleAudioCnn(num_classes)
    try:
        import timm
        return timm.create_model(model_name, pretrained=False, in_chans=1, num_classes=num_classes)
    except Exception:
        return SimpleAudioCnn(num_classes)

def load_payload(path):
    return torch.load(path, map_location='cpu')

def load_models(checkpoint_path, checkpoint):
    if checkpoint.get('checkpoint_type') == 'ensemble':
        models = []
        for member in checkpoint.get('members', []):
            member_path = checkpoint_path.parent / member
            models.extend(load_models(member_path, load_payload(member_path)))
        return models
    model = build_model(checkpoint.get('model_name', 'efficientnet_b0'), len(checkpoint['class_names']))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return [model]

sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
checkpoint = load_payload(CHECKPOINT_PATH)
class_names = [col for col in sample.columns if col != 'row_id']
assert list(checkpoint.get('class_names', class_names)) == class_names
audio_config = checkpoint.get('audio_config', {{}})
sample_rate = int(audio_config.get('sample_rate', 32000))
duration_seconds = float(audio_config.get('duration_seconds', 5.0))
n_mels = int(audio_config.get('n_mels', 128))
n_fft = int(audio_config.get('n_fft', 2048))
hop_length = int(audio_config.get('hop_length', 512))
models = load_models(CHECKPOINT_PATH, checkpoint)

rows = []
row_ids = []
cache = {{}}
batch = []
batch_ids = []

def flush_batch():
    if not batch:
        return
    x = torch.from_numpy(np.stack(batch).astype(np.float32))
    preds = []
    with torch.no_grad():
        for model in models:
            preds.append(torch.sigmoid(model(x)).numpy())
    mean_pred = np.mean(preds, axis=0)
    rows.extend(mean_pred)
    row_ids.extend(batch_ids)
    batch.clear()
    batch_ids.clear()

for row_id in sample['row_id'].astype(str):
    soundscape_id, end_second = parse_row_id(row_id)
    if soundscape_id not in cache:
        path = TEST_SOUNDSCAPES_DIR / f'{{soundscape_id}}.ogg'
        cache[soundscape_id] = read_audio(path, sample_rate)
    segment = segment_waveform(cache[soundscape_id], sample_rate, end_second, duration_seconds)
    mel = waveform_to_mel(segment, sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    batch.append(mel[None, :, :])
    batch_ids.append(row_id)
    if len(batch) >= 32:
        flush_batch()
flush_batch()

submission = pd.DataFrame(rows, columns=class_names)
submission.insert(0, 'row_id', row_ids)
submission.to_csv(OUTPUT_PATH, index=False)
submission.head()
""".strip()
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# BirdCLEF+ 2026 CPU Inference\n",
                    "\n",
                    "This notebook generates `submission.csv` from trained checkpoint weights.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in source.splitlines()],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Kaggle CPU inference notebook.")
    parser.add_argument("--checkpoint", default=str(Path("models") / DEFAULT_CHECKPOINT_NAME))
    parser.add_argument("--checkpoint-dataset", default="birdclef-2026-baseline")
    parser.add_argument("--output", default=str(NOTEBOOKS_DIR / "birdclef_2026_cpu_inference.ipynb"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
