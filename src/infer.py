from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.audio import read_audio, segment_waveform, waveform_to_mel
from src.config import (
    DEFAULT_CHECKPOINT_NAME,
    MODELS_DIR,
    RAW_DATA_DIR,
    ROW_ID_COL,
    SAMPLE_SUBMISSION_PATH,
    SUBMISSIONS_DIR,
    TEST_SOUNDSCAPES_DIR,
    ensure_project_dirs,
)
from src.data import load_sample_submission, parse_row_id
from src.modeling import build_model, load_checkpoint, require_torch


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample = load_sample_submission(args.sample_submission)
    checkpoint_path = Path(args.checkpoint)
    predictions = predict_submission(
        checkpoint_path=checkpoint_path,
        sample_submission=sample,
        test_soundscapes_dir=Path(args.test_soundscapes_dir),
        batch_size=args.batch_size,
        device=args.device,
    )
    predictions.to_csv(output_path, index=False)
    print(f"[saved] submission={output_path}")


def predict_submission(
    *,
    checkpoint_path: Path,
    sample_submission: pd.DataFrame,
    test_soundscapes_dir: Path,
    batch_size: int = 32,
    device: str = "",
) -> pd.DataFrame:
    torch = require_torch()
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    class_names = [col for col in sample_submission.columns if col != ROW_ID_COL]
    checkpoint_classes = checkpoint.get("class_names", class_names)
    if list(checkpoint_classes) != list(class_names):
        raise ValueError("Checkpoint class order does not match sample_submission.csv.")

    runtime_device = torch.device(device if device else "cpu")
    models = load_models_from_checkpoint(checkpoint_path, checkpoint, runtime_device)
    audio_config = checkpoint.get("audio_config", {})
    sample_rate = int(audio_config.get("sample_rate", 32_000))
    duration_seconds = float(audio_config.get("duration_seconds", 5.0))
    n_mels = int(audio_config.get("n_mels", 128))
    n_fft = int(audio_config.get("n_fft", 2048))
    hop_length = int(audio_config.get("hop_length", 512))

    rows = []
    mel_batch = []
    row_ids = []
    audio_cache: dict[str, np.ndarray] = {}

    for row_id in sample_submission[ROW_ID_COL].astype(str):
        parsed = parse_row_id(row_id)
        waveform = audio_cache.get(parsed.soundscape_id)
        if waveform is None:
            waveform = read_soundscape(test_soundscapes_dir, parsed.soundscape_id, sample_rate)
            audio_cache[parsed.soundscape_id] = waveform
        segment = segment_waveform(
            waveform,
            sample_rate=sample_rate,
            end_second=parsed.end_second,
            duration_seconds=duration_seconds,
        )
        mel = waveform_to_mel(
            segment,
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        mel_batch.append(mel[None, :, :])
        row_ids.append(row_id)
        if len(mel_batch) >= batch_size:
            rows.extend(run_batch(models, mel_batch, runtime_device))
            mel_batch = []

    if mel_batch:
        rows.extend(run_batch(models, mel_batch, runtime_device))

    output = pd.DataFrame(rows, columns=class_names)
    output.insert(0, ROW_ID_COL, row_ids)
    return output


def load_models_from_checkpoint(checkpoint_path: Path, checkpoint: dict[str, Any], device):  # type: ignore[no-untyped-def]
    torch = require_torch()
    checkpoint_type = checkpoint.get("checkpoint_type", "single")
    if checkpoint_type == "ensemble":
        models = []
        for member in checkpoint.get("members", []):
            member_path = checkpoint_path.parent / member
            member_payload = load_checkpoint(member_path, map_location="cpu")
            models.extend(load_models_from_checkpoint(member_path, member_payload, device))
        return models

    class_names = checkpoint["class_names"]
    model_name = checkpoint.get("model_name", "efficientnet_b0")
    model = build_model(model_name, num_classes=len(class_names), in_chans=1)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return [model]


def run_batch(models, mel_batch: list[np.ndarray], device) -> list[np.ndarray]:  # type: ignore[no-untyped-def]
    torch = require_torch()
    x = torch.from_numpy(np.stack(mel_batch).astype(np.float32)).to(device)
    predictions = []
    with torch.no_grad():
        for model in models:
            predictions.append(torch.sigmoid(model(x)).cpu().numpy())
    return list(np.mean(predictions, axis=0))


def read_soundscape(test_soundscapes_dir: Path, soundscape_id: str, sample_rate: int) -> np.ndarray:
    candidates = [
        test_soundscapes_dir / f"{soundscape_id}.ogg",
        test_soundscapes_dir / f"{soundscape_id}.mp3",
        test_soundscapes_dir / f"{soundscape_id}.wav",
        test_soundscapes_dir / soundscape_id,
    ]
    for path in candidates:
        if path.exists():
            return read_audio(path, sample_rate)
    raise FileNotFoundError(
        f"Unable to locate soundscape '{soundscape_id}' under {test_soundscapes_dir}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BirdCLEF+ 2026 submission.csv.")
    parser.add_argument("--checkpoint", default=str(MODELS_DIR / DEFAULT_CHECKPOINT_NAME))
    parser.add_argument("--sample-submission", default=str(SAMPLE_SUBMISSION_PATH))
    parser.add_argument("--test-soundscapes-dir", default=str(TEST_SOUNDSCAPES_DIR))
    parser.add_argument("--output", default=str(SUBMISSIONS_DIR / "submission.csv"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="")
    return parser.parse_args()


if __name__ == "__main__":
    main()
