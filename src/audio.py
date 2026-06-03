from __future__ import annotations

from pathlib import Path

import numpy as np


def read_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """Read an audio file as mono float32 and resample to the target rate."""

    path = Path(path)
    try:
        import soundfile as sf

        waveform, native_rate = sf.read(path, always_2d=False)
    except Exception:
        try:
            import librosa

            waveform, native_rate = librosa.load(path, sr=None, mono=False)
        except Exception as exc:
            raise RuntimeError(
                "Audio reading requires soundfile or librosa. "
                f"Unable to read {path}."
            ) from exc

    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1 if waveform.shape[0] > waveform.shape[1] else 0)
    if native_rate != sample_rate:
        waveform = resample_linear(waveform, native_rate, sample_rate)
    return np.nan_to_num(waveform, copy=False).astype(np.float32, copy=False)


def resample_linear(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return waveform.astype(np.float32, copy=False)
    if waveform.size == 0:
        return waveform.astype(np.float32, copy=False)
    target_length = max(1, int(round(waveform.shape[0] * target_rate / source_rate)))
    old_positions = np.linspace(0.0, 1.0, num=waveform.shape[0], endpoint=True)
    new_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=True)
    return np.interp(new_positions, old_positions, waveform).astype(np.float32)


def crop_or_pad(
    waveform: np.ndarray,
    sample_rate: int,
    duration_seconds: float,
    *,
    mode: str = "center",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    target_length = int(round(sample_rate * duration_seconds))
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.shape[0] == target_length:
        return waveform
    if waveform.shape[0] < target_length:
        output = np.zeros(target_length, dtype=np.float32)
        offset = (target_length - waveform.shape[0]) // 2
        output[offset : offset + waveform.shape[0]] = waveform
        return output

    max_start = waveform.shape[0] - target_length
    if mode == "random":
        if rng is None:
            rng = np.random.default_rng()
        start = int(rng.integers(0, max_start + 1))
    elif mode == "start":
        start = 0
    else:
        start = max_start // 2
    return waveform[start : start + target_length].astype(np.float32, copy=False)


def segment_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    end_second: int | float,
    duration_seconds: float,
) -> np.ndarray:
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


def waveform_to_mel(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: float = 20.0,
    fmax: float | None = None,
) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0:
        waveform = np.zeros(n_fft, dtype=np.float32)
    spectrum = _power_spectrogram(waveform, n_fft=n_fft, hop_length=hop_length)
    filters = _mel_filterbank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax or sample_rate / 2,
    )
    mel = filters @ spectrum
    mel = np.log1p(np.maximum(mel, 0.0))
    mean = float(mel.mean())
    std = float(mel.std())
    if std > 1e-6:
        mel = (mel - mean) / std
    else:
        mel = mel - mean
    return mel.astype(np.float32)


def _power_spectrogram(waveform: np.ndarray, *, n_fft: int, hop_length: int) -> np.ndarray:
    if waveform.shape[0] < n_fft:
        waveform = np.pad(waveform, (0, n_fft - waveform.shape[0]))
    frame_count = 1 + max(0, (waveform.shape[0] - n_fft) // hop_length)
    if frame_count <= 0:
        frame_count = 1
    frames = np.lib.stride_tricks.sliding_window_view(waveform, n_fft)[::hop_length]
    if frames.shape[0] == 0:
        frames = waveform[:n_fft][None, :]
    window = np.hanning(n_fft).astype(np.float32)
    fft = np.fft.rfft(frames[:frame_count] * window[None, :], n=n_fft, axis=1)
    power = (np.abs(fft) ** 2).T
    return power.astype(np.float32)


def _mel_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
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


def _hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)

