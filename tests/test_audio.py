from __future__ import annotations

import numpy as np

from src.audio import crop_or_pad, segment_waveform, waveform_to_mel


def test_crop_or_pad_centers_short_waveform() -> None:
    output = crop_or_pad(np.array([1.0, 2.0], dtype=np.float32), sample_rate=2, duration_seconds=3.0)

    np.testing.assert_allclose(output, np.array([0.0, 0.0, 1.0, 2.0, 0.0, 0.0], dtype=np.float32))


def test_crop_or_pad_center_crops_long_waveform() -> None:
    output = crop_or_pad(np.arange(10, dtype=np.float32), sample_rate=2, duration_seconds=2.0)

    np.testing.assert_allclose(output, np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32))


def test_segment_waveform_returns_window_ending_at_requested_second() -> None:
    waveform = np.arange(10, dtype=np.float32)
    output = segment_waveform(waveform, sample_rate=2, end_second=3, duration_seconds=2.0)

    np.testing.assert_allclose(output, np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32))


def test_waveform_to_mel_has_expected_shape() -> None:
    waveform = np.sin(np.linspace(0, 4 * np.pi, 16000, dtype=np.float32))
    mel = waveform_to_mel(waveform, sample_rate=16000, n_mels=32, n_fft=512, hop_length=256)

    assert mel.shape[0] == 32
    assert mel.ndim == 2
    assert np.isfinite(mel).all()

