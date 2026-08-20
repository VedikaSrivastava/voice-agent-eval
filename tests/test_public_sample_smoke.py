from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.public_sample_smoke import inspect_public_sample


def test_public_sample_smoke_uses_real_audio_decode_path(tmp_path: Path) -> None:
    sample_rate = 16_000
    duration_seconds = 2.0
    time = np.arange(int(sample_rate * duration_seconds)) / sample_rate
    audio = 0.18 * np.sin(2 * np.pi * (180 + 20 * time) * time)

    audio_path = tmp_path / "sample.wav"
    sf.write(audio_path, audio, sample_rate)

    result = inspect_public_sample(audio_path, probe_seconds=1.5)

    assert result["duration_seconds"] == 2.0
    assert result["sample_rate_hz"] == sample_rate
    assert result["channels"] == 1
    assert result["probe_seconds"] == 1.5
    assert result["acoustic_preflight"]["pitch_variation_semitones"] is not None
