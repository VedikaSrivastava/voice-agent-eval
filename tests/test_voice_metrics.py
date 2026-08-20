from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from voice_eval import Segment, calculate_voice_metrics


def test_voice_metrics_on_synthetic_audio(tmp_path: Path) -> None:
    sample_rate = 16_000
    duration_seconds = 2.0
    time = np.arange(int(sample_rate * duration_seconds)) / sample_rate
    envelope = 0.65 + 0.25 * np.sin(2 * np.pi * 0.7 * time)
    audio = 0.2 * envelope * np.sin(2 * np.pi * 220 * time)

    wav_path = tmp_path / "agent.wav"
    sf.write(wav_path, audio, sample_rate)

    metrics = calculate_voice_metrics(
        wav_path=wav_path,
        segments=[Segment("agent", 0.0, 2.0, "This is a short test response")],
        agent_speaker="agent",
    )

    assert metrics["speech_rate_wpm"] == 180.0
    assert metrics["pitch_variation_semitones"] is not None
    assert metrics["loudness_variation_db"] is not None
    assert metrics["clipping_ratio"] == 0.0
