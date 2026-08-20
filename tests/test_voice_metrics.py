from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from voice_agent_eval import Segment, calculate_voice_metrics


def test_voice_metrics_on_synthetic_agent_audio(tmp_path: Path) -> None:
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
    assert "segments attributed to the agent" in metrics["note"]


def test_customer_clipping_is_not_counted_as_agent_clipping(tmp_path: Path) -> None:
    sample_rate = 16_000
    customer = np.ones(sample_rate, dtype=np.float32)
    time = np.arange(sample_rate) / sample_rate
    agent = 0.2 * np.sin(2 * np.pi * 220 * time)
    audio = np.concatenate([customer, agent])

    wav_path = tmp_path / "mixed.wav"
    sf.write(wav_path, audio, sample_rate, subtype="FLOAT")

    metrics = calculate_voice_metrics(
        wav_path=wav_path,
        segments=[
            Segment("customer", 0.0, 1.0, "Customer speech"),
            Segment("agent", 1.0, 2.0, "Agent response"),
        ],
        agent_speaker="agent",
    )

    assert metrics["clipping_ratio"] == 0.0
    assert metrics["agent_audio_seconds"] == 1.0
