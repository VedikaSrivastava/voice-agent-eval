from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from voice_agent_eval import (
    AgentResponseReview,
    Segment,
    TranscriptEvaluation,
    VoiceAgentEvaluator,
)


def test_local_pipeline_combines_turn_timing_and_agent_audio(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    sample_rate = 16_000
    duration_seconds = 2.5
    time = np.arange(int(sample_rate * duration_seconds)) / sample_rate
    audio = np.zeros_like(time, dtype=np.float32)

    customer_mask = time < 1.0
    agent_mask = (time >= 1.4) & (time < 2.4)
    audio[customer_mask] = 0.15 * np.sin(2 * np.pi * 180 * time[customer_mask])
    audio[agent_mask] = 0.2 * np.sin(2 * np.pi * 220 * time[agent_mask])

    audio_path = tmp_path / "call.wav"
    sf.write(audio_path, audio, sample_rate, subtype="FLOAT")

    segments = [
        Segment("customer", 0.0, 1.0, "Can you explain the next step?"),
        Segment("agent", 1.4, 2.4, "I will explain the next step."),
    ]
    transcript_eval = TranscriptEvaluation(
        agent_speaker="agent",
        agent_speaker_confidence=0.99,
        overall_assessment="pass",
        task_completion_score=95,
        task_completion_reason="The agent answered the request.",
        response_alignment_score=5,
        coherence_score=5,
        relevance_score=5,
        context_retention_score=5,
        repetition_score=5,
        response_reviews=[
            AgentResponseReview(
                customer_timestamp_seconds=0.0,
                agent_timestamp_seconds=1.4,
                customer_request="The customer asked for the next step.",
                agent_response="The agent said it would explain the next step.",
                outcome="answered",
                score=5,
                note="The response directly addressed the request.",
            )
        ],
        required_fact_coverage_score=None,
        missing_or_incorrect_facts=[],
        factuality_status="not_checked",
        factuality_note="No reference facts were supplied.",
        summary="The agent answered the request clearly.",
        issues=[],
    )

    monkeypatch.setattr(
        "voice_agent_eval.pipeline.resolve_api_key",
        lambda **_kwargs: "test-key",
    )
    monkeypatch.setattr(
        "voice_agent_eval.pipeline.transcribe_with_speakers",
        lambda *_args, **_kwargs: segments,
    )
    monkeypatch.setattr(
        "voice_agent_eval.pipeline.evaluate_transcript",
        lambda **_kwargs: transcript_eval,
    )

    report = VoiceAgentEvaluator(api_key="test-key").evaluate(audio_path)

    assert report["interaction"]["median_response_seconds"] == 0.4
    assert report["voice"]["agent_audio_seconds"] == 1.0
    assert report["voice"]["clipping_ratio"] == 0.0
    assert report["agent_response"]["turn_reviews"][0]["outcome"] == "answered"
