from __future__ import annotations

from pathlib import Path

from voice_agent_eval import (
    AgentResponseReview,
    Segment,
    TranscriptEvaluation,
    VoiceAgentEvaluator,
)


def test_evaluator_orchestrates_the_shared_pipeline(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "call.mp3"
    audio_path.write_bytes(b"audio")

    segments = [
        Segment("customer", 0.0, 1.0, "Can you help me?"),
        Segment("agent", 1.4, 2.4, "Yes, I can help."),
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
                customer_request="The customer asked for help.",
                agent_response="The agent confirmed it could help.",
                outcome="answered",
                score=5,
                note="The response directly addressed the request.",
            )
        ],
        required_fact_coverage_score=None,
        missing_or_incorrect_facts=[],
        factuality_status="not_checked",
        factuality_note="No reference facts were supplied.",
        summary="The agent handled the request clearly.",
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
    monkeypatch.setattr(
        "voice_agent_eval.pipeline.calculate_interaction_metrics",
        lambda **_kwargs: {"median_response_seconds": 0.4},
    )
    monkeypatch.setattr(
        "voice_agent_eval.pipeline.convert_to_analysis_wav",
        lambda _source, target: target.write_bytes(b"wav") or target,
    )
    monkeypatch.setattr(
        "voice_agent_eval.pipeline.calculate_voice_metrics",
        lambda **_kwargs: {"speech_rate_wpm": 145.0},
    )

    report = VoiceAgentEvaluator(api_key="test-key").evaluate(
        audio_path,
        task="Help the customer",
    )

    assert report["assessment"] == "pass"
    assert report["scope"]["evaluated_party"] == "agent"
    assert report["agent_response"]["turn_reviews"][0]["outcome"] == "answered"
    assert report["metadata"]["evaluation_model"] == "gpt-5-mini"
