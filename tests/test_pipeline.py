from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from voice_eval import (
    Segment,
    TranscriptEvaluation,
    build_report,
    evaluate_transcript,
    transcribe_with_speakers,
)


class _FakeTranscriptions:
    def create(self, **_: object) -> object:
        return {
            "segments": [
                {
                    "speaker": "speaker_1",
                    "start": 0.0,
                    "end": 1.2,
                    "text": "How can I help?",
                },
                {
                    "speaker": "speaker_0",
                    "start": 1.5,
                    "end": 2.8,
                    "text": "I need to update my address.",
                },
            ]
        }


class _FakeAudio:
    transcriptions = _FakeTranscriptions()


class _FakeResponses:
    def __init__(self, evaluation: TranscriptEvaluation) -> None:
        self._evaluation = evaluation

    def parse(self, **_: object) -> object:
        return SimpleNamespace(output_parsed=self._evaluation)


class _FakeOpenAI:
    evaluation: TranscriptEvaluation | None = None

    def __init__(self, **_: object) -> None:
        self.audio = _FakeAudio()
        if self.evaluation is not None:
            self.responses = _FakeResponses(self.evaluation)


def test_transcription_response_is_normalized(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))
    audio_path = tmp_path / "call.mp3"
    audio_path.write_bytes(b"test-audio")

    segments = transcribe_with_speakers(audio_path, api_key="test-key")

    assert [segment.speaker for segment in segments] == ["speaker_1", "speaker_0"]
    assert segments[1].text == "I need to update my address."


def test_transcript_evaluation_and_report(
    monkeypatch: object,
) -> None:
    evaluation = TranscriptEvaluation(
        agent_speaker="speaker_1",
        agent_speaker_confidence=0.93,
        overall_assessment="review",
        task_completion_score=80,
        task_completion_reason="The next step was not confirmed.",
        coherence_score=5,
        relevance_score=4,
        context_retention_score=4,
        repetition_score=5,
        required_fact_coverage_score=None,
        missing_or_incorrect_facts=[],
        factuality_status="not_checked",
        factuality_note="No reference facts were supplied.",
        summary="The request was handled, but the next step was unclear.",
        issues=[],
    )
    _FakeOpenAI.evaluation = evaluation
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))

    segments = [
        Segment("speaker_0", 0.0, 1.0, "I need help."),
        Segment("speaker_1", 1.5, 2.5, "I can help with that."),
    ]
    result = evaluate_transcript(
        segments=segments,
        api_key="test-key",
        task="Resolve the request",
        reference_facts="",
    )
    report = build_report(
        segments=segments,
        transcript_eval=result,
        interaction_metrics={"median_response_seconds": 0.5},
        voice_metrics={"speech_rate_wpm": 140.0},
    )

    assert result.agent_speaker == "speaker_1"
    assert report["assessment"] == "review"
    assert report["task"]["score"] == 80
    assert report["facts"]["status"] == "not_checked"
