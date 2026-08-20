from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import librosa
import numpy as np
from pydantic import BaseModel, Field
from pydub import AudioSegment

MAX_AUDIO_BYTES = 25_000_000


@dataclass(frozen=True)
class Segment:
    speaker: str
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class ReviewIssue(BaseModel):
    category: Literal[
        "task_completion",
        "missing_fact",
        "factuality",
        "repetition",
        "context",
        "coherence",
        "compliance",
        "other",
    ]
    severity: Literal["low", "medium", "high"]
    timestamp_seconds: float | None = None
    evidence: str
    explanation: str


class TranscriptEvaluation(BaseModel):
    agent_speaker: str
    agent_speaker_confidence: float = Field(ge=0, le=1)
    overall_assessment: Literal["pass", "review", "fail"]
    task_completion_score: int = Field(ge=0, le=100)
    task_completion_reason: str
    coherence_score: int = Field(ge=1, le=5)
    relevance_score: int = Field(ge=1, le=5)
    context_retention_score: int = Field(ge=1, le=5)
    repetition_score: int = Field(
        ge=1,
        le=5,
        description="5 means no unnecessary repetition; 1 means severe looping or repeated questions.",
    )
    required_fact_coverage_score: int | None = Field(default=None, ge=0, le=100)
    missing_or_incorrect_facts: list[str]
    factuality_status: Literal["not_checked", "supported", "mixed", "problematic"]
    factuality_note: str
    summary: str
    issues: list[ReviewIssue]


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: _plain(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def transcribe_with_speakers(audio_path: Path, api_key: str) -> list[Segment]:
    """Transcribe an uploaded recording and keep speaker/timestamp information."""
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if audio_path.stat().st_size > MAX_AUDIO_BYTES:
        raise ValueError("The demo accepts audio files up to 25 MB.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=30 * 60)
    with audio_path.open("rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="gpt-4o-transcribe-diarize",
            file=audio_file,
            response_format="diarized_json",
            chunking_strategy="auto",
        )

    data = _plain(response)
    raw_segments = data.get("segments", []) if isinstance(data, dict) else []
    segments: list[Segment] = []

    for item in raw_segments:
        item = _plain(item)
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        speaker = str(item.get("speaker") or "unknown")
        start = float(item.get("start") or 0.0)
        end = float(item.get("end") or start)
        if text:
            segments.append(Segment(speaker=speaker, start=start, end=end, text=text))

    if not segments:
        raise ValueError("No speaker-attributed transcript was returned for this recording.")

    return sorted(segments, key=lambda segment: (segment.start, segment.end))


def format_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return ""
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def transcript_for_model(segments: list[Segment]) -> str:
    return "\n".join(
        f"{segment.speaker} | {format_timestamp(segment.start)}-{format_timestamp(segment.end)} | {segment.text}"
        for segment in segments
    )


def evaluate_transcript(
    segments: list[Segment],
    api_key: str,
    task: str,
    reference_facts: str,
    model: str = "gpt-5-mini",
) -> TranscriptEvaluation:
    speakers = sorted({segment.speaker for segment in segments})
    transcript = transcript_for_model(segments)

    system_prompt = """
You are reviewing a completed two-person call between a customer and a voice agent.
Use only the transcript and the optional reference information supplied below.

First identify which diarized speaker is the agent. Do not assume speaker_0 is the agent.
Score the call plainly and conservatively. Every issue must point to evidence in the transcript.

Rules:
- Task completion measures whether the stated goal was achieved. If no goal is supplied, infer only the basic goal visible in the call and say that you inferred it.
- Required-fact coverage checks whether supplied facts or required talking points were correctly covered.
- If no reference facts are supplied, set factuality_status to not_checked, required_fact_coverage_score to null, and do not claim that business facts were correct.
- A high factuality or compliance issue should normally make the overall assessment review or fail.
- Do not judge the acoustic quality, latency, pitch, emotion, or speaking style. Those are calculated separately from the audio.
- Keep the summary short and useful to an engineer reviewing the call.
""".strip()

    user_prompt = f"""
Known diarized speakers: {json.dumps(speakers)}

Call goal:
{task.strip() or "Not provided"}

Reference facts or required talking points:
{reference_facts.strip() or "Not provided"}

Speaker-attributed transcript:
{transcript}
""".strip()

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text_format=TranscriptEvaluation,
    )

    evaluation = response.output_parsed
    if evaluation is None:
        raise RuntimeError("The evaluator returned no structured result.")
    if evaluation.agent_speaker not in speakers:
        raise RuntimeError(
            f"The evaluator identified {evaluation.agent_speaker!r} as the agent, "
            f"but the transcript contains {speakers}."
        )
    return evaluation


def merge_adjacent_segments(
    segments: list[Segment],
    max_same_speaker_gap: float = 0.35,
) -> list[Segment]:
    """Merge fragments that likely belong to the same conversational turn."""
    if not segments:
        return []

    merged: list[Segment] = [segments[0]]
    for current in segments[1:]:
        previous = merged[-1]
        gap = current.start - previous.end
        if current.speaker == previous.speaker and gap <= max_same_speaker_gap:
            merged[-1] = Segment(
                speaker=previous.speaker,
                start=previous.start,
                end=max(previous.end, current.end),
                text=f"{previous.text} {current.text}".strip(),
            )
        else:
            merged.append(current)
    return merged


def calculate_interaction_metrics(
    segments: list[Segment],
    agent_speaker: str,
    long_pause_seconds: float = 2.0,
    minimum_overlap_seconds: float = 0.15,
) -> dict[str, Any]:
    turns = merge_adjacent_segments(segments)
    response_gaps: list[float] = []
    interruptions: list[dict[str, Any]] = []
    long_pauses: list[dict[str, Any]] = []

    for previous, current in zip(turns, turns[1:]):
        if current.speaker == agent_speaker and previous.speaker != agent_speaker:
            gap = current.start - previous.end
            if gap >= 0:
                response_gaps.append(gap)
                if gap >= long_pause_seconds:
                    long_pauses.append(
                        {
                            "timestamp_seconds": current.start,
                            "gap_seconds": round(gap, 2),
                        }
                    )
            elif abs(gap) >= minimum_overlap_seconds:
                interruptions.append(
                    {
                        "timestamp_seconds": current.start,
                        "overlap_seconds": round(abs(gap), 2),
                    }
                )

    agent_turns = [turn for turn in turns if turn.speaker == agent_speaker]
    customer_turns = [turn for turn in turns if turn.speaker != agent_speaker]
    all_speech = sum(turn.duration for turn in turns)
    agent_speech = sum(turn.duration for turn in agent_turns)

    return {
        "response_count": len(response_gaps),
        "median_response_seconds": (
            round(float(np.median(response_gaps)), 2) if response_gaps else None
        ),
        "p95_response_seconds": (
            round(float(np.percentile(response_gaps, 95)), 2) if response_gaps else None
        ),
        "max_response_seconds": round(max(response_gaps), 2) if response_gaps else None,
        "long_pause_count": len(long_pauses),
        "long_pauses": long_pauses,
        "agent_interruption_count": len(interruptions),
        "agent_interruptions": interruptions,
        "agent_turn_count": len(agent_turns),
        "customer_turn_count": len(customer_turns),
        "agent_speech_share": round(agent_speech / all_speech, 3) if all_speech else None,
        "note": "Response time and interruption metrics are estimated from diarized speech timestamps.",
    }


def convert_to_analysis_wav(audio_path: Path, wav_path: Path) -> Path:
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1)
    audio.export(wav_path, format="wav")
    return wav_path


def _agent_audio_arrays(
    y: np.ndarray,
    sample_rate: int,
    segments: list[Segment],
    agent_speaker: str,
) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for segment in segments:
        if segment.speaker != agent_speaker or segment.duration < 0.35:
            continue
        start = max(0, int(segment.start * sample_rate))
        end = min(len(y), int(segment.end * sample_rate))
        if end > start:
            arrays.append(y[start:end])
    return arrays


def calculate_voice_metrics(
    wav_path: Path,
    segments: list[Segment],
    agent_speaker: str,
) -> dict[str, Any]:
    """Calculate a few understandable acoustic features for the agent voice."""
    y, sample_rate = librosa.load(wav_path, sr=None, mono=True)
    if y.size == 0:
        raise ValueError("The decoded audio is empty.")

    agent_arrays = _agent_audio_arrays(y, sample_rate, segments, agent_speaker)
    if not agent_arrays:
        return {
            "speech_rate_wpm": None,
            "pitch_variation_semitones": None,
            "loudness_variation_db": None,
            "voice_modulation": "not available",
            "clipping_ratio": round(float(np.mean(np.abs(y) >= 0.999)), 5),
            "note": "No usable agent speech segments were available for acoustic analysis.",
        }

    agent_duration = sum(len(array) / sample_rate for array in agent_arrays)
    agent_words = sum(
        len(segment.text.split()) for segment in segments if segment.speaker == agent_speaker
    )
    speech_rate = 60.0 * agent_words / agent_duration if agent_duration else None

    f0_values: list[float] = []
    rms_db_values: list[float] = []

    for audio in agent_arrays:
        if len(audio) < int(sample_rate * 0.35):
            continue

        f0, _, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sample_rate,
        )
        if f0 is not None:
            voiced = f0[np.isfinite(f0)]
            f0_values.extend(float(value) for value in voiced if value > 0)

        rms = librosa.feature.rms(y=audio)[0]
        if rms.size:
            rms_db = librosa.amplitude_to_db(rms, ref=1.0)
            # Ignore near-silent frames inside a diarized segment.
            active = rms_db[rms_db > -55]
            rms_db_values.extend(float(value) for value in active)

    pitch_variation = None
    if f0_values:
        f0_array = np.asarray(f0_values)
        median_f0 = float(np.median(f0_array))
        semitones = 12.0 * np.log2(f0_array / median_f0)
        pitch_variation = float(np.std(semitones))

    loudness_variation = (
        float(np.std(np.asarray(rms_db_values))) if rms_db_values else None
    )

    if pitch_variation is None or loudness_variation is None:
        modulation = "not enough clean voiced audio"
    elif pitch_variation < 1.5 and loudness_variation < 3.0:
        modulation = "fairly flat"
    elif pitch_variation > 5.0 or loudness_variation > 8.0:
        modulation = "highly variable"
    else:
        modulation = "balanced"

    return {
        "speech_rate_wpm": round(speech_rate, 1) if speech_rate is not None else None,
        "pitch_variation_semitones": (
            round(pitch_variation, 2) if pitch_variation is not None else None
        ),
        "loudness_variation_db": (
            round(loudness_variation, 2) if loudness_variation is not None else None
        ),
        "voice_modulation": modulation,
        "clipping_ratio": round(float(np.mean(np.abs(y) >= 0.999)), 5),
        "note": (
            "Voice modulation is a simple heuristic based on pitch and loudness variation. "
            "It is not an emotion or personality classifier."
        ),
    }


def build_report(
    segments: list[Segment],
    transcript_eval: TranscriptEvaluation,
    interaction_metrics: dict[str, Any],
    voice_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "assessment": transcript_eval.overall_assessment,
        "summary": transcript_eval.summary,
        "detected_agent_speaker": transcript_eval.agent_speaker,
        "agent_speaker_confidence": transcript_eval.agent_speaker_confidence,
        "task": {
            "score": transcript_eval.task_completion_score,
            "reason": transcript_eval.task_completion_reason,
        },
        "conversation": {
            "coherence_score": transcript_eval.coherence_score,
            "relevance_score": transcript_eval.relevance_score,
            "context_retention_score": transcript_eval.context_retention_score,
            "repetition_score": transcript_eval.repetition_score,
        },
        "facts": {
            "coverage_score": transcript_eval.required_fact_coverage_score,
            "status": transcript_eval.factuality_status,
            "note": transcript_eval.factuality_note,
            "missing_or_incorrect": transcript_eval.missing_or_incorrect_facts,
        },
        "interaction": interaction_metrics,
        "voice": voice_metrics,
        "issues": [issue.model_dump() for issue in transcript_eval.issues],
        "transcript": [asdict(segment) for segment in segments],
    }
