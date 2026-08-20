"""Public API for the voice-agent evaluation package."""

from .models import (
    AgentResponseReview,
    EvaluationConfig,
    ReviewIssue,
    Segment,
    TranscriptEvaluation,
)
from .pipeline import (
    MAX_AUDIO_BYTES,
    VoiceAgentEvaluator,
    build_report,
    calculate_interaction_metrics,
    calculate_voice_metrics,
    convert_to_analysis_wav,
    evaluate_call,
    evaluate_transcript,
    format_timestamp,
    merge_adjacent_segments,
    normalize_reference_facts,
    resolve_api_key,
    transcribe_with_speakers,
)

__all__ = [
    "MAX_AUDIO_BYTES",
    "AgentResponseReview",
    "EvaluationConfig",
    "ReviewIssue",
    "Segment",
    "TranscriptEvaluation",
    "VoiceAgentEvaluator",
    "build_report",
    "calculate_interaction_metrics",
    "calculate_voice_metrics",
    "convert_to_analysis_wav",
    "evaluate_call",
    "evaluate_transcript",
    "format_timestamp",
    "merge_adjacent_segments",
    "normalize_reference_facts",
    "resolve_api_key",
    "transcribe_with_speakers",
]

__version__ = "0.1.0"
