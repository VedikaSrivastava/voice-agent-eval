from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class Segment:
    speaker: str
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class EvaluationConfig:
    transcription_model: str = "gpt-4o-transcribe-diarize"
    evaluation_model: str = "gpt-5-mini"
    long_pause_seconds: float = 2.0
    minimum_overlap_seconds: float = 0.15


class ReviewIssue(BaseModel):
    category: Literal[
        "task_completion",
        "unanswered_request",
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


class AgentResponseReview(BaseModel):
    customer_timestamp_seconds: float | None = None
    agent_timestamp_seconds: float | None = None
    customer_request: str
    agent_response: str
    outcome: Literal[
        "answered",
        "partially_answered",
        "not_answered",
        "clarification",
        "no_response_required",
    ]
    score: int = Field(
        ge=1,
        le=5,
        description="5 means the agent handled the customer turn fully and correctly.",
    )
    note: str


class TranscriptEvaluation(BaseModel):
    agent_speaker: str
    agent_speaker_confidence: float = Field(ge=0, le=1)
    overall_assessment: Literal["pass", "review", "fail"]
    task_completion_score: int = Field(ge=0, le=100)
    task_completion_reason: str
    response_alignment_score: int = Field(
        ge=1,
        le=5,
        description="How well the agent's responses address the customer's requests across the call.",
    )
    coherence_score: int = Field(
        ge=1,
        le=5,
        description="How clearly the agent's responses follow the customer's request and prior context.",
    )
    relevance_score: int = Field(
        ge=1,
        le=5,
        description="How directly the agent answers the customer's request.",
    )
    context_retention_score: int = Field(
        ge=1,
        le=5,
        description="How well the agent retains details and corrections supplied by the customer.",
    )
    repetition_score: int = Field(
        ge=1,
        le=5,
        description="5 means the agent avoids unnecessary repetition; 1 means severe looping.",
    )
    response_reviews: list[AgentResponseReview]
    required_fact_coverage_score: int | None = Field(default=None, ge=0, le=100)
    missing_or_incorrect_facts: list[str]
    factuality_status: Literal["not_checked", "supported", "mixed", "problematic"]
    factuality_note: str
    summary: str
    issues: list[ReviewIssue]
