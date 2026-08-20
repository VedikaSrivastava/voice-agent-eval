# Evaluation Notes

## Scope

This project evaluates the **voice agent**, not the customer.

Customer speech is still necessary because it establishes what the agent needed to answer, which details were already supplied, when the customer finished speaking, and whether the agent cut into a turn. None of the scores should judge the customer's accent, fluency, tone, emotion, or willingness to cooperate.

The first version accepts a completed recording rather than live system traces. That makes it useful for post-call review while keeping the input simple.

## 1. Customer request to agent response

The most direct evaluation is a turn-level comparison:

```text
customer question, request, correction, or concern
                    ↓
next relevant agent response
```

For each substantive customer turn, the transcript evaluator records:

- what the customer asked, provided, or corrected
- what the agent said in response
- whether the response answered, partially answered, missed, clarified, or did not require an answer
- a one-to-five agent-response score
- a short explanation and the relevant timestamps

Greetings and acknowledgements are skipped unless they reveal a failure. The timestamps are taken from the speaker-attributed transcript so a reviewer can return to the relevant point in the recording.

This turn-level view is important because an overall score can be imperfect. A reviewer should still be able to see the exact customer context and agent response that produced a flag.

## 2. Task completion

Task completion measures whether the agent achieved or meaningfully advanced the call goal supplied by the reviewer. It is not a score for whether the customer cooperated.

When a goal cannot be completed because the customer declines to continue or the call ends early, the evaluator should describe that context and judge whether the agent handled the situation appropriately.

A production version should replace the free-text goal with a call-type rubric containing required fields, actions, outcomes, and escalation rules.

## 3. Agent response quality

These scores apply only to the agent:

- **Response alignment:** did the agent address the customer's actual question, request, or correction?
- **Coherence:** did the response follow logically from the prior turn?
- **Relevance:** did the agent stay on the request instead of drifting into unrelated information?
- **Context retention:** did the agent reuse details and corrections already supplied by the customer?
- **Repetition:** did the agent avoid asking the same question or repeating the same explanation unnecessarily?

These are model-based judgments. The timestamped evidence and turn reviews remain visible so the scores can be audited.

## 4. Facts and required points

Reference facts are optional. When they are supplied, the evaluator checks whether the agent covered them and whether any agent statement conflicts with them.

When no source of truth is supplied, factuality is marked `not_checked`. The evaluation model is not asked to guess whether a domain-specific statement is true.

A stronger version can retrieve supporting passages from approved documents, product data, tool results, or call-specific backend records, then attach the supporting source to each evaluated claim.

## 5. Agent response time

For each customer-to-agent turn:

```text
response gap = agent speech start - customer speech end
```

The report includes the median, P95, and maximum positive gap. The median describes the typical experience, while P95 makes occasional slow responses visible.

This is the delay heard in the recording. It cannot separate ASR, model, tool-call, speech-generation, buffering, or network time. The practical next step is to join the audio report with runtime event timestamps for a full latency breakdown.

## 6. Long pauses and interruptions

A positive response gap above the configured threshold is reported as a long pause.

When the agent begins before the customer segment ends:

```text
overlap = customer speech end - agent speech start
```

The overlap is marked as a possible agent interruption. This is intentionally cautious because a short acknowledgement may be natural. A stronger classifier should inspect the surrounding exchange and distinguish backchannels from disruptive cutoffs.

Speaker diarization is another source of error. Dual-channel recordings, an explicit agent-speaker label, or a quick manual speaker correction would make these metrics more reliable.

## 7. Agent voice delivery

Acoustic features are extracted from segments attributed to the agent. Customer-only portions of the recording are excluded from the reported voice metrics.

### Speech rate

```text
speech rate = agent transcript words / agent active speech time
```

The result is reported in words per minute. It can surface unusually rushed or slow delivery, but should eventually be compared with a baseline for the selected voice and call type.

### Pitch variation

The evaluator estimates F0 on voiced agent frames and measures movement relative to the agent's median pitch. This avoids comparing absolute pitch between different voices.

### Loudness variation and clipping

The evaluator measures loudness variation across active agent frames and reports the share of agent samples near digital clipping.

The current modulation label is a simple heuristic over pitch and loudness variation. It is not an emotion, personality, or customer-sentiment classifier. A production version should calibrate it against human-rated examples for each agent voice or replace it with a validated perceptual speech-quality model.

## 8. Report shape and reuse

The Streamlit app, Python package, and command-line interface all run the same evaluator and return the same JSON structure. This keeps the UI thin and allows the evaluation logic to be reused in another application, a batch job, or a test suite.

The report contains a `schema_version` and model/threshold metadata so later versions can be compared without silently changing the meaning of an existing result. A synthetic example is available in [example-report.json](example-report.json).

## Why the report keeps component metrics separate

A single score can hide important differences. A slow but factually correct call is not the same problem as a smooth call that gives an unsupported answer.

The prototype therefore returns a pass, review, or fail assessment while keeping task, response, timing, voice, and factuality signals separate. Use-case-specific weighting and hard-failure rules should be added only after reviewing real calls.

## Validation plan

The next meaningful step is a small human-reviewed call set. Reviewers should label:

- whether each agent response answered the corresponding customer turn
- task success
- missing or incorrect facts
- repeated questions or lost context
- slow responses and disruptive interruptions
- voice-delivery problems
- overall acceptability

The evaluator can then be measured using issue precision and recall, timestamp accuracy, score agreement, and false-positive rate. Thresholds should be tuned by call type and agent voice rather than treated as universal values.
