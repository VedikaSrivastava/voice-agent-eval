# Evaluation Notes

## Why combine audio and text?

A transcript can look correct even when the call feels poor. The agent may speak over the customer, pause for several seconds, or deliver a response in a flat and difficult-to-follow way. Audio alone has the opposite problem: a call can sound smooth while the agent repeats a question, forgets an earlier answer, or gives unsupported information.

This prototype keeps the two views separate and combines them only in the final report.

## Metric groups

### 1. Response time

For each customer-to-agent turn, the app measures:

```text
response gap = agent speech start - customer speech end
```

It reports the median, P95, and maximum positive gap.

The median describes the normal experience across the call. P95 makes occasional slow responses visible instead of hiding them inside an average. A negative gap means the agent began speaking before the customer segment ended and is handled as a possible interruption.

This is perceived latency from the recording. It does not explain whether the delay came from transcription, model inference, a tool call, speech generation, or network buffering. That breakdown requires system traces.

### 2. Long pauses

A customer-to-agent response gap above the configured threshold is marked as a long pause and added to the timestamped issue list.

The current default is two seconds. It is intentionally configurable because an acceptable pause depends on the call. A longer delay may be reasonable while the agent is checking information, but awkward during a simple conversational reply.

### 3. Turn-taking and interruptions

When an agent segment begins before the previous customer segment ends, the overlap is recorded as a possible agent interruption.

```text
overlap = customer speech end - agent speech start
```

Very small overlaps are ignored. The current implementation reports the count, duration, and timestamp of the remaining overlaps.

This is an estimate. Speaker diarization is imperfect, and not every overlap is disruptive. A brief acknowledgement can be natural. A stronger version of the evaluator would classify the surrounding exchange and distinguish acknowledgements from cases where the agent actually cuts the customer off.

### 4. Speech rate

Agent speech rate is calculated as:

```text
speech rate = agent transcript words / agent active speech time
```

The result is reported in words per minute. This is useful for spotting delivery that is unusually rushed or slow, but it should not be treated as a universal pass or fail rule. Speaking rate should eventually be compared with a baseline for the selected synthetic voice and the type of call.

### 5. Pitch variation

The app estimates the fundamental frequency of voiced agent frames and expresses pitch movement relative to the agent's median pitch. The reported value is the standard deviation of that movement in semitones.

This avoids comparing absolute pitch across different voices. A very low value can be a sign of flat delivery, while a very high value can indicate unstable or exaggerated output. The current labels are heuristics and need calibration before they can support production thresholds.

### 6. Loudness variation and clipping

Loudness variation is calculated over active agent frames in decibels. The app also reports the fraction of samples that are close to digital clipping.

These signals can surface inconsistent volume, abrupt level changes, or distorted audio. They are basic quality indicators rather than a full perceptual speech-quality score.

### 7. Task completion

The transcript evaluator compares the conversation with the call goal supplied by the reviewer. It scores whether the goal was completed and gives a short reason.

A production version should define task-specific rubrics instead of relying on a free-text goal. For example, a rubric can list required fields, required actions, acceptable outcomes, and escalation conditions.

### 8. Coherence, relevance, context, and repetition

The transcript evaluator scores four conversational qualities from one to five:

- **Coherence:** each response follows logically from the prior turn
- **Relevance:** the agent addresses the customer's actual request
- **Context retention:** earlier answers and corrections are used correctly
- **Repetition:** the agent avoids repeated questions, explanations, or loops

These are model-based judgments, so the report includes transcript evidence and should be reviewed rather than treated as ground truth.

### 9. Facts and required points

Reference facts are optional. When they are supplied, the evaluator checks whether the agent covered them and whether any statement conflicts with them.

When they are not supplied, factuality is marked as `not_checked`. The app does not ask the evaluation model to guess whether domain-specific claims are true.

For a production workflow, the source of truth could come from approved policy documents, product data, a knowledge base, or call-specific backend records.

## Why there is no opaque overall score

The app returns a simple pass, review, or fail assessment, but keeps the component metrics visible. A single number can hide the difference between a slow but correct call and a smooth call containing a serious factual error.

Before adding a weighted score, the metric weights and hard-failure rules should be defined by the use case. A compliance failure or unsupported promise may need to override otherwise strong voice and conversation scores.

## How the evaluator should be validated

The next step after the prototype is a small human-reviewed dataset. Reviewers should label task success, factual issues, interruptions, slow responses, repetition, and overall acceptability. Those labels can be used to measure precision, recall, false-positive rate, timestamp accuracy, and agreement with human scores.

That process is also how latency, pause, pitch, and loudness thresholds should be calibrated. The current values are useful for a demo, but they are not universal production standards.
