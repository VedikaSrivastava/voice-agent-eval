# Validation

The project uses three levels of validation so that a passing unit test is not confused with a verified production evaluation.

## 1. Deterministic and mocked tests

The regular test suite covers:

- customer-to-agent response gaps, long pauses, and overlaps
- agent-only acoustic feature extraction
- report schema and turn-level response reviews
- `.env.local` loading
- CLI and package entry points
- mocked transcription and structured transcript evaluation
- Streamlit import behavior
- public-sample download integrity and cleanup behavior

These tests do not spend API credits and run in GitHub Actions.

## 2. Public recording preflight

CI downloads one English support and billing call from the [AxonData English Contact Center Audio Dataset](https://huggingface.co/datasets/AxonData/english-contact-center-audio-dataset). The dataset is published under CC BY-NC 4.0. The recording is not committed or redistributed by this repository.

The sample is pinned to dataset commit `f89f1d3318ad1939c4cadf4d5dde06a2fb5badc5` and verified before use:

```text
File: 1755884171.51632.mp3
Expected size: 836208 bytes
SHA-256: 894959639d644cfb1022ae42c83a1d5f53afecae9e1dbdcdd7cb1ff4f881a348
```

The preflight exercises:

- remote MP3 download and checksum verification
- MP3 decoding
- FFmpeg-backed conversion to WAV
- loading the decoded waveform
- pitch, loudness, and clipping feature extraction

This confirms that the audio-processing path works on a real telephone-quality recording. It does **not** call the transcription or transcript-judge APIs, and it does not claim that the sample agent was evaluated.

The check is implemented by:

```text
scripts/download_public_sample.py
scripts/public_sample_smoke.py
```

## 3. Live end-to-end evaluation

A live end-to-end check additionally requires an API key and sends the selected recording to the configured transcription and evaluation APIs. Run it locally with a recording that you are permitted to process:

```bat
voice-agent-eval call.mp3 --goal "Answer the customer's request and confirm the next step" --output report.json
```

A successful live run verifies:

1. speaker diarization and transcription
2. identification of the agent speaker
3. turn-by-turn customer-request-to-agent-response evaluation
4. response timing and possible interruption detection
5. agent-only acoustic feature extraction
6. JSON report generation

The live API step is deliberately not part of pull-request CI because it requires a secret, sends audio to external services, and incurs usage cost.

## Before production use

Before treating scores as quality gates, validate the evaluator against a human-reviewed call set and measure:

- issue precision and recall
- timestamp accuracy
- agreement with human task and response-quality scores
- behavior across accents, call types, audio quality, and agent voices
- stability across evaluator-model and rubric versions
