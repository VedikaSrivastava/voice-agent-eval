from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import streamlit as st

from voice_eval import (
    MAX_AUDIO_BYTES,
    build_report,
    calculate_interaction_metrics,
    calculate_voice_metrics,
    convert_to_analysis_wav,
    evaluate_transcript,
    format_timestamp,
    transcribe_with_speakers,
)

st.set_page_config(page_title="Voice Agent Eval", page_icon="🎧", layout="centered")


def get_api_key() -> str | None:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    try:
        return str(st.secrets["OPENAI_API_KEY"])
    except Exception:
        return None


def show_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}s"


st.title("Voice Agent Eval")
st.write(
    "Upload a customer-agent recording and get a small post-call scorecard. "
    "The demo looks at the transcript, response timing, interruptions, and a few basic voice-delivery signals."
)

uploaded_file = st.file_uploader(
    "Call recording",
    type=["mp3", "wav", "m4a", "mp4", "webm"],
    help="Maximum file size for this demo: 25 MB.",
)

call_goal = st.text_input(
    "What was the agent supposed to accomplish?",
    placeholder="For example: resolve the customer request and confirm the next step",
)

reference_facts = st.text_area(
    "Facts or required talking points (optional)",
    placeholder=(
        "Add any facts the agent should have used, one per line. "
        "Without this, the demo will not claim that business facts were correct."
    ),
    height=120,
)

if uploaded_file is not None:
    st.audio(uploaded_file)

run_clicked = st.button(
    "Evaluate call",
    type="primary",
    disabled=uploaded_file is None,
    use_container_width=True,
)

if run_clicked and uploaded_file is not None:
    api_key = get_api_key()
    if not api_key:
        st.error(
            "OPENAI_API_KEY is not configured. Add it to your environment or Streamlit secrets."
        )
        st.stop()

    if uploaded_file.size > MAX_AUDIO_BYTES:
        st.error("This demo accepts audio files up to 25 MB.")
        st.stop()

    suffix = Path(uploaded_file.name).suffix.lower() or ".mp3"

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            original_path = temp_path / f"call{suffix}"
            wav_path = temp_path / "analysis.wav"
            original_path.write_bytes(uploaded_file.getvalue())

            with st.status("Evaluating the call", expanded=True) as status:
                st.write("Separating speakers and generating the transcript...")
                segments = transcribe_with_speakers(original_path, api_key)

                st.write("Reviewing task completion, context, repetition, and facts...")
                transcript_eval = evaluate_transcript(
                    segments=segments,
                    api_key=api_key,
                    task=call_goal,
                    reference_facts=reference_facts,
                )

                st.write("Calculating response-time and voice-delivery metrics...")
                interaction = calculate_interaction_metrics(
                    segments=segments,
                    agent_speaker=transcript_eval.agent_speaker,
                )
                convert_to_analysis_wav(original_path, wav_path)
                voice = calculate_voice_metrics(
                    wav_path=wav_path,
                    segments=segments,
                    agent_speaker=transcript_eval.agent_speaker,
                )

                report = build_report(
                    segments=segments,
                    transcript_eval=transcript_eval,
                    interaction_metrics=interaction,
                    voice_metrics=voice,
                )
                status.update(label="Evaluation complete", state="complete", expanded=False)

        assessment = report["assessment"].upper()
        if assessment == "PASS":
            st.success(f"{assessment}: {report['summary']}")
        elif assessment == "FAIL":
            st.error(f"{assessment}: {report['summary']}")
        else:
            st.warning(f"{assessment}: {report['summary']}")

        st.caption(
            f"Detected agent speaker: {report['detected_agent_speaker']} "
            f"({report['agent_speaker_confidence']:.0%} confidence)"
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Task completion", f"{report['task']['score']}%")
        col2.metric(
            "Median response",
            show_seconds(report["interaction"]["median_response_seconds"]),
        )
        col3.metric(
            "P95 response",
            show_seconds(report["interaction"]["p95_response_seconds"]),
        )
        col4.metric(
            "Agent interruptions",
            report["interaction"]["agent_interruption_count"],
        )

        st.subheader("Conversation")
        conv1, conv2, conv3, conv4 = st.columns(4)
        conv1.metric("Coherence", f"{report['conversation']['coherence_score']}/5")
        conv2.metric("Relevance", f"{report['conversation']['relevance_score']}/5")
        conv3.metric(
            "Context",
            f"{report['conversation']['context_retention_score']}/5",
        )
        conv4.metric(
            "No repetition",
            f"{report['conversation']['repetition_score']}/5",
        )

        st.subheader("Voice delivery")
        voice1, voice2, voice3, voice4 = st.columns(4)
        wpm = report["voice"]["speech_rate_wpm"]
        voice1.metric("Speech rate", "n/a" if wpm is None else f"{wpm:.0f} wpm")
        pitch = report["voice"]["pitch_variation_semitones"]
        voice2.metric("Pitch variation", "n/a" if pitch is None else f"{pitch:.2f} st")
        loudness = report["voice"]["loudness_variation_db"]
        voice3.metric(
            "Loudness variation",
            "n/a" if loudness is None else f"{loudness:.2f} dB",
        )
        voice4.metric("Modulation", report["voice"]["voice_modulation"])
        st.caption(report["voice"]["note"])

        st.subheader("Facts and required points")
        if report["facts"]["status"] == "not_checked":
            st.info(report["facts"]["note"])
        else:
            fact_score = report["facts"]["coverage_score"]
            st.write(
                f"**Coverage:** {'n/a' if fact_score is None else str(fact_score) + '%'}  "
                f"\n**Status:** {report['facts']['status']}"
            )
            st.write(report["facts"]["note"])
            if report["facts"]["missing_or_incorrect"]:
                for fact in report["facts"]["missing_or_incorrect"]:
                    st.write(f"- {fact}")

        st.subheader("Issues to review")
        issue_rows = []
        for issue in report["issues"]:
            issue_rows.append(
                {
                    "time": format_timestamp(issue["timestamp_seconds"]),
                    "severity": issue["severity"],
                    "category": issue["category"],
                    "what happened": issue["explanation"],
                    "evidence": issue["evidence"],
                }
            )

        for pause in report["interaction"]["long_pauses"]:
            issue_rows.append(
                {
                    "time": format_timestamp(pause["timestamp_seconds"]),
                    "severity": "medium",
                    "category": "response_time",
                    "what happened": f"Agent response gap was {pause['gap_seconds']:.2f} seconds.",
                    "evidence": "Calculated from speaker timestamps.",
                }
            )

        for interruption in report["interaction"]["agent_interruptions"]:
            issue_rows.append(
                {
                    "time": format_timestamp(interruption["timestamp_seconds"]),
                    "severity": "medium",
                    "category": "interruption",
                    "what happened": (
                        f"Agent speech overlapped the customer by approximately "
                        f"{interruption['overlap_seconds']:.2f} seconds."
                    ),
                    "evidence": "Calculated from speaker timestamps.",
                }
            )

        if issue_rows:
            st.dataframe(issue_rows, use_container_width=True, hide_index=True)
        else:
            st.write("No clear issues were flagged in this pass.")

        with st.expander("Speaker-attributed transcript"):
            for segment in report["transcript"]:
                st.markdown(
                    f"**{segment['speaker']} [{format_timestamp(segment['start'])}]:** "
                    f"{segment['text']}"
                )

        st.download_button(
            "Download JSON report",
            data=json.dumps(report, indent=2),
            file_name="voice_agent_evaluation.json",
            mime="application/json",
            use_container_width=True,
        )

        st.caption(
            "This is a post-call demo. It measures customer-perceived response gaps from the recording, "
            "not internal ASR, model, tool-call, or TTS latency."
        )

    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")
        st.exception(exc)
