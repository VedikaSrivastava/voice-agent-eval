from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from voice_agent_eval import MAX_AUDIO_BYTES, VoiceAgentEvaluator, format_timestamp

st.set_page_config(page_title="Voice Agent Eval", page_icon="🎧", layout="centered")


def show_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}s"


st.title("Voice Agent Eval")
st.write(
    "Upload a recorded customer-agent call and review how the agent handled it. "
    "Customer speech is used as context for what was asked and as a timing reference; the customer is not scored."
)

uploaded_file = st.file_uploader(
    "Call recording",
    type=["mp3", "wav", "m4a", "mp4", "webm"],
    help="Maximum file size for this prototype: 25 MB.",
)

call_goal = st.text_input(
    "What was the agent supposed to accomplish?",
    placeholder="For example: answer the request and confirm the next step",
)

reference_facts = st.text_area(
    "Facts or required talking points (optional)",
    placeholder=(
        "Add one fact or required point per line. "
        "Without a source of truth, factual correctness is left as not checked."
    ),
    height=120,
)

if uploaded_file is not None:
    st.audio(uploaded_file)

run_clicked = st.button(
    "Evaluate agent",
    type="primary",
    disabled=uploaded_file is None,
    use_container_width=True,
)

if run_clicked and uploaded_file is not None:
    if uploaded_file.size > MAX_AUDIO_BYTES:
        st.error("This prototype accepts audio files up to 25 MB.")
        st.stop()

    suffix = Path(uploaded_file.name).suffix.lower() or ".mp3"

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / f"call{suffix}"
            audio_path.write_bytes(uploaded_file.getvalue())

            with st.status("Evaluating the agent", expanded=True) as status:
                evaluator = VoiceAgentEvaluator(env_file=".env.local")
                report = evaluator.evaluate(
                    audio_path,
                    task=call_goal,
                    reference_facts=reference_facts,
                    progress=st.write,
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
            f"({report['agent_speaker_confidence']:.0%} confidence). "
            "Only the agent is evaluated."
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
        st.caption(report["task"]["reason"])

        st.subheader("Customer request to agent response")
        response = report["agent_response"]
        if response["turn_reviews"]:
            turn_rows = []
            for review in response["turn_reviews"]:
                turn_rows.append(
                    {
                        "customer time": format_timestamp(review["customer_timestamp_seconds"]),
                        "customer asked": review["customer_request"],
                        "agent time": format_timestamp(review["agent_timestamp_seconds"]),
                        "agent responded": review["agent_response"],
                        "outcome": review["outcome"],
                        "score": review["score"],
                        "note": review["note"],
                    }
                )
            st.dataframe(turn_rows, use_container_width=True, hide_index=True)
        else:
            st.write("No substantive customer requests were identified for turn-by-turn review.")

        st.subheader("Agent response quality")
        conv1, conv2, conv3, conv4, conv5 = st.columns(5)
        conv1.metric("Answers request", f"{response['alignment_score']}/5")
        conv2.metric("Coherence", f"{response['coherence_score']}/5")
        conv3.metric("Relevance", f"{response['relevance_score']}/5")
        conv4.metric("Context", f"{response['context_retention_score']}/5")
        conv5.metric("No repetition", f"{response['repetition_score']}/5")

        st.subheader("Agent voice delivery")
        voice1, voice2, voice3, voice4, voice5 = st.columns(5)
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
        clipping = report["voice"]["clipping_ratio"]
        voice5.metric(
            "Clipping",
            "n/a" if clipping is None else f"{100 * clipping:.3f}%",
        )
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
                    "evidence": "Calculated from customer-to-agent speaker timestamps.",
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
            st.write("No clear agent issues were flagged in this pass.")

        with st.expander("Customer and agent transcript"):
            for segment in report["transcript"]:
                role = segment["role"].title()
                st.markdown(
                    f"**{role} [{format_timestamp(segment['start'])}]:** "
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
            "This is a post-call evaluator. It measures the delay heard in the recording, "
            "not internal ASR, model, tool-call, or TTS latency."
        )

    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")
        st.caption("Check the terminal output for the full error details.")
