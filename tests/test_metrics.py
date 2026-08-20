from voice_agent_eval import Segment, calculate_interaction_metrics, merge_adjacent_segments


def test_merges_adjacent_same_speaker_fragments() -> None:
    segments = [
        Segment("customer", 0.0, 1.0, "I need"),
        Segment("customer", 1.1, 2.0, "some help."),
        Segment("agent", 2.6, 3.5, "Sure."),
    ]

    merged = merge_adjacent_segments(segments)

    assert len(merged) == 2
    assert merged[0].text == "I need some help."


def test_response_gap_and_interruption_metrics() -> None:
    segments = [
        Segment("customer", 0.0, 2.0, "First question"),
        Segment("agent", 2.8, 4.0, "First answer"),
        Segment("customer", 5.0, 7.0, "Second question"),
        Segment("agent", 6.6, 8.0, "Interrupted answer"),
        Segment("customer", 9.0, 10.0, "Third question"),
        Segment("agent", 12.5, 13.0, "Slow answer"),
    ]

    metrics = calculate_interaction_metrics(segments, agent_speaker="agent")

    assert metrics["response_count"] == 2
    assert metrics["median_response_seconds"] == 1.65
    assert metrics["p95_response_seconds"] == 2.42
    assert metrics["agent_interruption_count"] == 1
    assert metrics["long_pause_count"] == 1
    assert "Customer turns are used only as timing anchors" in metrics["note"]
