from pathlib import Path

from voice_agent_eval.cli import _reference_facts, build_parser


def test_cli_parses_audio_goal_and_output() -> None:
    args = build_parser().parse_args(
        ["call.mp3", "--goal", "Resolve the request", "--output", "report.json"]
    )

    assert args.audio == Path("call.mp3")
    assert args.goal == "Resolve the request"
    assert args.output == Path("report.json")


def test_cli_combines_inline_and_file_facts(tmp_path: Path) -> None:
    facts_file = tmp_path / "facts.txt"
    facts_file.write_text("Second fact\n\nThird fact\n", encoding="utf-8")

    assert _reference_facts(["First fact"], facts_file) == [
        "First fact",
        "Second fact",
        "Third fact",
    ]
