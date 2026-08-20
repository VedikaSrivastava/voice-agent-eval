from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .pipeline import evaluate_call


def _reference_facts(facts: list[str], facts_file: Path | None) -> list[str]:
    values = [fact.strip() for fact in facts if fact.strip()]
    if facts_file is not None:
        values.extend(
            line.strip()
            for line in facts_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice-agent-eval",
        description="Evaluate a recorded voice agent call and write a JSON report.",
    )
    parser.add_argument("audio", type=Path, help="Path to an MP3, WAV, M4A, MP4, or WebM recording.")
    parser.add_argument(
        "--goal",
        default="",
        help="What the agent was expected to accomplish during the call.",
    )
    parser.add_argument(
        "--fact",
        action="append",
        default=[],
        help="Reference fact or required talking point. Repeat this option for multiple facts.",
    )
    parser.add_argument(
        "--facts-file",
        type=Path,
        help="Text file containing one reference fact or required point per line.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to write the JSON report. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env.local"),
        help="Path to the local environment file. Defaults to .env.local.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        facts = _reference_facts(args.fact, args.facts_file)
        report = evaluate_call(
            args.audio,
            task=args.goal,
            reference_facts=facts,
            env_file=args.env_file,
        )
        rendered = json.dumps(report, indent=2)
        if args.output is None:
            print(rendered)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
            print(f"Wrote evaluation report to {args.output}")
        return 0
    except Exception as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
