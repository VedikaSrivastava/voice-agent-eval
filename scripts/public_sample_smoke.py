from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from pydub import AudioSegment

from voice_agent_eval import Segment, calculate_voice_metrics, convert_to_analysis_wav


def inspect_public_sample(audio_path: Path, probe_seconds: float = 45.0) -> dict[str, object]:
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio = AudioSegment.from_file(audio_path)
    duration_seconds = len(audio) / 1000.0
    if duration_seconds <= 1.0:
        raise ValueError("The public sample is too short to validate audio preprocessing.")

    inspected_seconds = min(duration_seconds, probe_seconds)
    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = Path(temp_dir) / "sample.wav"
        convert_to_analysis_wav(audio_path, wav_path)
        acoustic = calculate_voice_metrics(
            wav_path=wav_path,
            segments=[Segment("probe", 0.0, inspected_seconds, "")],
            agent_speaker="probe",
        )

    if acoustic["pitch_variation_semitones"] is None:
        raise RuntimeError("Pitch extraction returned no result for the public sample.")
    if acoustic["loudness_variation_db"] is None:
        raise RuntimeError("Loudness extraction returned no result for the public sample.")

    return {
        "file": audio_path.name,
        "sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        "file_size_bytes": audio_path.stat().st_size,
        "duration_seconds": round(duration_seconds, 2),
        "sample_rate_hz": audio.frame_rate,
        "channels": audio.channels,
        "sample_width_bytes": audio.sample_width,
        "probe_seconds": round(inspected_seconds, 2),
        "acoustic_preflight": {
            "pitch_variation_semitones": acoustic["pitch_variation_semitones"],
            "loudness_variation_db": acoustic["loudness_variation_db"],
            "clipping_ratio": acoustic["clipping_ratio"],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decode a public conversation recording and exercise the package's audio "
            "conversion and acoustic-feature path without calling an external API."
        )
    )
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe-seconds", type=float, default=45.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = inspect_public_sample(args.audio, probe_seconds=args.probe_seconds)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
