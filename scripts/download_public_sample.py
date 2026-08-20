from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

PUBLIC_SAMPLE_URL = (
    "https://huggingface.co/datasets/AxonData/english-contact-center-audio-dataset/"
    "resolve/f89f1d3318ad1939c4cadf4d5dde06a2fb5badc5/"
    "Call%20center%20data%20samples%20EN/"
    "1755884171.51632%20%28EN%20Support-Billing%29/"
    "1755884171.51632.mp3?download=true"
)
PUBLIC_SAMPLE_SHA256 = "894959639d644cfb1022ae42c83a1d5f53afecae9e1dbdcdd7cb1ff4f881a348"
PUBLIC_SAMPLE_SIZE_BYTES = 836_208
PUBLIC_SAMPLE_LICENSE = "CC BY-NC 4.0"
PUBLIC_SAMPLE_SOURCE = (
    "https://huggingface.co/datasets/AxonData/english-contact-center-audio-dataset"
)


def download_public_sample(
    output_path: Path,
    *,
    url: str = PUBLIC_SAMPLE_URL,
    expected_sha256: str = PUBLIC_SAMPLE_SHA256,
    expected_size_bytes: int = PUBLIC_SAMPLE_SIZE_BYTES,
    timeout_seconds: float = 120.0,
) -> dict[str, object]:
    """Download the pinned public sample and verify it before use."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(f"{output_path.suffix}.part")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "voice-agent-eval-public-sample-preflight/0.1"},
    )
    digest = hashlib.sha256()
    size = 0

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            with partial_path.open("wb") as destination:
                while chunk := response.read(1024 * 1024):
                    destination.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)

        actual_sha256 = digest.hexdigest()
        if size != expected_size_bytes:
            raise ValueError(
                f"Unexpected public sample size: expected {expected_size_bytes}, got {size}."
            )
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "Unexpected public sample checksum: "
                f"expected {expected_sha256}, got {actual_sha256}."
            )

        partial_path.replace(output_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise

    return {
        "file": str(output_path),
        "size_bytes": size,
        "sha256": actual_sha256,
        "source": PUBLIC_SAMPLE_SOURCE,
        "license": PUBLIC_SAMPLE_LICENSE,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and verify the pinned public contact-center recording."
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = download_public_sample(args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
