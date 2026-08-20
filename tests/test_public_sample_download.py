from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import pytest

from scripts import download_public_sample as sample_download


class _Response(BytesIO):
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def test_download_public_sample_verifies_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    content = b"public-contact-center-audio"
    expected_hash = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(
        sample_download.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(content),
    )

    output = tmp_path / "sample.mp3"
    result = sample_download.download_public_sample(
        output,
        url="https://example.invalid/sample.mp3",
        expected_sha256=expected_hash,
        expected_size_bytes=len(content),
    )

    assert output.read_bytes() == content
    assert result["sha256"] == expected_hash
    assert result["size_bytes"] == len(content)


def test_download_public_sample_removes_partial_file_on_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sample_download.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(b"unexpected"),
    )

    output = tmp_path / "sample.mp3"
    with pytest.raises(ValueError, match="Unexpected public sample size"):
        sample_download.download_public_sample(
            output,
            url="https://example.invalid/sample.mp3",
            expected_sha256="0" * 64,
            expected_size_bytes=999,
        )

    assert not output.exists()
    assert not (tmp_path / "sample.mp3.part").exists()
