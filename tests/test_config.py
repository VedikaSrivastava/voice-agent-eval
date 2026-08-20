from __future__ import annotations

from pathlib import Path

import pytest

from voice_agent_eval import normalize_reference_facts, resolve_api_key


def test_loads_api_key_from_env_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env.local"
    env_file.write_text("OPENAI_API_KEY=test-from-file\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert resolve_api_key(env_file=env_file) == "test-from-file"


def test_normalizes_reference_fact_list() -> None:
    assert normalize_reference_facts(["First fact", "", " Second fact "]) == (
        "First fact\nSecond fact"
    )


def test_explicit_api_key_takes_precedence(tmp_path: Path) -> None:
    env_file = tmp_path / ".env.local"
    env_file.write_text("OPENAI_API_KEY=from-file\n", encoding="utf-8")

    assert resolve_api_key("explicit-key", env_file=env_file) == "explicit-key"
