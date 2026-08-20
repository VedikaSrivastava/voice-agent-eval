from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace


def test_streamlit_app_imports_without_starting_an_evaluation(monkeypatch: object) -> None:
    fake_streamlit = SimpleNamespace(
        set_page_config=lambda **_kwargs: None,
        title=lambda *_args, **_kwargs: None,
        write=lambda *_args, **_kwargs: None,
        file_uploader=lambda *_args, **_kwargs: None,
        text_input=lambda *_args, **_kwargs: "",
        text_area=lambda *_args, **_kwargs: "",
        button=lambda *_args, **_kwargs: False,
    )
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    sys.modules.pop("app", None)

    importlib.import_module("app")
