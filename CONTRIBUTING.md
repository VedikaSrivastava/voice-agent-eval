# Contributing

Issues and pull requests are welcome, especially for reproducible call examples, metric calibration, provider integrations, and report improvements.

## Local setup on Windows

```bat
git clone https://github.com/VedikaSrivastava/voice-agent-eval.git
cd voice-agent-eval
py -m venv .venv
.venv\Scripts\activate
py -m pip install --upgrade pip
pip install -e ".[app,dev]"
```

Create `.env.local` only when a live API-backed evaluation is required. Unit tests do not require an API key.

## Before opening a pull request

```bat
pytest -q
py -m compileall -q app.py src scripts
py -m build
voice-agent-eval --help
```

Keep sample recordings out of the repository unless their license and speaker consent clearly permit redistribution. Prefer a small script that downloads a versioned public sample and verifies its checksum.
