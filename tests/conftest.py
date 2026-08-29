"""Fixtures for the unit tests.

These tests do not need an index.  Anything that does belongs in ``evals/``,
where the missing-index failure message tells you which command to run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from rights_agent.config import Settings, reload_settings
from rights_agent.store import reset_client_cache

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """Settings whose artefacts land in ``tmp_path``."""
    monkeypatch.setenv("RIGHTS_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("RIGHTS_TRACING", "false")
    reset_client_cache()
    yield reload_settings()
    reset_client_cache()
    reload_settings()


@pytest.fixture(scope="session")
def corpus_text() -> str:
    from rights_agent.tools.corpus import render

    return render()
