"""Constants the session 6 gate shares.

The *fixtures* live in ``conftest.py``, where pytest finds them without an
import -- importing a fixture shadows the test's own parameter name, which is a
lint error in this repository and a readability problem in any repository.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EVALS_DIR = Path(__file__).parent
THRESHOLDS_PATH = EVALS_DIR / "thresholds.json"
FALSEPOS_PATH = EVALS_DIR / "falsepos.jsonl"
ADVERSARIAL_PATH = EVALS_DIR / "adversarial.jsonl"

MISSING_TWINS = (
    "the attack lab's indexes have not been built. They are a separate step, and "
    "the poison is ingested through the assistant's unmodified pipeline:\n"
    "    make poison        # the hostile document included\n"
    "    make clean-index   # its clean twin\n"
    "or, with Docker:\n"
    "    docker compose run --rm ingest -- --with-poison"
)


def read_thresholds() -> dict[str, Any]:
    return json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
