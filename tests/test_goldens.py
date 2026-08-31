"""The dataset generator, and what it is allowed to overwrite."""

from __future__ import annotations

import json
import re
from pathlib import Path

from rights_agent.tools.goldens import (
    DEFAULT_QUALITY_THRESHOLDS,
    GoldenRow,
    write_baseline,
)


def test_regenerating_rewrites_known_failures_only(tmp_path: Path) -> None:
    """A threshold, and the measurement behind it, are human judgements. The
    generator rebuilt the file from scratch, which deleted the record of why a
    gate was where it was -- so the next change to it had nothing to argue
    against."""
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps(
            {
                "known_failures": ["g001"],
                "quality_thresholds": {"answer_relevance_mean": 0.35},
                "note": "a note someone wrote on purpose",
                "observed_when_set": {
                    "index_version": "parser-5+hashing-bow-512+bc461767",
                    "answer_relevance": {"mean": 0.4281},
                },
            }
        )
    )

    payload = write_baseline(
        path,
        [
            GoldenRow(id="g002", question="q", intent="lookup", known_failure=True),
            GoldenRow(id="g003", question="q", intent="lookup"),
        ],
    )

    assert payload["known_failures"] == ["g002"]
    assert payload["quality_thresholds"] == {"answer_relevance_mean": 0.35}
    assert payload["note"] == "a note someone wrote on purpose"
    assert payload["observed_when_set"]["index_version"] == "parser-5+hashing-bow-512+bc461767"


def test_a_first_baseline_gets_the_defaults(tmp_path: Path) -> None:
    payload = write_baseline(
        tmp_path / "baseline.json",
        [GoldenRow(id="g001", question="q", intent="lookup")],
    )
    assert payload["known_failures"] == []
    assert payload["quality_thresholds"] == DEFAULT_QUALITY_THRESHOLDS


def test_every_threshold_the_gate_reads_has_a_default() -> None:
    """A fresh baseline must not KeyError the first time the gate runs."""
    source = (Path("evals") / "test_quality.py").read_text()
    gate_keys = set(re.findall(r"""thresholds\[["']([a-z0-9_]+)["']\]""", source))
    assert gate_keys, "no threshold lookups found; has the gate moved?"
    assert gate_keys <= set(DEFAULT_QUALITY_THRESHOLDS), (
        f"the gate reads thresholds with no default: {sorted(gate_keys - set(DEFAULT_QUALITY_THRESHOLDS))}"
    )
