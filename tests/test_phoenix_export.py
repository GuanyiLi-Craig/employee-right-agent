"""Shaping the golden set for Phoenix, without needing a Phoenix.

The upload path is wrapped in a blanket ``except`` on purpose -- the experiments
API surface moves between releases and the lesson has to survive without the UI.
That makes it exactly the code most likely to be silently broken, so the parts
that can be checked offline are checked: the example shape, the tags, and the
defensive readers.
"""

from __future__ import annotations

import pytest

from rights_agent.config import PROMPT_VERSION
from rights_agent.tools.evaluate import (
    _browsable,
    _evaluator_citation_coverage,
    _evaluator_groundedness,
    _example_input,
    _field,
)


# --------------------------------------------------------------------------- #
# Reading whatever the client returned
# --------------------------------------------------------------------------- #
def test_fields_are_read_from_mappings_and_objects_alike() -> None:
    """``RanExperiment`` is a mapping in this release and was an object before."""
    assert _field({"experiment_id": "e1"}, "experiment_id") == "e1"
    assert _field(type("X", (), {"experiment_id": "e2"})(), "experiment_id") == "e2"
    assert _field({}, "missing") is None


def test_example_input_survives_either_spelling() -> None:
    assert _example_input({"input": {"question": "q"}}) == {"question": "q"}
    assert _example_input({"inputs": {"question": "q"}}) == {"question": "q"}
    assert _example_input(type("E", (), {"input": {"question": "q"}})()) == {"question": "q"}
    assert _example_input(None) == {}


def test_evaluators_read_the_scores_the_task_returned() -> None:
    output = {"groundedness": 0.75, "citation_coverage": 1.0}
    assert _evaluator_groundedness(output=output) == 0.75
    assert _evaluator_citation_coverage(output=output) == 1.0


def test_evaluators_return_zero_rather_than_raising_on_a_bad_output() -> None:
    """A task that failed must not take the whole experiment down with it."""
    assert _evaluator_groundedness(output=None) == 0.0
    assert _evaluator_citation_coverage(output="not a dict") == 0.0


# --------------------------------------------------------------------------- #
# The link
# --------------------------------------------------------------------------- #
def test_a_container_only_host_becomes_a_path() -> None:
    """Handing a presenter ``http://phoenix:6006/...`` wastes a minute on stage."""
    out = _browsable("http://phoenix:6006/datasets/D/compare?experimentId=E")
    assert out == "<your Phoenix UI>/datasets/D/compare?experimentId=E"


def test_a_reachable_host_is_left_alone() -> None:
    url = "http://localhost:6006/datasets/D/compare?experimentId=E"
    assert _browsable(url) == url


# --------------------------------------------------------------------------- #
# The example shape the server actually requires
# --------------------------------------------------------------------------- #
def build_examples(rows, index_version: str) -> list[dict[str, object]]:
    """Mirror of the shaping in ``push_to_phoenix``, exercised here directly."""
    return [
        {
            "input": {"question": str(row["question"])},
            "output": {
                "expected_citations": ", ".join(row.get("must_cite") or []),
                "should_refuse": bool(row.get("should_refuse")),
            },
            "metadata": {
                "id": str(row["id"]),
                "intent": str(row.get("intent", "")),
                "known_failure": bool(row.get("known_failure")),
                "note": str(row.get("note", "")),
                "index_version": index_version,
                "prompt_version": PROMPT_VERSION,
            },
        }
        for row in rows
    ]


@pytest.fixture
def golden_rows() -> list[dict[str, object]]:
    """Any committed golden set: this suite asserts on the *shape* of what is
    uploaded to Phoenix, which is the same whichever embedder generated it."""
    import json
    from pathlib import Path

    from rights_agent.datasets import available

    evals = Path(__file__).resolve().parents[1] / "evals"
    names = available(evals)
    if not names:
        pytest.skip("no eval datasets committed")
    path = evals / "datasets" / names[0] / "golden.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_every_example_is_nested_not_flat(golden_rows) -> None:
    """The ``*_keys`` arguments belong to the dataframe and CSV paths; flat rows
    passed with them are rejected outright by the server."""
    for example in build_examples(golden_rows, "v1"):
        assert set(example) == {"input", "output", "metadata"}
        assert set(example["input"]) == {"question"}
        assert example["input"]["question"]


def test_expected_citations_travel_with_the_example(golden_rows) -> None:
    examples = build_examples(golden_rows, "v1")
    answerable = [
        e for e, r in zip(examples, golden_rows)
        if r.get("must_cite") and not r.get("should_refuse")
    ]
    assert answerable
    assert all(e["output"]["expected_citations"] for e in answerable)


def test_refusal_rows_are_marked_as_such(golden_rows) -> None:
    examples = build_examples(golden_rows, "v1")
    refusals = [e for e in examples if e["output"]["should_refuse"]]
    assert len(refusals) == sum(1 for r in golden_rows if r.get("should_refuse"))


def test_every_example_is_tagged_with_the_index_and_prompt_version(golden_rows) -> None:
    """Tag each experiment with both, or two runs are not comparable."""
    for example in build_examples(golden_rows, "parser-3+x+abcd1234"):
        assert example["metadata"]["index_version"] == "parser-3+x+abcd1234"
        assert example["metadata"]["prompt_version"] == PROMPT_VERSION


def test_known_failures_survive_the_round_trip(golden_rows) -> None:
    """Otherwise a Phoenix experiment silently counts them as regressions."""
    examples = build_examples(golden_rows, "v1")
    flagged = sum(1 for e in examples if e["metadata"]["known_failure"])
    assert flagged == sum(1 for r in golden_rows if r.get("known_failure"))
    assert flagged > 0
