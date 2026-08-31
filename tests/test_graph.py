"""The workflow: intent, routing, state and the stage-timing reducer."""

from __future__ import annotations

import pytest

from rights_agent.graph import (
    DEFAULT_INTENT,
    AgentState,
    _merge_ms,
    _route,
    classify_intent,
    initial_state,
    route_after_assess,
)


# --------------------------------------------------------------------------- #
# Intent
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("What does the document say about unfair dismissal?", "dismissal"),
        ("What does the document say about bereavement leave?", "leave"),
        ("When must an employer give notice of a shift?", "hours"),
        ("How are tips allocated between workers?", "pay"),
        ("What is the duty to prevent sexual harassment?", "harassment"),
        ("How does a trade union apply for recognition?", "unions"),
        ("What is a penalty notice?", "enforcement"),
        ("What is the short title?", DEFAULT_INTENT),
    ],
)
def test_intent_is_a_lookup_not_a_model_call(question: str, intent: str) -> None:
    """A model here would cost a round trip to produce a label a table gets
    right, and would make the label non-deterministic between eval runs."""
    assert classify_intent(question) == intent


def test_intent_precedence_puts_dismissal_before_pay() -> None:
    """"dismissal for refusing a variation" is a dismissal question."""
    assert classify_intent("dismissal for refusing a pay variation") == "dismissal"


def test_intent_is_case_insensitive() -> None:
    assert classify_intent("UNFAIR DISMISSAL") == "dismissal"


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def test_sufficient_retrieval_goes_straight_to_generate() -> None:
    assert _route(score=0.8, threshold=0.45, attempts=0, max_attempts=2) == "generate"


def test_thin_retrieval_refines_while_attempts_remain() -> None:
    assert _route(score=0.1, threshold=0.45, attempts=0, max_attempts=2) == "refine"
    assert _route(score=0.1, threshold=0.45, attempts=1, max_attempts=2) == "refine"


def test_thin_retrieval_refuses_once_attempts_are_exhausted() -> None:
    assert _route(score=0.1, threshold=0.45, attempts=2, max_attempts=2) == "refuse"


def test_a_score_exactly_on_the_threshold_generates() -> None:
    assert _route(score=0.45, threshold=0.45, attempts=0, max_attempts=2) == "generate"


def test_zero_max_attempts_refuses_immediately() -> None:
    assert _route(score=0.1, threshold=0.45, attempts=0, max_attempts=0) == "refuse"


def test_the_conditional_edge_reads_the_recorded_decision() -> None:
    """``assess`` records the route as a span attribute *and* in state, so the
    trace and the edge cannot disagree."""
    state: AgentState = {"route": "generate"}
    assert route_after_assess(state) == "generate"


def test_a_missing_route_refuses_rather_than_guessing() -> None:
    assert route_after_assess({}) == "refuse"


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
def test_initial_state_sets_every_field_explicitly() -> None:
    """Relying on ``total=False`` means a node reads ``None`` on the first pass
    and an int on the second -- a bug that only shows up under refinement."""
    state = initial_state("a question", session_id="s1", user_id="u1")
    for key in (
        "question",
        "session_id",
        "user_id",
        "rewritten_query",
        "intent",
        "route",
        "docs",
        "context",
        "answer",
        "citations",
        "attempts",
        "sufficiency",
        "refused",
        "scores",
        "stage_ms",
        "llm_stats",
        "index_version",
        "error",
    ):
        assert key in state, key
    assert state["rewritten_query"] == "a question"
    assert state["attempts"] == 0


def test_state_holds_only_plain_types() -> None:
    """The checkpointer serialises it; a dataclass in here fails later, not now."""
    for value in initial_state("q").values():
        assert isinstance(value, (str, int, float, bool, list, dict)), type(value)


# --------------------------------------------------------------------------- #
# Stage timings
# --------------------------------------------------------------------------- #
def test_stage_timings_accumulate_across_the_refine_loop() -> None:
    """``retrieve`` runs again after a refinement; the honest total is the sum."""
    merged = _merge_ms({"retrieve": 10.0}, {"retrieve": 4.0, "refine": 1.0})
    assert merged == {"retrieve": 14.0, "refine": 1.0}


def test_merge_handles_the_first_write() -> None:
    assert _merge_ms(None, {"classify": 0.5}) == {"classify": 0.5}
    assert _merge_ms({"classify": 0.5}, None) == {"classify": 0.5}
    assert _merge_ms(None, None) == {}


# --------------------------------------------------------------------------- #
# LangGraph drops updates for undeclared keys
# --------------------------------------------------------------------------- #
def test_every_key_a_node_writes_is_declared_in_the_state() -> None:
    """LangGraph **silently drops** an update whose key the schema does not
    declare. An undeclared field looks like it is being written and is not, and
    the consumer quietly falls back to something else -- which is how
    ``model`` came to record the configured model rather than the one that
    answered. This scans the node bodies so the next such field fails here.
    """
    import re
    from pathlib import Path

    source = Path(__import__("rights_agent.graph", fromlist=["x"]).__file__).read_text()
    written = set(re.findall(r'updates\[\s*"([a-z_]+)"\s*\]\s*=', source))
    written |= {
        key
        for group in re.findall(r"updates\.update\(([^)]*)\)", source, re.DOTALL)
        for key in re.findall(r"^\s*([a-z_]+)\s*=", group, re.MULTILINE)
    }
    declared = set(AgentState.__annotations__)
    assert written <= declared, f"nodes write keys the state does not declare: {sorted(written - declared)}"


def test_initial_state_covers_every_declared_key() -> None:
    """The other half of the same trap: a field read before any node writes it
    is ``None`` on the first pass and typed on the second."""
    assert set(initial_state("q")) == set(AgentState.__annotations__)
