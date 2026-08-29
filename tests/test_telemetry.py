"""Observability that cannot break the app."""

from __future__ import annotations

import json

from rights_agent.telemetry import (
    CHAIN,
    RETRIEVER,
    SEMCONV,
    coerce_attribute,
    current_span,
    init_telemetry,
    set_retrieval_documents,
    shutdown_telemetry,
    span,
    telemetry_status,
)


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def test_disabled_tracing_reports_why_and_does_not_raise() -> None:
    assert init_telemetry("p", "http://localhost:6006", enabled=False) is False
    assert "disabled" in telemetry_status().status


def test_a_nonsense_endpoint_never_raises() -> None:
    """A missing package, a bad endpoint, a version mismatch inside Phoenix --
    none of them are reasons for the agent to stop answering."""
    init_telemetry("p", "not-a-url-at-all", enabled=False)
    assert isinstance(telemetry_status().status, str)


def test_shutdown_is_safe_without_a_provider() -> None:
    shutdown_telemetry(10)


# --------------------------------------------------------------------------- #
# The no-op shim
# --------------------------------------------------------------------------- #
def test_call_sites_never_branch_on_whether_tracing_is_on() -> None:
    """When tracing is off the shim absorbs everything, so instrumentation
    cannot drift out of sync with the code it describes."""
    ran = False
    with span("anything", RETRIEVER, extra=1) as current:
        ran = True
        current.set_input("q")
        current.set_output("a")
        current.set_attribute("x", object())
        current.set_attributes({"a": 1, "b": None})
        current.record_exception(RuntimeError("boom"))
        set_retrieval_documents(current, [{"id": "1", "score": 0.5, "content": "c"}])
        assert current.span_id() == "" or isinstance(current.span_id(), str)
    assert ran


def test_an_exception_inside_a_span_still_propagates() -> None:
    """Swallowing export failures must never mean swallowing app failures."""
    try:
        with span("failing", CHAIN):
            raise ValueError("real failure")
    except ValueError as exc:
        assert str(exc) == "real failure"
    else:  # pragma: no cover
        raise AssertionError("the exception was swallowed")


def test_current_span_is_always_usable() -> None:
    current_span().set_attribute("k", "v")


# --------------------------------------------------------------------------- #
# Attribute coercion
# --------------------------------------------------------------------------- #
def test_primitives_pass_through() -> None:
    assert coerce_attribute(1) == 1
    assert coerce_attribute(1.5) == 1.5
    assert coerce_attribute(True) is True
    assert coerce_attribute("text") == "text"


def test_homogeneous_sequences_are_kept_as_sequences() -> None:
    assert coerce_attribute(["a", "b"]) == ["a", "b"]
    assert coerce_attribute([1, 2]) == [1, 2]


def test_mixed_sequences_and_mappings_are_json_encoded() -> None:
    """OTel rejects heterogeneous sequences; JSON keeps the information."""
    assert json.loads(coerce_attribute([1, "a"])) == [1, "a"]
    assert json.loads(coerce_attribute({"k": "v"})) == {"k": "v"}


def test_oversized_values_are_truncated() -> None:
    """A trace UI is unusable if one attribute is a megabyte of context."""
    value = coerce_attribute("x" * 100_000)
    assert len(value) < 5_000 and value.endswith("...")


def test_objects_become_strings_rather_than_failing() -> None:
    assert isinstance(coerce_attribute(object()), str)


# --------------------------------------------------------------------------- #
# Semantic conventions
# --------------------------------------------------------------------------- #
def test_the_attribute_names_are_the_openinference_wire_format() -> None:
    """Literal fallbacks must match the package, or traces render differently
    depending on which extras are installed."""
    assert SEMCONV.SPAN_KIND == "openinference.span.kind"
    assert SEMCONV.INPUT_VALUE == "input.value"
    assert SEMCONV.OUTPUT_VALUE == "output.value"
    assert SEMCONV.RETRIEVAL_DOCUMENTS == "retrieval.documents"
    assert SEMCONV.LLM_TOKEN_COUNT_PROMPT == "llm.token_count.prompt"
    assert SEMCONV.SESSION_ID == "session.id"


def test_span_kinds_are_openinference_not_opentelemetry() -> None:
    """A different concept from OTel's SpanKind (SERVER/CLIENT/INTERNAL)."""
    assert {CHAIN, RETRIEVER} == {"CHAIN", "RETRIEVER"}
