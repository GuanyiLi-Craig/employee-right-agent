"""The tool surface: one dispatcher, and no path around it.

**"The agent must have no path to a tool that does not pass through the broker"
is a claim about the code, not about intentions.**  Two tests here check it
structurally, because a reviewer reading a diff cannot.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from attacklab import tools
from attacklab.context import request_context
from attacklab.registry import Registry
from attacklab.tools import (
    ACCIDENTAL_TOOL,
    REVIEWED_ALLOWLIST,
    RUNTIME_SURFACE,
    dispatch,
    dispatch_from_answer,
    extract_tool_calls,
    format_tool_call,
    strip_tool_calls,
)
from rights_agent import hooks
from rights_agent.hooks import Decision, NullHooks, Principal

TOOLS_SOURCE = Path(tools.__file__)
ALICE = Principal("alice@example.gov.uk", frozenset({"employee"}))


@pytest.fixture(autouse=True)
def _null_hooks():
    hooks.reset()
    yield
    hooks.reset()


# --------------------------------------------------------------------------- #
# The structural claims
# --------------------------------------------------------------------------- #
def test_no_tool_implementation_is_public() -> None:
    """Every implementation is module-private and reachable only via dispatch."""
    public = [name for name in tools.__all__ if not name.startswith("_")]
    assert "_stub" not in public
    for name in RUNTIME_SURFACE:
        assert name.name not in public, (
            f"{name.name} is exposed as a module attribute, so the gate can be skipped"
        )


def test_dispatch_is_the_only_caller_of_the_implementation() -> None:
    """Parsed, not grepped: a second call site would be a path around the gate."""
    tree = ast.parse(TOOLS_SOURCE.read_text(encoding="utf-8"))
    callers: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "_stub"
            ):
                callers.append(node.name)
    assert callers == ["dispatch"], (
        f"_stub is called from {callers}; only dispatch may reach it, because only "
        "dispatch consults the hook first"
    )


def test_the_hook_is_consulted_before_the_tool_runs() -> None:
    """Authorise, then act.  The order is the whole design."""
    order: list[str] = []

    class _Recording(NullHooks):
        def on_tool_call(self, name, params, principal):  # type: ignore[no-untyped-def]
            order.append("hook")
            return Decision.allow()

    hooks.install(_Recording())
    result = dispatch("search_document", {"query": "x"}, ALICE)
    order.append("dispatched")
    assert order == ["hook", "dispatched"]
    assert result.outcome == "allow"


def test_a_denied_call_never_reaches_the_implementation() -> None:
    class _Denies(NullHooks):
        def on_tool_call(self, name, params, principal):  # type: ignore[no-untyped-def]
            return Decision.deny("no grant", control="tool_broker")

    hooks.install(_Denies())
    result = dispatch("send_notification", {"to": "x@y.example.net"}, ALICE)
    assert result.outcome == "deny"
    assert "not executed" in result.detail


def test_a_needs_approval_call_never_reaches_the_implementation() -> None:
    class _Asks(NullHooks):
        def on_tool_call(self, name, params, principal):  # type: ignore[no-untyped-def]
            return Decision.needs_approval_for("irreversible")

    hooks.install(_Asks())
    result = dispatch("update_employee_record", {}, ALICE)
    assert result.outcome == "needs_approval"
    assert "not executed" in result.detail


def test_every_attempt_is_recorded_whichever_way_it_went() -> None:
    """A denied tool call is your best injection detector, so it has to be kept."""
    with request_context(Registry()) as ctx:
        dispatch("search_document", {"query": "x"}, ALICE, ctx)
        assert len(ctx.tools) == 1
        assert ctx.tools[0].outcome == "allow"


# --------------------------------------------------------------------------- #
# The marker format
# --------------------------------------------------------------------------- #
def test_the_marker_does_not_look_like_a_citation() -> None:
    """It used to, and it cost demo 1 a metric.

    The first version used ``[[tool: ...]]`` and the assistant's citation regex
    matched it, so every run reported the marker as an extra citation and
    citation coverage scored the demo-1 answer at 0.8 instead of 1.0. Demo 1's
    whole argument is that *every* metric reported success.
    """
    from rights_agent.llm import extract_citations

    marker = format_tool_call("send_notification", {"to": "x@y.example.net"})
    assert extract_citations(f"Some answer [s.19] provides text. {marker}") == ["s.19"]


def test_a_marker_round_trips() -> None:
    marker = format_tool_call("send_notification", {"to": "a@b.example.net", "body": "hi"})
    assert extract_tool_calls(marker) == [
        ("send_notification", {"to": "a@b.example.net", "body": "hi"})
    ]


def test_stripping_leaves_what_a_person_reads() -> None:
    answer = f"Leave is two weeks. {format_tool_call('send_notification', {'to': 'a@b.example.net'})}"
    assert strip_tool_calls(answer) == "Leave is two weeks."


def test_several_markers_dispatch_in_order() -> None:
    answer = (
        format_tool_call("search_document", {"query": "a"})
        + " "
        + format_tool_call("submit_leave_request", {"start": "x", "days": "1", "reason": "y"})
    )
    results = dispatch_from_answer(answer, ALICE)
    assert [r.name for r in results] == ["search_document", "submit_leave_request"]


# --------------------------------------------------------------------------- #
# The surface itself
# --------------------------------------------------------------------------- #
def test_the_accidental_tool_is_in_the_runtime_surface_and_not_the_allowlist() -> None:
    """CVE-2026-25592, as a fixture.  The sandbox held; the tool surface did not."""
    assert ACCIDENTAL_TOOL.name in {spec.name for spec in RUNTIME_SURFACE}
    assert ACCIDENTAL_TOOL.name not in REVIEWED_ALLOWLIST


def test_nothing_leaves_the_process() -> None:
    """The lab's first rule, asserted over the source.

    No implementation may open a socket, read a file or start a process. There
    is exactly one implementation and it returns a fixed string -- twelve
    different fakes would invite somebody to make one of them real.
    """
    forbidden = {"urlopen", "requests", "socket", "subprocess", "smtplib", "httpx"}
    tree = ast.parse(TOOLS_SOURCE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & forbidden), (
        f"{sorted(imported & forbidden)} imported into the tool module. No payload in "
        "this lab does anything: the point of every demonstration is the attempt."
    )


def test_deny_by_default_has_two_distinct_shapes() -> None:
    """A tool can be unreachable two ways, and both matter.

    Some name **no role at all**: nothing can ever mint a grant for them, which
    is the strongest form. ``fetch_url`` and ``read_file`` are the July 2026
    pair -- the same effect by two mechanisms, so scoping by mechanism would
    have covered one and missed the other.

    ``export_audit_log`` is the other shape: it names a role (``auditor``) that
    **no persona in the lab holds**. That is the ordinary production case, and
    it is the one worth having a fixture for, because it is the case where a
    reviewer reading the tool table sees a role and assumes somebody has it.
    """
    from attacklab.controls.tool_broker import PERSONAS

    ungranted = {spec.name for spec in RUNTIME_SURFACE if not spec.roles}
    assert ungranted == {
        "download_file_to_host",
        "fetch_url",
        "read_file",
        "run_snippet",
    }

    held_roles: set[str] = set()
    for principal in PERSONAS.values():
        held_roles |= principal.roles
    unheld = {
        spec.name
        for spec in RUNTIME_SURFACE
        if spec.roles and not (spec.roles & held_roles)
    }
    assert unheld == {"export_audit_log"}
