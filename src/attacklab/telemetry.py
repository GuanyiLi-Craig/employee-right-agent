"""Attack and control spans, emitted into the same Phoenix the assistant uses.

That join is the point: your security telemetry becomes the telemetry you
already built, and "which controls were on for this trace" is a span attribute
rather than a note in a runbook.

Three conventions:

* one ``attack`` span per catalogue run, parent of the assistant's normal trace;
* one ``control`` span per control invocation, child of the phase it guards;
* **a denied tool call gets a span marked as an error**, so it surfaces in
  Phoenix's error view with no custom query.  That is the mechanism behind "a
  denied call is your best injection detector" — one click on stage.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from rights_agent.telemetry import CHAIN, SEMCONV, span

if TYPE_CHECKING:  # pragma: no cover - types only
    from attacklab.controls.base import ControlResult


@contextmanager
def attack_span(
    payload_id: str,
    *,
    vector: str,
    goal: str,
    owasp: str,
    asi: str | None,
    controls: str,
    question: str = "",
) -> Iterator[Any]:
    """The parent span for one catalogue run."""
    with span(
        "attacklab.attack",
        CHAIN,
        **{
            SEMCONV.INPUT_VALUE: question,
            "attack.payload_id": payload_id,
            "attack.vector": vector,
            "attack.goal": goal,
            "attack.owasp": owasp,
            "attack.asi": asi or "",
            "attacklab.controls": controls,
        },
    ) as current:
        yield current


def emit_control_span(
    key: str,
    layer: int,
    deterministic: bool,
    result: ControlResult,
    controls: str,
) -> None:
    """One span per control invocation.

    Emitted even when the control passed: a panel that only shows blocks cannot
    report a false-positive rate, and a block rate without a false-positive rate
    is a marketing number.
    """
    with span(
        f"attacklab.control.{key}",
        CHAIN,
        **{
            "control.key": key,
            "control.layer": layer,
            "control.deterministic": deterministic,
            "control.passed": result.passed,
            "control.reason": result.reason,
            "control.latency_ms": result.latency_ms,
            "control.cost_usd": result.cost_usd,
            "control.tokens_in": result.tokens_in,
            "control.tokens_out": result.tokens_out,
            "attacklab.controls": controls,
        },
    ) as current:
        if not result.passed:
            current.set_attribute(SEMCONV.OUTPUT_VALUE, f"blocked: {result.reason}")


def emit_tool_span(
    name: str,
    params: dict[str, Any],
    outcome: str,
    reason: str,
    principal: str,
    controls: str,
) -> None:
    """One span per tool call, marked as an error when denied.

    ``error=True`` is what puts it in Phoenix's error view.  The alternative --
    a custom query over an attribute -- works and nobody runs it.
    """
    with span(
        f"attacklab.tool.{name}",
        "TOOL",
        **{
            SEMCONV.INPUT_VALUE: repr(params),
            "tool.name": name,
            "tool.outcome": outcome,
            "tool.reason": reason,
            "tool.principal": principal,
            "attacklab.controls": controls,
        },
    ) as current:
        if outcome != "allow":
            current.set_attribute("error", True)
            current.set_attribute("error.type", f"tool_{outcome}")
            current.set_attribute("error.message", reason)
            current.set_attribute(SEMCONV.OUTPUT_VALUE, f"{outcome}: {reason}")
