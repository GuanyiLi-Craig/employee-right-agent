"""The per-request context every control reads.

One object, created at request start, holding the registry snapshot and
everything the controls record.  It exists for one reason from §5 of the spec:
**a toggle flipped mid-request must not produce a half-defended run.**  Controls
never read the registry; they read this.

Carried on a :class:`~contextvars.ContextVar` rather than threaded through the
assistant's call signatures, matching how the assistant already carries its
token sink.  The graph's node signatures belong to the workflow, not to whatever
happens to be watching it, and a context variable is correct under the console's
thread pool.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from attacklab.registry import Registry, snapshot_attribute
from rights_agent.hooks import Principal

if TYPE_CHECKING:  # pragma: no cover - types only
    from attacklab.controls.base import ControlResult


@dataclass(slots=True)
class ControlEvent:
    """What one control did, once, on this request."""

    key: str
    layer: int
    deterministic: bool
    hook: str
    passed: bool
    #: Whether this control *changed* the value it was given.
    #:
    #: Recorded because a rewrite can contain a payload without ever blocking
    #: it -- PII masking removes the destination, so the model never assembles
    #: the call, so nothing is denied and ``blocked_by`` stays empty. Without
    #: this the panel reported "contained" and named no control, which reads as
    #: luck rather than as the masking layer doing its job.
    rewrote: bool = False
    reason: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    #: The offending text, so the console can highlight it in place.
    #: Highlighting is what makes the demo legible from the back of the room.
    matched: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "layer": self.layer,
            "deterministic": self.deterministic,
            "hook": self.hook,
            "passed": self.passed,
            "rewrote": self.rewrote,
            "reason": self.reason,
            "latency_ms": round(self.latency_ms, 3),
            "cost_usd": round(self.cost_usd, 8),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "matched": self.matched,
        }


@dataclass(slots=True)
class ToolAttempt:
    """One tool call the agent tried to make, and what the gate said."""

    name: str
    params: dict[str, Any]
    outcome: str
    reason: str = ""
    parameter: str = ""
    grant_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": dict(self.params),
            "outcome": self.outcome,
            "reason": self.reason,
            "parameter": self.parameter,
            "grant_id": self.grant_id,
        }


@dataclass(slots=True)
class RequestContext:
    """One request's view of the lab."""

    controls: dict[str, bool]
    principal: Principal
    route: str = "ask"
    #: Per-request fence delimiter.  A fixed delimiter can be forged by an
    #: attacker who has read your source; this one cannot be known in advance.
    nonce: str = field(default_factory=lambda: secrets.token_hex(8))
    payload_id: str = ""
    events: list[ControlEvent] = field(default_factory=list)
    tools: list[ToolAttempt] = field(default_factory=list)
    #: Ids of retrieved blocks some control flagged as instruction-shaped.
    flagged_ids: set[str] = field(default_factory=set)
    #: token → original value, for the PII vault.  Never logged, never traced.
    pii_map: dict[str, str] = field(default_factory=dict)
    pii_classes: dict[str, int] = field(default_factory=dict)
    #: The system + user strings actually sent, so the console can show the fence.
    sent_system: str = ""
    sent_user: str = ""
    #: The tool list the model was offered, before and after minting.
    tools_offered: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def get(self, key: str) -> bool:
        return bool(self.controls.get(key, False))

    @property
    def controls_attribute(self) -> str:
        return snapshot_attribute(self.controls)

    def record(self, result: ControlResult, *, hook: str) -> None:
        self.events.append(
            ControlEvent(
                key=result.key,
                layer=result.layer,
                deterministic=result.deterministic,
                hook=hook,
                passed=result.passed,
                rewrote=result.mutated is not None,
                reason=result.reason,
                latency_ms=result.latency_ms,
                cost_usd=result.cost_usd,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                matched=result.matched,
            )
        )

    # ---- readings the console and the report want -------------------------
    @property
    def blocks(self) -> list[ControlEvent]:
        return [event for event in self.events if not event.passed]

    @property
    def blocked_by(self) -> tuple[str, ...]:
        """Controls that stopped something, in the order they did it."""
        seen: list[str] = []
        for event in self.blocks:
            if event.key not in seen:
                seen.append(event.key)
        return tuple(seen)

    @property
    def rewrites(self) -> list[ControlEvent]:
        """Controls that changed a value without blocking anything."""
        return [event for event in self.events if event.rewrote and event.passed]

    @property
    def rewrote_by(self) -> tuple[str, ...]:
        seen: list[str] = []
        for event in self.rewrites:
            if event.key not in seen:
                seen.append(event.key)
        return tuple(seen)

    @property
    def denied_tools(self) -> list[ToolAttempt]:
        return [attempt for attempt in self.tools if attempt.outcome != "allow"]

    def cost_by_layer(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for event in self.events:
            totals[event.key] = round(totals.get(event.key, 0.0) + event.cost_usd, 8)
        return totals

    def latency_by_layer(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for event in self.events:
            totals[event.key] = round(totals.get(event.key, 0.0) + event.latency_ms, 3)
        return totals

    def to_dict(self) -> dict[str, Any]:
        return {
            "controls": dict(self.controls),
            "controls_attribute": self.controls_attribute,
            "principal": self.principal.to_dict(),
            "route": self.route,
            "payload_id": self.payload_id,
            "events": [event.to_dict() for event in self.events],
            "tools": [attempt.to_dict() for attempt in self.tools],
            "blocked_by": list(self.blocked_by),
            "rewrote_by": list(self.rewrote_by),
            "flagged_ids": sorted(self.flagged_ids),
            # The map itself is never rendered: it is the thing the separation of
            # stores exists to keep away from whoever reads a trace.
            "pii_classes": dict(self.pii_classes),
            "pii_tokens": len(self.pii_map),
            "sent_system": self.sent_system,
            "sent_user": self.sent_user,
            "tools_offered": list(self.tools_offered),
            "cost_by_layer": self.cost_by_layer(),
            "latency_by_layer": self.latency_by_layer(),
            "notes": list(self.notes),
        }


_current: ContextVar[RequestContext | None] = ContextVar("attacklab_request", default=None)

#: The persona a request runs as when nothing says otherwise.  The ordinary
#: employee, not the administrator: a lab whose default identity can do
#: everything demonstrates nothing.
DEFAULT_PRINCIPAL = Principal(
    subject="e.employee@example.gov.uk",
    roles=frozenset({"employee"}),
    tenant="default",
)


@contextmanager
def request_context(
    registry: Registry,
    *,
    principal: Principal | None = None,
    route: str = "ask",
    payload_id: str = "",
    overrides: dict[str, bool] | None = None,
) -> Iterator[RequestContext]:
    """Snapshot the registry and make it the active context for one request."""
    snapshot = registry.snapshot()
    if overrides is not None:
        # ``overrides`` is the **complete** control set for this run: any key it
        # does not mention is off, whatever the registry says.
        #
        # It merged onto the snapshot at first, and ``controls={}`` -- the
        # harness's way of saying "run this with nothing on" -- was falsy, so a
        # bare run silently inherited whatever the console's toggles happened to
        # be. The tally then reported a 0% block rate while the panel showed two
        # of three payloads contained, because every bare probe came back
        # already-defended. A partial override is also the wrong contract for a
        # measurement: "only tool_broker" has to mean only tool_broker.
        snapshot = {key: bool(overrides.get(key, False)) for key in snapshot}
    ctx = RequestContext(
        controls=snapshot,
        principal=principal or DEFAULT_PRINCIPAL,
        route=route,
        payload_id=payload_id,
    )
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)


def current(registry: Registry | None = None) -> RequestContext | None:
    """The active context, or ``None``.

    Returns ``None`` rather than inventing one: a hook firing outside a request
    context means someone called the assistant without going through the lab,
    and silently defending that request would make the lab's own measurements
    wrong.  Callers treat ``None`` as "controls off".
    """
    return _current.get()
