"""What every control has in common: a result type, a price, and a span.

Two disciplines are enforced here rather than left to review.

**A control that rewrites returns** ``mutated``; **a control that judges returns**
``passed`` **and** ``reason``.  Never both.  Mixing them produces controls that
silently change behaviour when they were supposed to be observing, and you find
out on stage.

**A control never raises to signal a block.**  Blocking is a return value.  An
exception is a bug, and what you want in front of a room is a clean "blocked by
layer 3" row rather than a traceback.  :meth:`Control.__call__` therefore
catches everything a ``_run`` can throw and converts it into a *pass* with the
error in the reason — a broken control must not silently become a blocking one.

Every control is priced.  Slide 8 depends on it: **if you cannot say what a
control costs, you cannot defend it in a design review.**  A model-backed input
scanner is another inference call, and the panel says so in milliseconds and in
dollars.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from attacklab.context import RequestContext
from attacklab.telemetry import emit_control_span
from rights_agent.log import get_logger

log = get_logger("attacklab.control")


@dataclass(slots=True)
class ControlResult:
    """What one control invocation produced."""

    passed: bool = True
    reason: str = ""
    #: Replacement value, if this control rewrites.  ``None`` means "unchanged".
    mutated: Any | None = None
    #: The offending text, for in-place highlighting on the console.
    matched: str = ""
    latency_ms: float = 0.0
    #: Non-zero only for model-backed controls.
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    #: Filled in by :meth:`Control.__call__`; a ``_run`` never sets these.
    key: str = ""
    layer: int = 0
    deterministic: bool = False

    @property
    def blocked(self) -> bool:
        return not self.passed


#: A control that is switched off returns this, and it must be indistinguishable
#: from "ran and found nothing" for every caller except the report.
DISABLED = "disabled"


class Control(ABC):
    """One toggleable control.

    Subclasses set :attr:`key`, :attr:`layer` and :attr:`deterministic`, and
    implement :meth:`_run`.  Everything else -- the enabled check, the timing,
    the span, the error containment -- happens here, once.
    """

    key: str = ""
    layer: int = 0
    deterministic: bool = False

    def __call__(self, value: Any, ctx: RequestContext, *, hook: str = "") -> ControlResult:
        if not ctx.get(self.key):
            return self._stamp(ControlResult(passed=True, reason=DISABLED))
        started = time.perf_counter()
        try:
            result = self._run(value, ctx)
        except Exception as exc:
            # Fail *open*, loudly. A control whose bug turns into a block is a
            # control that takes the system down on a bad deploy, and the panel
            # would read "contained" when nothing contained anything.
            log.exception("control %s raised; treating as pass", self.key)
            result = ControlResult(passed=True, reason=f"control error: {type(exc).__name__}: {exc}")
        result.latency_ms = round((time.perf_counter() - started) * 1_000, 3)
        result = self._stamp(result)
        emit_control_span(
            self.key, self.layer, self.deterministic, result, ctx.controls_attribute
        )
        ctx.record(result, hook=hook or type(self).__name__)
        return result

    def _stamp(self, result: ControlResult) -> ControlResult:
        result.key = self.key
        result.layer = self.layer
        result.deterministic = self.deterministic
        return result

    @abstractmethod
    def _run(self, value: Any, ctx: RequestContext) -> ControlResult:
        """Do the work.  Return a result; never raise to signal a block."""
