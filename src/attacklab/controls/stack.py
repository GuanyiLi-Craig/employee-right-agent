"""``ControlStack``: the seven hooks, implemented over the registry.

This is the indirection that makes the toggles live.  Every hook consults the
active :class:`~attacklab.context.RequestContext`, which was snapshotted from
the registry at request start, so flipping a switch in the console changes
behaviour on the **next** request with no restart.  The presenter never restarts
anything on stage.

Two rules the stack itself has to obey, because the controls cannot enforce them
from inside:

**No request context, no controls.**  A hook firing outside a context means
somebody called the assistant without going through the lab.  Defending that
request silently would put an unrecorded control invocation into the lab's own
measurements, so the stack passes it straight through.  The console and the
harness always open a context; nothing else does.

**One hook, one direction.**  Inbound hooks return the rewritten value; outbound
hooks return a verdict.  The stack is where that shows: ``on_question`` returns a
string even when a control blocked, and ``on_answer`` returns a
:class:`~rights_agent.hooks.Verdict` and changes nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from attacklab import context as request_ctx
from attacklab.controls.input_scan import HEURISTIC, InputScan
from attacklab.controls.output_verify import OutputVerify
from attacklab.controls.pii import PiiMask
from attacklab.controls.provenance import Provenance
from attacklab.controls.residency import Residency
from attacklab.controls.tool_broker import ToolBroker, ToolBrokerControl
from attacklab.registry import Registry, enabled_keys
from rights_agent.hooks import (
    AnswerContext,
    Decision,
    ModelRef,
    Principal,
    ResidencyError,
    Verdict,
    default_model_ref,
)
from rights_agent.log import get_logger

if TYPE_CHECKING:  # pragma: no cover - types only
    from rights_agent.retrieval import Doc

log = get_logger("attacklab.stack")


class ControlStack:
    """Implements :class:`rights_agent.hooks.Hooks` by consulting the registry."""

    def __init__(
        self,
        registry: Registry,
        *,
        scan_mode: str = HEURISTIC,
        judge: bool = True,
        broker: ToolBroker | None = None,
    ) -> None:
        self.registry = registry
        self.broker = broker or ToolBroker()
        self.input_scan = InputScan(scan_mode)
        self.provenance = Provenance()
        self.output_verify = OutputVerify(judge=judge)
        self.tool_broker = ToolBrokerControl(self.broker)
        self.pii = PiiMask()
        self.residency = Residency()

    # ---- 1. on_question ---------------------------------------------------
    def on_question(self, q: str) -> str:
        ctx = request_ctx.current()
        if ctx is None:
            return q
        # PII first, and the order is the control: masking has to happen before
        # anything reads the string, and the scanner reading a masked question
        # is fine -- an instruction is instruction-shaped with or without a
        # National Insurance number in it.
        masked = self.pii(q, ctx, hook="on_question")
        if masked.mutated is not None:
            q = str(masked.mutated)
        scanned = self.input_scan(q, ctx, hook="on_question")
        if scanned.blocked and scanned.mutated is not None:
            return str(scanned.mutated)
        return q

    # ---- 2. on_context ----------------------------------------------------
    def on_context(self, blocks: Sequence[Doc]) -> Sequence[Doc]:
        ctx = request_ctx.current()
        if ctx is None:
            return blocks
        self.provenance(list(blocks), ctx, hook="on_context")
        scanned = self.input_scan(list(blocks), ctx, hook="on_context")
        if scanned.blocked and scanned.mutated is not None:
            return list(scanned.mutated)
        return blocks

    # ---- 3. on_prompt -----------------------------------------------------
    def on_prompt(self, system: str, user: str) -> tuple[str, str]:
        ctx = request_ctx.current()
        if ctx is None:
            return system, user
        fenced = self.provenance((system, user), ctx, hook="on_prompt")
        if fenced.mutated is not None:
            system, user = fenced.mutated
        # Masked here as well as at on_question, because the *retrieved* text
        # can carry personal data the question never did. This is the assembled
        # prompt: what the console shows, and the last thing before the wire.
        masked = self.pii(user, ctx, hook="on_prompt")
        if masked.mutated is not None:
            user = str(masked.mutated)
        ctx.sent_system, ctx.sent_user = system, user
        return system, user

    # ---- 4. on_answer -----------------------------------------------------
    def on_answer(self, answer: str, answer_ctx: AnswerContext) -> Verdict:
        ctx = request_ctx.current()
        if ctx is None:
            return Verdict.allow()

        # Outbound PII scan before output verification and, critically, before
        # anything is written: `on_log` runs after this, and a log written first
        # is the leak. Order is the control.
        leak = self.pii(("outbound", answer), ctx, hook="on_answer")
        if leak.blocked:
            return Verdict.block(
                leak.reason,
                control=self.pii.key,
                layer=self.pii.layer,
                replacement=(
                    "I cannot return this answer: an outbound scan found personal data "
                    "in it that was not in your request."
                ),
            )

        verified = self.output_verify((answer, answer_ctx), ctx, hook="on_answer")
        if verified.blocked:
            return Verdict.block(
                verified.reason,
                control=self.output_verify.key,
                layer=self.output_verify.layer,
                replacement=(
                    "I withheld this answer. It proposed an action nobody asked for: "
                    f"{verified.reason}"
                ),
            )
        return Verdict.allow(verified.reason)

    # ---- 5. on_tool_call --------------------------------------------------
    def on_tool_call(self, name: str, params: dict[str, Any], principal: Principal) -> Decision:
        ctx = request_ctx.current()
        if ctx is None:
            return Decision.allow(reason="no request context: the lab is not watching")
        if not ctx.get(self.tool_broker.key):
            # The demo-1 world. Recorded as an attempt, allowed, and stubbed --
            # which is the point: nothing gated it, and nothing had to.
            return Decision.allow(reason="tool_broker off: nothing gated this call")
        result = self.tool_broker((name, params, principal), ctx, hook="on_tool_call")
        if isinstance(result.mutated, Decision):
            return result.mutated
        # Only reachable if the control errored: the base class fails controls
        # open, and an open failure here means the gate did not run. Denied
        # anyway, because "the gate is broken" is not a reason to allow a call.
        return Decision.deny(
            f"the gate did not produce a decision ({result.reason})",
            control=self.tool_broker.key,
        )

    # ---- 6. on_log --------------------------------------------------------
    def on_log(self, record: dict[str, Any]) -> dict[str, Any]:
        """Annotate the audit record with what the controls did.

        **Unreachable by the model.**  This is called by the agent, with fields
        the agent assembled, from inside
        :meth:`rights_agent.audit.AuditLog.append`.  Nothing the model emits
        arrives here.  That is the whole answer to the audit-suppression payload
        class: not a filter, the absence of a path.
        """
        ctx = request_ctx.current()
        if ctx is None:
            return record
        controls: dict[str, Any] = {
            "enabled": list(enabled_keys(ctx.controls)),
            "blocked_by": list(ctx.blocked_by),
            "principal": ctx.principal.to_dict(),
        }
        if ctx.payload_id:
            controls["payload_id"] = ctx.payload_id
        if ctx.tools:
            # A denied tool call is the most interesting line in the log: it is a
            # request the system made that no user asked for.
            controls["tool_calls"] = [attempt.to_dict() for attempt in ctx.tools]
        if ctx.get(self.pii.key):
            fields = self.pii.log_fields(ctx)
            if fields:
                # Classes and counts. Never values -- "we redacted this" is the
                # auditable event, and the values are the thing being protected.
                controls["pii"] = fields
        record["controls"] = controls
        return record

    # ---- 7. resolve_model -------------------------------------------------
    def resolve_model(
        self, role: str, region: str | None, configured: str = ""
    ) -> ModelRef:
        ctx = request_ctx.current()
        effective_region = region or ""
        if ctx is None or not ctx.get(self.residency.key):
            return default_model_ref(role, effective_region, configured)
        result = self.residency(
            (role, effective_region, configured), ctx, hook="resolve_model"
        )
        if result.blocked:
            # The one place the lab raises. See the module docstrings in
            # rights_agent.hooks and attacklab.controls.residency: a residency
            # refusal that a caller could treat as advisory is not a guarantee.
            raise ResidencyError(result.reason)
        if isinstance(result.mutated, ModelRef):
            return result.mutated
        return default_model_ref(role, effective_region, configured)
