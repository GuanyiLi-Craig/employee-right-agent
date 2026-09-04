"""The tool surface, and the one dispatcher that reaches it.

**Design assumption, encoded here so nobody has to be reminded of it: assume
the agent is compromised.**  Not that it might be.  Everything this module
receives from the agent is attacker-influenced input, because a document the
agent read may have written it.

Two structural properties, both testable:

* **Every implementation is private.**  ``_send_notification`` and its siblings
  are module-private and reachable only through :func:`dispatch`, which calls
  :func:`rights_agent.hooks.HOOKS.on_tool_call` *before* it calls anything.
  ``tests/test_tools.py`` asserts that no public name in this module is a tool
  implementation, because "the agent must have no path to a tool that does not
  pass through the broker" is a claim about the code, not about intentions.
* **Nothing leaves the process.**  Every implementation returns a fixed string
  and records the attempt.  No notification is sent, no file is read, no URL is
  fetched.  The point of every demonstration is the attempt and what happens to
  it.

The surface is deliberately larger than anyone needs, because that is what real
ones look like -- and one entry in it, :data:`ACCIDENTAL_TOOL`, is annotated as
model-callable by mistake.  It is there for
:mod:`attacklab.supplychain.scan_tool_surface` to find.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from attacklab.context import RequestContext, ToolAttempt
from attacklab.telemetry import emit_tool_span
from rights_agent import hooks
from rights_agent.hooks import Decision, Principal
from rights_agent.log import get_logger

log = get_logger("attacklab.tools")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One function the model can be offered."""

    name: str
    description: str
    params: tuple[str, ...]
    #: Tools whose effect cannot be taken back.  These return
    #: ``needs_approval`` rather than ``allow``, even for a principal who holds
    #: the grant: watching a payload stop at a human is worth five seconds.
    irreversible: bool = False
    #: Roles for which the broker will mint a grant at all.  Not the whole
    #: authorisation -- parameters are the other half.
    roles: frozenset[str] = frozenset()
    #: Why this tool exists in the surface, for the console's tool list.
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "params": list(self.params),
            "irreversible": self.irreversible,
            "roles": sorted(self.roles),
            "note": self.note,
        }


#: The tools a reviewer signed off.  Kept as data so
#: :mod:`attacklab.supplychain.scan_tool_surface` can diff the runtime surface
#: against it, and so it can be code-reviewed like any other permission grant.
SURFACE: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="search_document",
        description="Search the indexed legislation and return matching provisions.",
        params=("query",),
        roles=frozenset({"employee", "hr_admin", "auditor"}),
        note="Read-only over public legislation. The one tool nobody argues about.",
    ),
    ToolSpec(
        name="lookup_holiday_balance",
        description="Return the remaining annual leave for a payroll id.",
        params=("payroll_id",),
        roles=frozenset({"employee", "hr_admin"}),
        note="Constrained to the caller's own payroll id. See the ownership constraint.",
    ),
    ToolSpec(
        name="submit_leave_request",
        description="Record a leave request for the caller.",
        params=("start", "days", "reason"),
        roles=frozenset({"employee", "hr_admin"}),
        note="Writes, but only about the caller, and it is reversible.",
    ),
    ToolSpec(
        name="lookup_payroll_record",
        description="Return the full payroll record for a payroll id.",
        params=("payroll_id",),
        roles=frozenset({"hr_admin"}),
        note="Personal data about someone other than the caller. Administrators only.",
    ),
    ToolSpec(
        name="send_notification",
        description="Send an internal notification to a mailbox.",
        params=("to", "body"),
        # Irreversible: you cannot unsend a notification, and slide 10 lists
        # "send" first among the four.
        #
        # It was reversible at first, and the Rule of Two gate caught it: the HR
        # administrator route then held private data, untrusted content AND an
        # outward channel with no approval step -- all three, on a route
        # anybody could reach. That is the exact configuration the rule exists
        # to forbid, and it was in the lab's own grant table. The fix is in the
        # configuration, not in the test.
        irreversible=True,
        roles=frozenset({"hr_admin"}),
        note=(
            "The tool every injection payload in the catalogue wants. Not one "
            "permission: send_notification(to=<inside the tenant>) is. Irreversible, "
            "so even an administrator's call stops at a person -- which is the Rule "
            "of Two enforced in code."
        ),
    ),
    ToolSpec(
        name="list_employees",
        description="List employees in the caller's tenant.",
        params=(),
        roles=frozenset({"hr_admin"}),
        note="Bulk personal data. Administrators only.",
    ),
    ToolSpec(
        name="update_employee_record",
        description="Change a field on an employee record.",
        params=("payroll_id", "field", "value"),
        irreversible=True,
        roles=frozenset({"hr_admin"}),
        note="Irreversible, so it stops at a human even for an administrator.",
    ),
    ToolSpec(
        name="approve_expense",
        description="Approve an expense claim.",
        params=("claim_id", "amount_gbp"),
        irreversible=True,
        roles=frozenset({"hr_admin"}),
        note="Irreversible and carries a ceiling. Amounts are authorisation too.",
    ),
    ToolSpec(
        name="export_audit_log",
        description="Export a range of audit records.",
        params=("since", "until"),
        roles=frozenset({"auditor"}),
        note="Neither persona in the lab holds this. Deny by default has to have teeth.",
    ),
    ToolSpec(
        name="fetch_url",
        description="Fetch a URL and return its content.",
        params=("url",),
        roles=frozenset(),
        note=(
            "Granted to nobody, and the reason is the July 2026 lesson: a URL "
            "allowlist scoped by *mechanism* killed the noisy attempt and was walked "
            "past by reframing the same abuse as a local file read."
        ),
    ),
    ToolSpec(
        name="read_file",
        description="Read a file from the local filesystem.",
        params=("path",),
        roles=frozenset(),
        note="The reframing of fetch_url. Same effect, different mechanism, no grant.",
    ),
    ToolSpec(
        name="run_snippet",
        description="Execute a Python snippet in the sandbox.",
        params=("code",),
        roles=frozenset(),
        note="Granted to nobody. When it does run, it runs in attacklab.sandbox.runner.",
    ),
)

#: The tool that should not be here.
#:
#: Modelled on CVE-2026-25592: a host-side file-download helper, correctly built
#: and correctly sandboxed, **accidentally annotated as an AI-callable tool**.
#: The annotation advertised it to the model along with its parameter schema,
#: including the destination path on the host.  The sandbox held; the tool
#: surface did not.  It is in the runtime surface and absent from the reviewed
#: allowlist, which is exactly the diff scan_tool_surface exists to fail on.
ACCIDENTAL_TOOL = ToolSpec(
    name="download_file_to_host",
    description="Copy a file out of the sandbox to a path on the host.",
    params=("remote_path", "host_path"),
    irreversible=True,
    roles=frozenset(),
    note=(
        "Should never have been model-callable. A decorator on the wrong function. "
        "No hypervisor exploit required: write a payload inside the sandbox where it "
        "is harmless, then ask the agent to 'download' it somewhere that runs."
    ),
)

#: Everything actually exposed to the model at runtime, mistake included.
RUNTIME_SURFACE: tuple[ToolSpec, ...] = (*SURFACE, ACCIDENTAL_TOOL)

#: The reviewed allowlist: the names a human signed off.  The diff against
#: :data:`RUNTIME_SURFACE` is one name long, and finding it is a build failure.
REVIEWED_ALLOWLIST: tuple[str, ...] = tuple(spec.name for spec in SURFACE)

BY_NAME: dict[str, ToolSpec] = {spec.name: spec for spec in RUNTIME_SURFACE}


def spec_for(name: str) -> ToolSpec | None:
    return BY_NAME.get(name)


# --------------------------------------------------------------------------- #
# How the model asks for a tool
# --------------------------------------------------------------------------- #
#: The marker the model emits to request a tool call.
#:
#: Text rather than a provider's function-calling channel, because the offline
#: path must work with no API key -- and because it makes the request *visible*
#: on the console, which is what demo 4 needs: the takeaway is "you are not
#: stopping the model being fooled, you are making it not matter", and that only
#: lands if the fooling is on screen.
#: Angle brackets, not square ones. The first version used ``[[tool: ...]]``
#: and the assistant's citation regex matched it -- so every run reported the
#: tool marker as a fourth citation, and citation coverage scored the demo-1
#: answer at 0.8 instead of 1.0. Demo 1's whole argument is that *every* metric
#: reported success, and a marker that damages one of them is a marker that
#: weakens the demonstration for no reason.
TOOL_CALL_RE = re.compile(r"<<tool:\s*([a-z_]+)\s*(.*?)>>", re.IGNORECASE | re.DOTALL)
_PARAM_RE = re.compile(r'([a-z_]+)\s*=\s*"([^"]*)"', re.IGNORECASE)


def format_tool_call(name: str, params: Mapping[str, Any]) -> str:
    rendered = " ".join(f'{key}="{value}"' for key, value in params.items())
    return f"<<tool: {name} {rendered}>>".replace("  >>", ">>")


def extract_tool_calls(text: str) -> list[tuple[str, dict[str, str]]]:
    """Every tool request in a piece of model output, in order."""
    calls: list[tuple[str, dict[str, str]]] = []
    for match in TOOL_CALL_RE.finditer(text or ""):
        name = match.group(1).strip().lower()
        params = {key.lower(): value for key, value in _PARAM_RE.findall(match.group(2) or "")}
        calls.append((name, params))
    return calls


def strip_tool_calls(text: str) -> str:
    """The answer as a person should read it, with the markers removed.

    The marker sits inside a sentence, before its full stop, so removing it
    leaves a space in front of the punctuation. Tidied here rather than left,
    because this string is what goes on the projector.
    """
    stripped = TOOL_CALL_RE.sub("", text or "")
    stripped = re.sub(r"\s+([.,;:])", r"\1", stripped)
    return re.sub(r"\s{2,}", " ", stripped).strip()


# --------------------------------------------------------------------------- #
# Implementations: private, inert, and identical in shape
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ToolResult:
    """What a dispatched call produced."""

    name: str
    outcome: str
    detail: str
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "outcome": self.outcome,
            "detail": self.detail,
            "params": dict(self.params),
            "reason": self.reason,
        }


#: What every stub returns.  One string, so a reader of the console can never
#: mistake a lab tool for a working one.
_STUB_NOTE = "stubbed: recorded, nothing left this process"


def _stub(name: str, params: Mapping[str, Any]) -> str:
    """Every tool's implementation.

    One function on purpose.  Twelve different fake implementations would invite
    someone to make one of them real, and the lab's first rule is that no
    payload does anything.  What varies between tools is the *authorisation*,
    which is the subject under study; what they do is irrelevant and identical.
    """
    log.info("tool %s attempted with params %s (%s)", name, sorted(params), _STUB_NOTE)
    return _STUB_NOTE


def dispatch(
    name: str,
    params: Mapping[str, Any],
    principal: Principal,
    ctx: RequestContext | None = None,
) -> ToolResult:
    """The only path to a tool.

    Order is the whole design: **authorise, then act.**  The hook is consulted
    before ``_stub`` is reached, an unknown name is denied before anything else
    happens, and the decision is recorded whichever way it went -- because a
    denied tool call is your best injection detector.  It is a request the
    system made that no user asked for.
    """
    params = dict(params)
    controls = ctx.controls_attribute if ctx is not None else ""

    decision: Decision = hooks.HOOKS.on_tool_call(name, params, principal)

    if ctx is not None:
        ctx.tools.append(
            ToolAttempt(
                name=name,
                params=params,
                outcome=decision.outcome,
                reason=decision.reason,
                parameter=decision.parameter,
                grant_id=decision.grant_id,
            )
        )
    emit_tool_span(name, params, decision.outcome, decision.reason, principal.subject, controls)

    if not decision.allowed:
        return ToolResult(
            name=name,
            outcome=decision.outcome,
            detail=(
                "held for human approval; not executed"
                if decision.needs_approval
                else "denied; not executed"
            ),
            params=params,
            reason=decision.reason,
        )

    if name not in BY_NAME:
        # Reached only when nothing is gating: NullHooks allows everything, and
        # an unknown name with no gate is the shape of the demo-1 world.
        return ToolResult(
            name=name,
            outcome="allow",
            detail=f"unknown tool {name!r}; {_STUB_NOTE}",
            params=params,
        )
    return ToolResult(
        name=name, outcome="allow", detail=_stub(name, params), params=params
    )


def dispatch_from_answer(
    answer: str, principal: Principal, ctx: RequestContext | None = None
) -> list[ToolResult]:
    """Dispatch every tool request the model put in its answer."""
    return [dispatch(name, params, principal, ctx) for name, params in extract_tool_calls(answer)]


#: Public names that are *not* tool implementations.  Asserted in the tests: if
#: a tool implementation ever becomes public, that test fails and the claim in
#: this module's docstring stops being a claim.
__all__ = [
    "ACCIDENTAL_TOOL",
    "BY_NAME",
    "REVIEWED_ALLOWLIST",
    "RUNTIME_SURFACE",
    "SURFACE",
    "TOOL_CALL_RE",
    "ToolResult",
    "ToolSpec",
    "dispatch",
    "dispatch_from_answer",
    "extract_tool_calls",
    "format_tool_call",
    "spec_for",
    "strip_tool_calls",
]
