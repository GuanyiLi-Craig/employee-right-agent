"""Layer 4 -- the tool broker.  The most important module in the lab.

**Assume the agent is compromised.**  Not that it might be.  Everything this
broker receives from the agent is attacker-influenced input, because a document
the agent read may have written it.  There is no version of this module that
takes the agent's word for anything.

Slide 10 draws it as five boxes and this is built that way literally::

    User -> Agent -> Policy Gate -> Tool -> Sandbox

The agent has no path to a tool that does not pass through the gate: every
implementation in :mod:`attacklab.tools` is module-private and
:func:`attacklab.tools.dispatch` calls the hook before it calls anything.  If a
reviewer can find a path around it, the boundary does not exist -- there is only
a model deciding its own permissions.

Five behaviours, matching slide 10's five gate rules one to one.

1. **Scope to the caller.**  Grants derive from :class:`Principal`, never from a
   service account.  Two personas ship with the lab, and the same payload is
   contained for one and permitted for the other.
2. **Enumerate per request.**  The tool list handed to the model is
   :meth:`ToolBroker.mint`.  Before: thirteen tools.  After, for an ordinary
   employee: three.
3. **Parameters are authorisation.**  ``send_notification`` is not one
   permission; ``send_notification(to=<inside the tenant>)`` is.
4. **Human approval on the irreversible.**  Tools tagged ``irreversible``
   return ``needs_approval`` even for a principal who holds the grant.
5. **Deny by default, and log the denial.**  Unknown tool, expired grant,
   failed constraint -> deny, with a structured reason, into the audit chain.
   **A denied tool call is your best injection detector**, because it is a
   request the system made that no user asked for.

Rule 5a, which comes before all of them: **audit the tool surface itself.**  The
set of functions the model can see must be the set you intended.
:mod:`attacklab.supplychain.scan_tool_surface` is that check, and
CVE-2026-25592 is the whole argument -- least privilege over a tool the model
should never have been offered is a policy applied one layer too late.

**Say the cost honestly.**  Per-request minting means an identity-aware broker,
short-lived credentials, and a way to enumerate capability per caller.  It is
the most engineering work on the list and the only guarantee on it.  It is not
free, and the console prices it like everything else.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from attacklab.context import RequestContext
from attacklab.controls.base import Control, ControlResult
from attacklab.tools import RUNTIME_SURFACE, ToolSpec, spec_for
from rights_agent.hooks import Decision, Principal

#: How long a minted grant is good for.  Short, and checked: an unexpired grant
#: is the only thing that authorises a call, so "short-lived" has to be a
#: comparison somewhere rather than an adjective in a design document.
GRANT_TTL_S = 120

#: Mailbox domains inside the tenant.  The recipient allowlist, and the only
#: thing standing between ``send_notification`` and an exfiltration channel.
TENANT_DOMAINS: frozenset[str] = frozenset({"example.gov.uk"})

#: Ceiling on ``approve_expense``.  Amounts are authorisation too.
EXPENSE_CEILING_GBP = 250.0


# --------------------------------------------------------------------------- #
# Constraints
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Constraint:
    """A predicate over one parameter, with a reason it exists.

    ``kind`` is what the console renders next to the tool, because "this tool is
    granted" and "this tool is granted for these values" are different
    statements and only the second one is true.
    """

    kind: str
    detail: str

    def check(self, value: Any, principal: Principal) -> str:
        """Empty string when satisfied; the denial reason otherwise."""
        text = "" if value is None else str(value)
        if self.kind == "recipient_domain":
            domain = text.rpartition("@")[2].lower()
            if not domain:
                return f"{text!r} is not a mailbox"
            if domain not in TENANT_DOMAINS:
                return (
                    f"recipient domain {domain!r} is outside the tenant "
                    f"({', '.join(sorted(TENANT_DOMAINS))})"
                )
            return ""
        if self.kind == "amount_ceiling":
            try:
                amount = float(text.replace("£", "").replace(",", "") or 0)
            except ValueError:
                return f"{text!r} is not an amount"
            if amount > EXPENSE_CEILING_GBP:
                return f"£{amount:,.2f} exceeds the £{EXPENSE_CEILING_GBP:,.2f} ceiling"
            return ""
        if self.kind == "owns_object":
            own = payroll_id_for(principal)
            if text and text != own:
                return (
                    f"payroll id {text!r} does not belong to {principal.subject} "
                    f"(who is {own})"
                )
            return ""
        # An unrecognised constraint denies. A gate that shrugs at a rule it does
        # not understand is a gate with a hole the shape of a typo.
        return f"unrecognised constraint {self.kind!r}"


#: Payroll ids, so the ownership constraint has something to compare against.
PAYROLL_IDS: dict[str, str] = {
    "e.employee@example.gov.uk": "PR-0000042",
    "hr.admin@example.gov.uk": "PR-0000007",
}


def payroll_id_for(principal: Principal) -> str:
    return PAYROLL_IDS.get(principal.subject, "PR-UNKNOWN")


#: Which parameters of which tools are constrained.  Data, so it can be read in
#: a review the way a permission grant should be.
CONSTRAINTS: dict[str, dict[str, Constraint]] = {
    "send_notification": {
        "to": Constraint(
            "recipient_domain",
            "recipients must be inside the tenant; this is the exfiltration channel",
        )
    },
    "lookup_holiday_balance": {
        "payroll_id": Constraint("owns_object", "the caller's own record only")
    },
    "lookup_payroll_record": {
        "payroll_id": Constraint("owns_object", "administrators still read one record at a time")
    },
    "approve_expense": {
        "amount_gbp": Constraint("amount_ceiling", f"at most £{EXPENSE_CEILING_GBP:,.0f}")
    },
}


# --------------------------------------------------------------------------- #
# Personas
# --------------------------------------------------------------------------- #
#: An ordinary employee.  The default, deliberately: a lab whose default
#: identity can do everything demonstrates nothing.
EMPLOYEE = Principal(
    subject="e.employee@example.gov.uk",
    roles=frozenset({"employee"}),
    tenant="default",
)

#: An HR administrator.  The same payload, the same text, a different outcome --
#: which is what makes "the agent acts with the caller's rights" concrete.
HR_ADMIN = Principal(
    subject="hr.admin@example.gov.uk",
    roles=frozenset({"employee", "hr_admin"}),
    tenant="default",
)

PERSONAS: dict[str, Principal] = {"employee": EMPLOYEE, "hr_admin": HR_ADMIN}
PERSONA_LABELS: dict[str, str] = {
    "employee": "Ordinary employee",
    "hr_admin": "HR administrator",
}


def persona(name: str) -> Principal:
    try:
        return PERSONAS[name]
    except KeyError:
        raise KeyError(f"unknown persona {name!r}; known: {sorted(PERSONAS)}") from None


# --------------------------------------------------------------------------- #
# Grants
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Grant:
    """Permission to call one tool, for a bounded time, for bounded values."""

    tool: str
    param_constraints: Mapping[str, Constraint]
    ttl_s: int
    request_id: str
    issued_at: float = field(default_factory=time.monotonic)
    irreversible: bool = False

    def expired(self, now: float | None = None) -> bool:
        return (now or time.monotonic()) - self.issued_at > self.ttl_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "ttl_s": self.ttl_s,
            "irreversible": self.irreversible,
            "constraints": {
                name: {"kind": c.kind, "detail": c.detail}
                for name, c in self.param_constraints.items()
            },
        }


class ToolBroker:
    """Mint grants for a caller; check every call against them."""

    def __init__(self, *, ttl_s: int = GRANT_TTL_S) -> None:
        self.ttl_s = ttl_s

    # ---- rule 2: enumerate per request ------------------------------------
    def mint(self, principal: Principal, route: str = "ask") -> list[Grant]:
        """The grants this caller holds for this route.

        Derived from the principal, never from a service account.  A broker that
        minted for the service would have granted the union of everything anyone
        may do, which is the shape of every credential in the July 2026
        intrusion's "what failed" column.
        """
        request_id = f"{principal.subject}:{route}:{int(time.monotonic() * 1000)}"
        grants: list[Grant] = []
        for spec in RUNTIME_SURFACE:
            if not spec.roles or not (spec.roles & principal.roles):
                continue
            grants.append(
                Grant(
                    tool=spec.name,
                    param_constraints=CONSTRAINTS.get(spec.name, {}),
                    ttl_s=self.ttl_s,
                    request_id=request_id,
                    irreversible=spec.irreversible,
                )
            )
        return grants

    def offered_tools(self, principal: Principal | None, route: str = "ask") -> list[str]:
        """The tool list to hand the model.

        ``None`` is the *before* state: no broker, so the model sees the whole
        runtime surface -- including the one tool nobody meant to expose.  Show
        both lists on the console. It is the single most persuasive screen in the
        session.
        """
        if principal is None:
            return [spec.name for spec in RUNTIME_SURFACE]
        return [grant.tool for grant in self.mint(principal, route)]

    # ---- rules 1, 3, 4, 5: check ------------------------------------------
    def check(
        self, name: str, params: Mapping[str, Any], grants: list[Grant]
    ) -> Decision:
        held = {grant.tool: grant for grant in grants}
        spec: ToolSpec | None = spec_for(name)

        if spec is None:
            # Rule 5, first form: an unknown name is denied before anything else
            # happens. Not "logged and allowed" -- there is no such outcome here.
            return Decision.deny(
                f"unknown tool {name!r}: no grant exists and none can", control="tool_broker"
            )
        grant = held.get(name)
        if grant is None:
            return Decision.deny(
                f"no grant for {name!r}: this caller holds "
                f"{', '.join(sorted(held)) or 'no tool grants'}",
                control="tool_broker",
            )
        if grant.expired():
            return Decision.deny(
                f"the grant for {name!r} expired after {grant.ttl_s}s", control="tool_broker"
            )

        # Rule 3: parameters are authorisation.
        for param, constraint in grant.param_constraints.items():
            reason = constraint.check(params.get(param), _principal_of(grants))
            if reason:
                return Decision.deny(
                    f"{name}: {reason}", control="tool_broker", parameter=param
                )

        # Rule 4: human approval on the irreversible.
        if spec.irreversible:
            return Decision.needs_approval_for(
                f"{name} is irreversible; a person has to say yes", control="tool_broker"
            )
        return Decision.allow(grant_id=grant.request_id, reason=f"granted to the caller for {name}")


def _principal_of(grants: list[Grant]) -> Principal:
    """Recover the caller from a grant set.

    The grant's ``request_id`` carries the subject, which is what makes a grant
    self-describing: a constraint that had to be told who the caller is could be
    told the wrong thing.
    """
    if not grants:
        return EMPLOYEE
    subject = grants[0].request_id.partition(":")[0]
    for candidate in PERSONAS.values():
        if candidate.subject == subject:
            return candidate
    return Principal(subject=subject, roles=frozenset(), tenant="default")


class ToolBrokerControl(Control):
    """Layer 4, as a toggle.  **Deterministic**: the only guarantee in the lab.

    Deterministic in the strict sense: the decision is a function of the
    principal, the tool name and the parameters.  No text is read, no model is
    consulted, and the same inputs produce the same decision every time.  That
    is why the eval suite asserts *zero* escapes for every ``ACTION``-goal
    payload when this is on, rather than a threshold.
    """

    key = "tool_broker"
    layer = 4
    deterministic = True

    def __init__(self, broker: ToolBroker | None = None) -> None:
        self.broker = broker or ToolBroker()

    def decide(
        self, name: str, params: Mapping[str, Any], principal: Principal, ctx: RequestContext
    ) -> Decision:
        """The gate.  Called from the stack's ``on_tool_call``."""
        grants = self.broker.mint(principal, ctx.route)
        return self.broker.check(name, params, grants)

    def _run(self, value: Any, ctx: RequestContext) -> ControlResult:
        """``value`` is ``(name, params, principal)``.

        The :class:`~rights_agent.hooks.Decision` is returned in ``mutated``.
        That is not a rewrite in disguise: for this hook the decision **is** the
        value the hook returns, and carrying it here is what lets the base class
        do the timing, the span and the recording exactly once.  Calling
        :meth:`decide` a second time from the stack -- which is what the first
        version of this did -- minted a second grant set and lost the
        ``needs_approval`` outcome, so an irreversible tool read as allowed.
        """
        name, params, principal = value
        decision = self.decide(name, params, principal, ctx)
        return ControlResult(
            passed=decision.allowed,
            reason=decision.reason,
            mutated=decision,
            matched="" if decision.allowed else name,
        )
