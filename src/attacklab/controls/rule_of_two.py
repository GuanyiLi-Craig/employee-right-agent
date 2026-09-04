"""The Rule of Two: a reading of the configuration, not a runtime filter.

Two attributions, and they matter because slide 6 gives them: the **lethal
trifecta** is Simon Willison's framing, and the **Rule of Two** -- without a
human approval step, give an agent at most two of the three -- is Meta's.

Three questions about a configuration.  Does it give the agent

1. **private data** -- it can read something worth taking,
2. **untrusted content** -- it reads text an attacker can write, and
3. **the ability to act outward** -- it can send, call, post or write?

If the same agent has all three, one prompt injection can turn it into a data
exfiltration tool.  Hold at most two and the worst case is bounded.

This is not a control that runs.  It is computed from the registry state and the
current grant set and lit on the console as a permanent three-lamp indicator,
which is why it belongs here rather than beside the toggles.  Under the demo-1
preset all three are on and the lamp is red -- **which is the honest reading**,
because that is exactly the configuration that made demo 1 possible.  Enabling
the tool broker removes the third property for the ordinary-employee persona,
and the lamp goes amber.

Two reasons it earns its place on the console.  It is the cheapest thing in the
session for an audience to take back to work -- five minutes with a whiteboard,
no code, no budget, and it converts an unbounded argument about model behaviour
into a design constraint you can check in a code review.  And it makes visible
that the lab's controls are *not* independent knobs: ``tool_broker`` changes the
trifecta state and ``input_scan`` does not, and seeing that on screen makes the
layer 1-3 versus layer 4 distinction concrete rather than rhetorical.

The human-approval clause is the link to slide 10's fourth gate rule: approval
on the irreversible **is** the Rule of Two enforced in code, which is why
:func:`evaluate` takes ``require_approval`` and why a tool held for a human does
not light the third lamp.
"""

from __future__ import annotations

from dataclasses import dataclass

from attacklab.controls.tool_broker import ToolBroker
from attacklab.tools import spec_for
from rights_agent.hooks import Principal

#: Tools that can move data out of the system.  Outward communication is a
#: property of *effect*, not of a name: a tool that writes to a shared record
#: someone else reads is an outward channel too, which is why this list is
#: derived from what each tool does rather than from whether it says "send".
OUTWARD_TOOLS: frozenset[str] = frozenset(
    {
        "send_notification",
        "send_email",
        "fetch_url",
        "download_file_to_host",
        "export_audit_log",
        "update_employee_record",
        "run_snippet",
    }
)

#: Tools that read data belonging to a person.
PRIVATE_DATA_TOOLS: frozenset[str] = frozenset(
    {"lookup_holiday_balance", "lookup_payroll_record", "list_employees", "read_file"}
)

RED = "red"
AMBER = "amber"
GREEN = "green"


@dataclass(frozen=True, slots=True)
class Trifecta:
    """The three properties, and the lamp that follows from them."""

    private_data: bool
    untrusted_content: bool
    outward_communication: bool
    #: Tools that supply the outward property, so the lamp can be explained
    #: rather than merely displayed.
    outward_via: tuple[str, ...] = ()
    private_via: tuple[str, ...] = ()
    untrusted_via: str = ""

    @property
    def count(self) -> int:
        return sum((self.private_data, self.untrusted_content, self.outward_communication))

    @property
    def lamp(self) -> str:
        if self.count >= 3:
            return RED
        if self.count == 2:
            return AMBER
        return GREEN

    @property
    def verdict(self) -> str:
        if self.lamp == RED:
            return "all three: an injection here is a data breach"
        if self.lamp == AMBER:
            return "two of three: the worst case is bounded"
        return "at most one: nothing to combine"

    def to_dict(self) -> dict[str, object]:
        return {
            "private_data": self.private_data,
            "untrusted_content": self.untrusted_content,
            "outward_communication": self.outward_communication,
            "count": self.count,
            "lamp": self.lamp,
            "verdict": self.verdict,
            "outward_via": list(self.outward_via),
            "private_via": list(self.private_via),
            "untrusted_via": self.untrusted_via,
        }


def evaluate(
    controls: dict[str, bool],
    principal: Principal | None,
    *,
    poisoned_index: bool = True,
    broker: ToolBroker | None = None,
    require_approval: bool = True,
) -> Trifecta:
    """Compute the trifecta from configuration.  Nothing here reads a request.

    ``require_approval`` is what makes the reading honest for the irreversible
    tools: a tool that stops at a human is not an outward channel the agent
    controls, so it does not light the third lamp.  Turn approval off and it
    does, which is the correct answer and a good thing to be able to show.
    """
    broker = broker or ToolBroker()
    brokered = bool(controls.get("tool_broker"))
    offered = broker.offered_tools(principal if brokered else None)

    outward = []
    for name in offered:
        if name not in OUTWARD_TOOLS:
            continue
        spec = spec_for(name)
        if brokered and require_approval and spec is not None and spec.irreversible:
            # Held for a human. Not a channel the agent can use on its own.
            continue
        outward.append(name)

    private = [name for name in offered if name in PRIVATE_DATA_TOOLS]

    # Untrusted content is a property of the corpus, not of the toggles. The
    # input scanner reduces how much of it reaches the model; it does not change
    # the fact that the index holds documents the system did not write. Saying
    # otherwise is how a control gets credit for a property it does not have --
    # and it is exactly why input_scan must not move this lamp.
    untrusted = poisoned_index
    untrusted_via = (
        "the index contains a document the system did not author"
        if poisoned_index
        else "the index holds only reviewed corpus text"
    )

    return Trifecta(
        private_data=bool(private),
        untrusted_content=untrusted,
        outward_communication=bool(outward),
        outward_via=tuple(sorted(outward)),
        private_via=tuple(sorted(private)),
        untrusted_via=untrusted_via,
    )
