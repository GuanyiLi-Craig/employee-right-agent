"""Diff the functions actually exposed to the model against a reviewed allowlist.

**Treat this as the highest value-per-line check in the repository.**  Before
least privilege matters at all, the set of functions the model can see has to be
the set you intended it to see.

The motivating case is **CVE-2026-25592**, disclosed by Microsoft in May 2026
against their own Semantic Kernel.  That framework runs agent-generated Python
inside an isolated cloud sandbox -- a real boundary, correctly built -- and
provides host-side helper functions to move files across it.  One of those
helpers, the file download, was **accidentally annotated as an AI-callable
tool**.  The annotation advertised it to the model along with its parameter
schema, including the destination path on the host.

Two prompts were then enough: write a payload inside the sandbox where it is
harmless, then ask the agent to "download" it to the host's startup folder.
Next sign-in, full compromise.

**The sandbox held.  The tool surface did not.**  No hypervisor exploit, no
memory corruption -- a decorator on the wrong function.  A companion CVE in the
same disclosure showed an AST blocklist guarding an ``eval()`` sink being walked
around via Python's type hierarchy, which is the same lesson one layer down: a
blocklist is not a boundary.

This check is about forty lines of reflection over the framework's registry and
it would have caught both.  The allowlist is under code review like any other
permission grant -- because that is what it is.

And the practical instruction from slide 10 is embarrassingly simple: go and
read your own tool registry, and check that every function the model can see is
one you meant it to see.
"""

from __future__ import annotations

from attacklab.supplychain.findings import FAIL, PASS, Finding
from attacklab.tools import REVIEWED_ALLOWLIST, RUNTIME_SURFACE, spec_for


def runtime_surface() -> list[str]:
    """Every function actually exposed to the model, by reflection over the registry.

    In this lab the registry is a tuple, so "reflection" is one comprehension.
    In a real framework it is ``kernel.plugins``, or the set of functions
    carrying the framework's tool decorator, and the check is the same shape:
    enumerate what is *advertised*, not what is documented.
    """
    return [spec.name for spec in RUNTIME_SURFACE]


def scan() -> list[Finding]:
    """Fail on anything present at runtime and absent from the allowlist."""
    exposed = set(runtime_surface())
    reviewed = set(REVIEWED_ALLOWLIST)

    findings: list[Finding] = []
    for name in sorted(exposed - reviewed):
        spec = spec_for(name)
        findings.append(
            Finding(
                scanner="scan_tool_surface",
                severity=FAIL,
                subject=name,
                summary=(
                    "exposed to the model and absent from the reviewed allowlist"
                    + (f" -- {spec.description}" if spec else "")
                ),
                owasp="LLM03:2026",
                asi="ASI02",
                detail={
                    "params": list(spec.params) if spec else [],
                    "irreversible": bool(spec and spec.irreversible),
                    "note": spec.note if spec else "",
                },
                remedy=(
                    "Remove the annotation, or add the name to the allowlist in a "
                    "reviewed commit. Least privilege over a tool the model should "
                    "never have been offered is a policy applied one layer too late."
                ),
            )
        )
    for name in sorted(reviewed - exposed):
        findings.append(
            Finding(
                scanner="scan_tool_surface",
                severity=PASS,
                subject=name,
                summary="on the allowlist and no longer exposed; the allowlist is stale",
                remedy="Trim the allowlist so it still describes reality.",
            )
        )
    if not findings:
        findings.append(
            Finding(
                scanner="scan_tool_surface",
                severity=PASS,
                subject="tool surface",
                summary=f"{len(exposed)} exposed function(s), all reviewed",
            )
        )
    return findings
