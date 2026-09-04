"""Compare a skill manifest's declared purpose against the capability it requests.

**The highest-signal, lowest-effort check in the whole area**, and the framing
is worth stating: you are not detecting malice.  You are noticing that what
something *claims to do* and what it *asks for* do not match.  That is how
mobile app permission review works, and it works here.

It also has a hard limit worth naming on stage.  A malicious agent skill does
not need to exploit a deserialisation bug -- it only needs to be a working skill
that *also* does something else.  The line between "a skill that calls an API"
and "a skill that calls an API and exfiltrates credentials" is invisible to
static analysis that does not understand the intent graph, and no major public
registry offers behavioural sandboxing for skills today.  **This catches the
careless attacker.  It does not catch the careful one.**

Three 2026 data points make the case that this is now the softest target in the
stack, and one or two of them is plenty on stage:

* **LiteLLM**, March 2026: two backdoored releases, live for about **forty-six
  minutes**, roughly **47,000 downloads** in that window.  LiteLLM sits
  underneath a great deal -- it is the model gateway for CrewAI, DSPy, GraphRAG
  and a long tail of agent frameworks.  The publishing token came out of
  LiteLLM's **own CI**, which was running a security scanner that had itself
  been compromised upstream: the supply-chain attack arrived through the
  supply-chain scanner.
* One **agent skill registry** was found in February 2026 carrying **341
  malicious skills** -- about **twelve percent** of everything published there
  -- from a single campaign, distributing credential-stealing malware and
  concentrated in the categories where victims hold financial credentials.
* The first malicious **MCP server** in the wild shipped **fifteen clean
  releases** to build legitimacy before adding one line that BCC'd every email.
  Separately, core MCP infrastructure carried an RCE rated 9.6.

Same supply chain as your model files, less maturity, and a much shorter history
of anyone looking.
"""

from __future__ import annotations

import json
from pathlib import Path

from attacklab.supplychain.findings import FAIL, PASS, WARN, Finding

#: Capability -> the purposes that can justify it.
#:
#: Deliberately coarse.  The check is "does the stated purpose plausibly need
#: this", and a coarse mapping is the honest resolution of that question: a
#: fine-grained one would imply the check understands intent, which is exactly
#: the limit named in the module docstring.
JUSTIFIED_BY: dict[str, frozenset[str]] = {
    "read:leave_balance": frozenset({"leave", "holiday", "absence", "balance"}),
    "read:payroll": frozenset({"payroll", "pay", "salary", "wage"}),
    "write:leave_request": frozenset({"leave", "holiday", "absence", "request"}),
    "send:email": frozenset({"notify", "notification", "email", "message", "correspond"}),
    "net:outbound": frozenset({"fetch", "download", "sync", "integration"}),
    "fs:read": frozenset({"file", "document", "attachment", "upload"}),
    "exec:code": frozenset({"execute", "run", "sandbox", "script"}),
}

#: Capabilities that are never justified by a lookup-shaped purpose, whatever
#: the manifest says.  A holiday-balance skill has no business sending email.
ALWAYS_SUSPICIOUS: frozenset[str] = frozenset({"send:email", "net:outbound", "exec:code"})


def scan_manifest(path: Path) -> list[Finding]:
    """One manifest: does what it asks for match what it says it does?"""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [
            Finding(
                scanner="scan_skills",
                severity=FAIL,
                subject=path.name,
                summary=f"manifest is unreadable: {exc}",
                remedy="A manifest a reviewer cannot read is a skill a reviewer has not reviewed.",
            )
        ]

    name = str(manifest.get("name") or path.stem)
    description = str(manifest.get("description") or "")
    requested = [str(item) for item in (manifest.get("capabilities") or [])]
    if not requested:
        return [
            Finding(
                scanner="scan_skills",
                severity=WARN,
                subject=name,
                summary="the manifest declares no capabilities at all",
                remedy="An absent capability list is not the same as an empty one. Require it.",
            )
        ]

    words = {word.strip(".,;:").lower() for word in f"{name} {description}".split()}
    unjustified: list[str] = []
    for capability in requested:
        justifying = JUSTIFIED_BY.get(capability)
        if justifying is None:
            unjustified.append(capability)
            continue
        if not (justifying & words):
            unjustified.append(capability)

    if not unjustified:
        return [
            Finding(
                scanner="scan_skills",
                severity=PASS,
                subject=name,
                summary=(
                    f"requests {len(requested)} capability(ies), each justified by its "
                    "stated purpose"
                ),
                detail={"capabilities": requested},
            )
        ]
    severe = sorted(set(unjustified) & ALWAYS_SUSPICIOUS)
    return [
        Finding(
            scanner="scan_skills",
            severity=FAIL,
            subject=name,
            summary=(
                f"describes itself as {description.lower() or 'unspecified'} and requests "
                + ", ".join(sorted(unjustified))
                + (
                    f" -- {', '.join(severe)} cannot be justified by that purpose"
                    if severe
                    else ""
                )
            ),
            detail={
                "description": description,
                "capabilities": requested,
                "unjustified": sorted(unjustified),
            },
            remedy=(
                "Reject it, or require the author to narrow the capability. Note the "
                "limit: this catches the careless attacker, not the careful one."
            ),
        )
    ]


def scan(skills_dir: Path) -> list[Finding]:
    if not skills_dir.is_dir():
        return [
            Finding(
                scanner="scan_skills",
                severity=WARN,
                subject=str(skills_dir),
                summary="no skills directory; nothing inspected",
            )
        ]
    findings: list[Finding] = []
    for path in sorted(skills_dir.glob("*.json")):
        findings.extend(scan_manifest(path))
    return findings
