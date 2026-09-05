"""PII tokenisation, both directions.

Hooks: ``on_question``, ``on_prompt``, ``on_answer``, ``on_log``.

Five design constraints, each mapping to a bullet on slide 13:

* **At capture, not at read.**  Tokenisation happens before the prompt is
  assembled and before anything is written.  The console shows the assembled
  prompt with placeholders where the values were, which is far more convincing
  than describing it.
* **Both directions.**  Mask inbound, scan outbound.  A model can reproduce PII
  it *inferred* rather than received, and the outbound scan is the only thing
  that catches that.
* **Reversible, separately held.**  The token to value map lives in the request
  context and nowhere else, and detokenisation happens only at the final render
  to the requesting user.  That separation is the decision that makes the trace
  store safe to hand to engineers.
* **The decision is logged.**  ``on_log`` records which entity classes were
  masked and how many -- never the values.  "We redacted this" is itself an
  auditable event.
* **Order matters.**  Scan before logging.  A log written before the outbound
  scan *is* the leak.

Detection is regex plus validators for the demo set.  **Say plainly that
production wants a proper NER model, and that the regex tier is a floor rather
than a solution.**  The National Insurance number check validates the format
including the prefix rules, because a pattern that matches any two letters and
six digits flags half the reference numbers in an employment statute, and a
false-positive rate like that is how a masking layer gets switched off.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from attacklab.context import RequestContext
from attacklab.controls.base import Control, ControlResult

#: Entity classes, and the token prefix each one gets.
CLASSES: dict[str, str] = {
    "ni_number": "NINO",
    "email": "EMAIL",
    "phone": "PHONE",
    "payroll_id": "PAYROLL",
    "person_name": "NAME",
}

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
#: UK mobile and landline shapes, with an optional +44.
_PHONE_RE = re.compile(r"(?:(?<=\s)|^)(?:\+44\s?7\d{3}|\(?07\d{3}\)?)\s?\d{3}\s?\d{3}\b")
_PAYROLL_RE = re.compile(r"\bPR-\d{4,10}\b")
#: Two capitalised words, which is a *weak* signal and marked as such.
_NAME_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b")

#: The National Insurance number format, and then the rules on top of it.
_NI_RE = re.compile(r"\b([A-CEGHJ-PR-TW-Z]{2})\s?(\d{2})\s?(\d{2})\s?(\d{2})\s?([A-D])\b")
#: Prefixes the format allows but which are never issued.
#:
#: Note what the character class above already excludes: D, F, I, Q, U and V
#: cannot appear in either position, and O cannot appear in the second.  That is
#: why ``QQ 12 34 56 C`` -- the placeholder in most documentation, including
#: GOV.UK's -- is *correctly rejected* here.  It is the documented example
#: precisely because it can never belong to anybody.  Test with ``AB123456C``.
_NI_DISALLOWED_PREFIXES = frozenset({"BG", "GB", "NK", "KN", "TN", "NT", "ZZ"})


def valid_ni_number(candidate: str) -> bool:
    """Whether a candidate is a well-formed National Insurance number.

    The format alone is not enough.  ``AB 12 34 56 C`` and ``BG 12 34 56 C``
    are both shaped like one, and only the first could be issued.  Checking the
    disallowed prefixes costs one set lookup and is the difference between a
    masking layer people leave on and one they turn off after it redacted a
    tribunal case number.
    """
    match = _NI_RE.fullmatch(candidate.strip())
    if match is None:
        return False
    return match.group(1).upper() not in _NI_DISALLOWED_PREFIXES


@dataclass(slots=True)
class Finding:
    """One detected entity."""

    cls: str
    value: str
    start: int
    end: int


def detect(text: str) -> list[Finding]:
    """Every entity found, left to right, longest match wins on overlap."""
    findings: list[Finding] = []
    for match in _NI_RE.finditer(text or ""):
        if valid_ni_number(match.group(0)):
            findings.append(Finding("ni_number", match.group(0), match.start(), match.end()))
    for cls, pattern in (
        ("email", _EMAIL_RE),
        ("phone", _PHONE_RE),
        ("payroll_id", _PAYROLL_RE),
        ("person_name", _NAME_RE),
    ):
        for match in pattern.finditer(text or ""):
            findings.append(Finding(cls, match.group(0), match.start(), match.end()))
    findings.sort(key=lambda f: (f.start, -(f.end - f.start)))
    kept: list[Finding] = []
    for finding in findings:
        if kept and finding.start < kept[-1].end:
            continue
        kept.append(finding)
    return kept


@dataclass(slots=True)
class Vault:
    """The token to value map, held separately from everything else.

    Separately in the strict sense that matters for this demonstration: it lives
    on the request context and is excluded from that context's ``to_dict``, so
    it reaches neither the console, nor a span, nor the audit record.  In
    production this is a different store under different access control; the
    property being demonstrated is the separation, not the storage engine.
    """

    counters: dict[str, int] = field(default_factory=dict)

    def tokenise(self, text: str) -> tuple[str, dict[str, str], dict[str, int]]:
        """``(masked text, token->value, class->count)``."""
        findings = detect(text)
        if not findings:
            return text, {}, {}
        mapping: dict[str, str] = {}
        counts: dict[str, int] = {}
        out: list[str] = []
        cursor = 0
        for finding in findings:
            prefix = CLASSES[finding.cls]
            self.counters[prefix] = self.counters.get(prefix, 0) + 1
            token = f"[{prefix}_{self.counters[prefix]}]"
            mapping[token] = finding.value
            counts[finding.cls] = counts.get(finding.cls, 0) + 1
            out.append(text[cursor : finding.start])
            out.append(token)
            cursor = finding.end
        out.append(text[cursor:])
        return "".join(out), mapping, counts

    @staticmethod
    def detokenise(text: str, mapping: dict[str, str]) -> str:
        """Put the values back.

        Called at exactly one place: the final render to the requesting user.
        Anywhere else and the separation is decorative.
        """
        for token, value in mapping.items():
            text = text.replace(token, value)
        return text


class PiiMask(Control):
    """Cross-cutting.  Partly deterministic, and honest about which part.

    The *tokenisation* is deterministic: given the same text it produces the
    same tokens.  The *detection* is not, in the sense that matters -- a regex
    tier misses a name it has never seen and flags a reference number that
    looks like a NINO.  Marked non-deterministic on the console for that reason,
    because calling regex detection deterministic would be true about the code
    and misleading about the control.
    """

    key = "pii_mask"
    layer = 0
    deterministic = False

    def __init__(self) -> None:
        self.vault = Vault()

    def _run(self, value: Any, ctx: RequestContext) -> ControlResult:
        """Dispatch on the hook, which the stack encodes in the value shape."""
        if isinstance(value, tuple) and len(value) == 2 and value[0] == "outbound":
            return self._scan_outbound(str(value[1]), ctx)
        return self._mask_inbound(str(value), ctx)

    def _mask_inbound(self, text: str, ctx: RequestContext) -> ControlResult:
        masked, mapping, counts = self.vault.tokenise(text)
        if not mapping:
            return ControlResult(passed=True, reason="no PII detected inbound")
        ctx.pii_map.update(mapping)
        for cls, count in counts.items():
            ctx.pii_classes[cls] = ctx.pii_classes.get(cls, 0) + count
        return ControlResult(
            passed=True,
            reason=(
                "masked at capture: "
                + ", ".join(f"{count} {cls}" for cls, count in sorted(counts.items()))
            ),
            mutated=masked,
            matched=", ".join(sorted(mapping)),
        )

    def _scan_outbound(self, text: str, ctx: RequestContext) -> ControlResult:
        """The other direction: block an answer carrying raw PII.

        A **judgement**, not a rewrite, and the difference is deliberate.
        Silently scrubbing an answer would hide the fact that the model produced
        PII it was never given, which is the single most useful thing this scan
        tells you.
        """
        # Tokens the vault issued are not leaks; they are the masking working.
        stripped = text
        for token in ctx.pii_map:
            stripped = stripped.replace(token, "")
        findings = detect(stripped)
        # A mailbox quoted from the corpus is not the user's personal data. Only
        # values the vault has never seen, in classes that identify a person.
        known = {value.lower() for value in ctx.pii_map.values()}
        leaks = [
            finding
            for finding in findings
            if finding.value.lower() not in known
            and finding.cls in {"ni_number", "phone", "payroll_id", "person_name"}
        ]
        if not leaks:
            return ControlResult(passed=True, reason="outbound scan: no unmasked PII in the answer")
        first = leaks[0]
        return ControlResult(
            passed=False,
            reason=(
                f"outbound scan: the answer contains an unmasked {first.cls} that was not "
                "in the request -- the model reproduced it rather than received it"
            ),
            matched=f"<{first.cls}>",
        )

    def log_fields(self, ctx: RequestContext) -> dict[str, Any]:
        """What ``on_log`` records: classes and counts, never values."""
        if not ctx.pii_classes:
            return {}
        return {
            "masked_classes": dict(sorted(ctx.pii_classes.items())),
            "masked_total": sum(ctx.pii_classes.values()),
            "vault_tokens": len(ctx.pii_map),
        }
