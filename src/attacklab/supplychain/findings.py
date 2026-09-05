"""One finding type, so five scanners report the same shape.

The console renders these, the CLI prints them, and the eval suite asserts on
them.  A per-scanner result type would have meant three renderers and a test
that only understood one of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Severities, worst first.  ``fail`` fails the build; ``warn`` does not.
FAIL = "fail"
WARN = "warn"
# S105 flags the literal "pass" as a possible hardcoded password. It is a
# severity label. Renaming it to appease the linter would make every rendered
# report read worse than the suppression does.
PASS = "pass"  # noqa: S105


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing a scanner found, and what it means."""

    scanner: str
    severity: str
    subject: str
    summary: str
    #: OWASP identifiers, emitted into the JUnit XML.  It costs nothing, and it
    #: means the gate's output is already in the vocabulary a security reviewer
    #: uses -- which is the difference between a report that gets read and one
    #: that gets forwarded.
    owasp: str = "LLM04:2026"
    asi: str = "ASI04"
    detail: dict[str, Any] = field(default_factory=dict)
    #: What to do about it, in one sentence. A finding without a remedy is a
    #: complaint.
    remedy: str = ""

    @property
    def failed(self) -> bool:
        return self.severity == FAIL

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanner": self.scanner,
            "severity": self.severity,
            "subject": self.subject,
            "summary": self.summary,
            "owasp": self.owasp,
            "asi": self.asi,
            "detail": dict(self.detail),
            "remedy": self.remedy,
        }

    def render(self) -> str:
        mark = {FAIL: "FAIL", WARN: "WARN", PASS: "pass"}[self.severity]
        line = f"  [{mark}] {self.subject}: {self.summary}"
        if self.remedy and self.severity != PASS:
            line += f"\n         -> {self.remedy}"
        return line


@dataclass(slots=True)
class ScanReport:
    """Every finding from one run, and whether the build should fail."""

    findings: list[Finding] = field(default_factory=list)

    def add(self, finding: Finding) -> Finding:
        self.findings.append(finding)
        return finding

    def extend(self, findings: list[Finding]) -> None:
        self.findings.extend(findings)

    @property
    def failures(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.failed]

    @property
    def ok(self) -> bool:
        return not self.failures

    def by_scanner(self) -> dict[str, list[Finding]]:
        grouped: dict[str, list[Finding]] = {}
        for finding in self.findings:
            grouped.setdefault(finding.scanner, []).append(finding)
        return grouped

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "findings": [finding.to_dict() for finding in self.findings],
            "counts": {
                severity: sum(1 for f in self.findings if f.severity == severity)
                for severity in (FAIL, WARN, PASS)
            },
        }

    def render(self) -> str:
        lines: list[str] = []
        for scanner, findings in self.by_scanner().items():
            lines.append(f"{scanner}")
            lines.extend(finding.render() for finding in findings)
            lines.append("")
        counts = self.to_dict()["counts"]
        lines.append(
            f"{counts[FAIL]} failure(s), {counts[WARN]} warning(s), {counts[PASS]} pass(es)"
        )
        return "\n".join(lines)
