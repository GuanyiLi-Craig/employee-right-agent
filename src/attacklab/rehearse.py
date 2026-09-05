"""Prove all five demos land, headless.

    make session6-check

Milestone 10 of the build order: *every demo runs twice, cold, in under ninety
seconds each.*  This is that check, as code, so it can be run before the room
fills rather than remembered.

Each demo asserts the beat the deck depends on, and prints the number the
presenter will point at.  It exits non-zero if any beat fails, so it is also
usable as a pre-flight in CI.

What it deliberately does **not** do is check the numbers against thresholds --
that is ``make adversarial``.  This answers one question: *will the five things
I am about to do in front of people actually happen?*
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

from attacklab.attacks.catalogue import DEMO_QUESTION
from attacklab.console.app import PII_QUESTION
from attacklab.lab import Lab
from attacklab.registry import PRESETS, Registry
from attacklab.supplychain.__main__ import run_all as run_supplychain

#: Ninety seconds per demo, from the build order's last milestone.
BUDGET_S = 90.0


@dataclass
class Beat:
    """One thing that has to happen, and whether it did."""

    demo: str
    beat: str
    ok: bool
    detail: str = ""

    def render(self) -> str:
        return f"  [{'ok  ' if self.ok else 'FAIL'}] {self.beat}\n         {self.detail}"


@dataclass
class Rehearsal:
    beats: list[Beat] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def check(self, demo: str, beat: str, ok: bool, detail: str = "") -> Beat:
        found = Beat(demo=demo, beat=beat, ok=bool(ok), detail=detail)
        self.beats.append(found)
        return found

    @property
    def ok(self) -> bool:
        return all(beat.ok for beat in self.beats) and all(
            seconds <= BUDGET_S for seconds in self.timings.values()
        )

    def render(self) -> str:
        lines: list[str] = []
        current = ""
        for beat in self.beats:
            if beat.demo != current:
                current = beat.demo
                seconds = self.timings.get(current)
                stamp = f"  [{seconds:.1f}s]" if seconds else ""
                lines.append(f"\n{current}{stamp}")
            lines.append(beat.render())
        failed = [beat for beat in self.beats if not beat.ok]
        over = {name: s for name, s in self.timings.items() if s > BUDGET_S}
        lines.append("")
        lines.append(
            f"{len(self.beats) - len(failed)}/{len(self.beats)} beats land"
            + (f"; OVER BUDGET: {over}" if over else "")
        )
        if failed:
            lines.append("Do not present until these are green:")
            lines.extend(f"  - {beat.demo}: {beat.beat}" for beat in failed)
        return "\n".join(lines)


def rehearse(*, verbose: bool = True) -> Rehearsal:
    r = Rehearsal()
    lab = Lab(Registry(), poisoned=True)
    clean = Lab(Registry(), poisoned=False)

    # ---- Demo 1: one poisoned provision -------------------------------------
    started = time.perf_counter()
    lab.install()
    run = lab.run("p01", controls=PRESETS["all_off"])
    r.timings["DEMO 1 - one poisoned provision"] = time.perf_counter() - started
    d = "DEMO 1 - one poisoned provision"
    r.check(d, "the poisoned provision is retrieved and cited",
            any("43KA" in c for c in run.citations),
            f"citations: {run.citations}")
    r.check(d, "the model obeys it and attempts a tool call",
            bool(run.tools), f"tools: {[(t['name'], t['outcome']) for t in run.tools]}")
    r.check(d, "the payload ESCAPES with every control off", run.escaped, run.verdict)
    r.check(d, "nothing failed: the gate answered rather than refused",
            not run.refused, f"sufficiency {run.sufficiency:.3f}, refused={run.refused}")
    scores = run.scores
    r.check(d, "every Session 5 quality metric reports success",
            all(float(scores.get(k, 0)) >= 0.7 for k in
                ("groundedness", "context_relevance", "answer_relevance")),
            f"{ {k: round(float(v), 2) for k, v in scores.items() if isinstance(v, (int, float))} }")
    r.check(d, "the Rule of Two lamp is RED", run.trifecta["lamp"] == "red",
            f"{run.trifecta['count']}/3 - {run.trifecta['verdict']}")

    # ---- Demo 2: guardrails, and what they do not stop ----------------------
    started = time.perf_counter()
    caught = lab.run("p01", controls=PRESETS["layers_1_3"])
    rephrased = lab.run("p07", controls=PRESETS["layers_1_3"])
    scan_only = lab.run("p07", controls={"input_scan": True})
    r.timings["DEMO 2 - guardrails, and what they do not stop"] = time.perf_counter() - started
    d = "DEMO 2 - guardrails, and what they do not stop"
    r.check(d, "the obvious payload is caught by input scanning",
            "input_scan" in caught.blocked_by, caught.verdict)
    r.check(d, "the REPHRASED payload gets through input scanning",
            "input_scan" not in scan_only.blocked_by and scan_only.escaped,
            f"p07 with input_scan only: {scan_only.verdict}")
    r.check(d, "and is stopped on the way out",
            "output_verify" in rephrased.blocked_by, rephrased.verdict)
    r.check(d, "the fence is visible in the assembled prompt",
            "attacklab.fence" in rephrased.sent_user,
            f"{rephrased.sent_user.count('attacklab.fence') // 2} block(s) fenced")
    r.check(d, "layer 2 is nearly free and layer 3 is not",
            rephrased.cost_by_layer.get("provenance", 0.0) == 0.0
            and rephrased.cost_by_layer.get("output_verify", 0.0) > 0.0,
            f"cost by layer: {rephrased.cost_by_layer}")

    # ---- Demo 3: the supply chain you did not audit -------------------------
    started = time.perf_counter()
    report = run_supplychain(lab.settings)
    r.timings["DEMO 3 - the supply chain you did not audit"] = time.perf_counter() - started
    d = "DEMO 3 - the supply chain you did not audit"
    by_subject = {f.subject: f for f in report.findings}
    r.check(d, "the pickle model is flagged",
            any(f.failed and "tiny.pt" in f.subject for f in report.findings),
            "REDUCE plus a global a state dict does not need")
    r.check(d, "its safetensors twin passes",
            any(f.severity == "pass" and "safetensors" in f.subject for f in report.findings),
            "same weights, different container")
    r.check(d, "the poisoned document is found by CONTENT",
            any(f.scanner == "scan_index" and f.failed and "43KA" in f.subject
                for f in report.findings),
            "the manifest flag is never read - see scan_index's docstring")
    r.check(d, "the clean index comes back clean",
            any(f.scanner == "scan_index" and f.severity == "pass" for f in report.findings),
            "a hash tells you the corpus changed, not that what changed was hostile")
    r.check(d, "the overreaching skill manifest is flagged",
            any(f.scanner == "scan_skills" and f.failed for f in report.findings),
            "holiday-balance lookup requesting outbound email")
    r.check(d, "the moved digest fails the lockfile check",
            any(f.scanner == "scan_lock" and f.failed for f in report.findings),
            "and one artefact pinned by a mutable tag")
    r.check(d, "the accidental tool is caught by the surface diff",
            "download_file_to_host" in by_subject,
            "CVE-2026-25592: the sandbox held, the tool surface did not")

    # ---- Demo 4: the same attack, made impossible ---------------------------
    started = time.perf_counter()
    least = lab.run("p01", controls=PRESETS["least_privilege"])
    admin = lab.run("p01", controls=PRESETS["least_privilege"], persona_name="hr_admin")
    r.timings["DEMO 4 - the same attack, made impossible"] = time.perf_counter() - started
    d = "DEMO 4 - the same attack, made impossible"
    r.check(d, "layers 1-3 are visibly OFF",
            not any(least.controls[k] for k in ("input_scan", "provenance", "output_verify")),
            f"on: {[k for k, v in least.controls.items() if v]}")
    r.check(d, "the same payload, byte-identical, is CONTAINED", least.contained, least.verdict)
    r.check(d, "the model is STILL FOOLED - the instruction is in the trace",
            bool(least.directives) and "compliance mailbox" in least.answer,
            (least.directives[0]["sentence"][:110] if least.directives else "no directive found"))
    r.check(d, "the tool list shrank from thirteen to three",
            len(least.tools_offered) == 3, f"offered: {least.tools_offered}")
    r.check(d, "the denial is in the audit chain and the chain verifies",
            least.audit_verified and any(t["outcome"] == "deny" for t in least.tools),
            f"denial: {[t['reason'] for t in least.tools]}")
    r.check(d, "the lamp goes amber for the employee",
            least.trifecta["lamp"] == "amber",
            f"{least.trifecta['count']}/3 - {least.trifecta['verdict']}")
    r.check(d, "and the SAME payload is denied on a PARAMETER for the administrator",
            admin.contained and any("outside the tenant" in t["reason"] for t in admin.tools),
            f"{[t['reason'] for t in admin.tools]}")

    # ---- Demo 5: two problems, two different fixes --------------------------
    started = time.perf_counter()
    clean.install()
    masked = clean.run_question(PII_QUESTION, controls={"pii_mask": True})
    refused = clean.run_question(
        DEMO_QUESTION, controls={"residency": True}, region="us-east-1"
    )
    r.timings["DEMO 5 - two problems, two different fixes"] = time.perf_counter() - started
    d = "DEMO 5 - two problems, two different fixes"
    raw = ("AB123456C", "PR-0000042", "07700 900123", "jane.doe@example.gov.uk", "Mrs Jane Doe")
    leaked_prompt = [value for value in raw if value in masked.sent_user]
    leaked_answer = [value for value in raw if value in masked.answer]
    r.check(d, "the assembled prompt holds placeholders, not values",
            not leaked_prompt, f"leaked into the prompt: {leaked_prompt or 'nothing'}")
    r.check(d, "the answer holds no raw values either",
            not leaked_answer, f"leaked into the answer: {leaked_answer or 'nothing'}")
    r.check(d, "every entity class was masked at capture",
            len([e for e in masked.events if e["key"] == "pii_mask" and e["rewrote"]]) > 0,
            next((e["reason"] for e in masked.events if e["key"] == "pii_mask"), ""))
    r.check(d, "an unsupported region REFUSES rather than falling back",
            refused.refused and not refused.model,
            f"{refused.answer[:120]}")
    r.check(d, "and no model was called",
            not refused.model, f"model: {refused.model or '(none)'}")

    if verbose:
        print(r.render())
    return r


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    result = rehearse(verbose=not args.quiet)
    if args.quiet:
        print(result.render())
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
