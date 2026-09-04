"""Block rate, false-positive rate, latency and cost -- per control, per class.

    uv run python -m attacklab.report --all

**A block rate without a false-positive rate is a marketing number.**  Every
function here reports both, and the console renders them side by side, always.

Two other disciplines carried from session 5:

* **No number from here goes on a slide.**  Per-run measurements are read off
  the panel on the day; only fixed facts live on slides.
* **A delta needs a spread.**  :func:`provenance_delta` runs the indirect set
  repeatedly with the fence off and on and reports the difference *with* its
  spread, because §10's claim -- "structural, and nearly free" -- has two halves
  and both need numbers.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from attacklab.attacks.catalogue import RUNNABLE, Goal, Payload, Vector
from attacklab.lab import Lab
from attacklab.registry import RUNTIME_KEYS, Registry

#: Benign inputs that must not be blocked.  Loaded from the eval dataset so the
#: panel and the CI gate measure the same rows.
FALSEPOS_PATH = Path(__file__).resolve().parents[2] / "evals" / "falsepos.jsonl"


@dataclass(slots=True)
class ControlMeasurement:
    """What one control bought, and what it cost, over a set of runs."""

    key: str
    attempted: int = 0
    contained: int = 0
    #: Benign rows this control blocked.  The other half of the number.
    false_positives: int = 0
    benign_attempted: int = 0
    latency_ms: list[float] = field(default_factory=list)
    cost_usd: list[float] = field(default_factory=list)

    @property
    def block_rate(self) -> float:
        return round(self.contained / self.attempted, 4) if self.attempted else 0.0

    @property
    def false_positive_rate(self) -> float:
        if not self.benign_attempted:
            return 0.0
        return round(self.false_positives / self.benign_attempted, 4)

    @property
    def added_latency_ms(self) -> float:
        return round(statistics.mean(self.latency_ms), 3) if self.latency_ms else 0.0

    @property
    def added_cost_usd(self) -> float:
        return round(statistics.mean(self.cost_usd), 8) if self.cost_usd else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "attempted": self.attempted,
            "contained": self.contained,
            "block_rate": self.block_rate,
            "false_positives": self.false_positives,
            "benign_attempted": self.benign_attempted,
            "false_positive_rate": self.false_positive_rate,
            "added_latency_ms": self.added_latency_ms,
            "added_cost_usd": self.added_cost_usd,
        }


def _read_falsepos() -> list[dict[str, Any]]:
    if not FALSEPOS_PATH.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in FALSEPOS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def measure_control(
    lab: Lab, key: str, payloads: tuple[Payload, ...] = RUNNABLE
) -> ControlMeasurement:
    """One control against the catalogue, and against the benign set."""
    measurement = ControlMeasurement(key=key)
    for payload in payloads:
        bare = lab.run(payload, controls={})
        if not bare.escaped:
            # A payload nothing has to stop tells you nothing about this control.
            # Counting it as a block is how a block rate becomes a marketing
            # number: it rewards the control for containment it did not provide.
            continue
        run = lab.run(payload, controls={key: True})
        measurement.attempted += 1
        if run.contained:
            measurement.contained += 1
        measurement.latency_ms.append(run.latency_by_layer.get(key, 0.0))
        measurement.cost_usd.append(run.cost_by_layer.get(key, 0.0))

    for row in _read_falsepos():
        measurement.benign_attempted += 1
        run = lab.run_question(str(row["question"]), controls={key: True})
        if key in run.blocked_by:
            measurement.false_positives += 1
    return measurement


def provenance_delta(
    lab: Lab, *, repeats: int = 5
) -> dict[str, Any]:
    """The layer-2 delta, with its spread.

    §10's claim is "structural, and nearly free".  Both halves need numbers, and
    a single run gives neither: the fence changes how often an indirect payload
    is followed, and *how often* is a rate.  Repeated runs, reported with the
    spread rather than as one figure.
    """
    indirect = tuple(p for p in RUNNABLE if p.vector is Vector.INDIRECT)
    off_rates: list[float] = []
    on_rates: list[float] = []
    added_ms: list[float] = []
    added_cost: list[float] = []
    for _ in range(repeats):
        off = [lab.run(p, controls={}) for p in indirect]
        on = [lab.run(p, controls={"provenance": True}) for p in indirect]
        off_rates.append(sum(1 for r in off if r.escaped) / len(indirect))
        on_rates.append(sum(1 for r in on if r.escaped) / len(indirect))
        added_ms.extend(r.latency_by_layer.get("provenance", 0.0) for r in on)
        added_cost.extend(r.cost_by_layer.get("provenance", 0.0) for r in on)

    def spread(values: list[float]) -> dict[str, float]:
        return {
            "mean": round(statistics.mean(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        }

    return {
        "repeats": repeats,
        "payloads": [p.id for p in indirect],
        "escape_rate_fence_off": spread(off_rates),
        "escape_rate_fence_on": spread(on_rates),
        "delta_mean": round(statistics.mean(off_rates) - statistics.mean(on_rates), 4),
        "added_latency_ms": spread(added_ms),
        "added_cost_usd": spread(added_cost),
        "note": (
            "Reported with the spread, not as a single number. The fence is a signal, "
            "not a boundary: it changes how often a payload is followed, and that is a "
            "rate with variance rather than a guarantee."
        ),
    }


def action_goal_guarantee(lab: Lab) -> dict[str, Any]:
    """Every ACTION-goal payload with only the broker on.

    Not a threshold -- a hard assertion.  The tool broker is the only
    deterministic control in the lab, and a single escape here means the gate
    has a hole rather than that a rate drifted.
    """
    action = tuple(p for p in RUNNABLE if p.goal is Goal.ACTION)
    escapes: list[str] = []
    for payload in action:
        run = lab.run(payload, controls={"tool_broker": True})
        if run.escaped:
            escapes.append(payload.id)
    return {
        "payloads": [p.id for p in action],
        "escapes": escapes,
        "ok": not escapes,
        "note": "Zero escapes is an assertion, not a threshold. See evals/thresholds.json.",
    }


@dataclass(slots=True)
class Report:
    """Everything the panel and the gate read."""

    controls: dict[str, dict[str, Any]] = field(default_factory=dict)
    combined: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    broker: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "controls": self.controls,
            "combined": self.combined,
            "provenance_delta": self.provenance,
            "action_goal_guarantee": self.broker,
        }

    def render(self) -> str:
        lines = [
            f"{'control':16}{'block':>8}{'fp':>8}{'+ms':>10}{'+usd':>14}",
            "-" * 56,
        ]
        for key, row in self.controls.items():
            lines.append(
                f"{key:16}{row['block_rate']:>8.2f}{row['false_positive_rate']:>8.2f}"
                f"{row['added_latency_ms']:>10.2f}{row['added_cost_usd']:>14.8f}"
            )
        lines.append("")
        combined = self.combined
        if combined:
            lines.append(
                f"all controls on: block rate {combined['block_rate']:.2f}, "
                f"false-positive rate {combined['false_positive_rate']:.2f} "
                f"({combined['contained']}/{combined['attempted']} contained, "
                f"{combined['false_positives']}/{combined['benign_attempted']} benign blocked)"
            )
        if self.broker:
            verdict = "held" if self.broker["ok"] else f"ESCAPED: {self.broker['escapes']}"
            lines.append(f"tool_broker over every ACTION-goal payload: {verdict}")
        if self.provenance:
            off = self.provenance["escape_rate_fence_off"]
            on = self.provenance["escape_rate_fence_on"]
            lines.append(
                f"provenance delta over {self.provenance['repeats']} repeats: "
                f"escape {off['mean']:.2f} (±{off['stdev']:.2f}) -> "
                f"{on['mean']:.2f} (±{on['stdev']:.2f}); "
                f"+{self.provenance['added_latency_ms']['mean']:.2f}ms, "
                f"+${self.provenance['added_cost_usd']['mean']:.8f}"
            )
        lines.append("")
        lines.append(
            "Read on the day. No number here goes on a slide -- only fixed facts do."
        )
        return "\n".join(lines)


def build(lab: Lab | None = None, *, repeats: int = 3) -> Report:
    lab = lab or Lab(Registry(), poisoned=True)
    report = Report()
    for key in sorted(RUNTIME_KEYS):
        report.controls[key] = measure_control(lab, key).to_dict()

    everything = dict.fromkeys(RUNTIME_KEYS, True)
    combined = ControlMeasurement(key="all")
    for payload in RUNNABLE:
        if not lab.run(payload, controls={}).escaped:
            continue
        combined.attempted += 1
        run = lab.run(payload, controls=everything)
        if run.contained:
            combined.contained += 1
        combined.latency_ms.append(sum(run.latency_by_layer.values()))
        combined.cost_usd.append(sum(run.cost_by_layer.values()))
    for row in _read_falsepos():
        combined.benign_attempted += 1
        if lab.run_question(str(row["question"]), controls=everything).blocked_by:
            combined.false_positives += 1
    report.combined = combined.to_dict()

    report.provenance = provenance_delta(lab, repeats=repeats)
    report.broker = action_goal_guarantee(lab)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Block-rate and cost aggregation.")
    parser.add_argument("--all", action="store_true", help="run the whole catalogue")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--repeats", type=int, default=3, help="repeats for the provenance delta")
    parser.add_argument("--clean", action="store_true", help="use the clean index")
    args = parser.parse_args(argv)

    lab = Lab(Registry(), poisoned=not args.clean)
    report = build(lab, repeats=args.repeats)
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.render())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
