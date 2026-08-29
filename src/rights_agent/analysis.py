"""Analyses over recorded metrics rows: drift and repricing.

Both read ``runs/metrics.jsonl`` and nothing else.  That is the point of the
file: one append-only row per request is enough to answer "is it getting worse"
and "what would it cost on a different model" without a second store.

Three things get called drift, and separating them is worth real money because
the investigations are completely different:

* **Corpus drift** -- the source documents changed under the index.  Re-index;
  add effective-date filters.
* **Intent drift** -- the *questions* moved.  Nothing is broken; attention
  moved.  This is what :func:`psi_report` measures.
* **Performance drift** -- the scores fell.

The naming matters: "data drift" in the wider ML sense means the *input*
distribution moved, which here is intent drift, so calling the corpus one "data
drift" collides with how most people already use the term.  Quality drift can be
caused by either of the others, or by the model changing underneath you --
re-indexing does not fix a prompt regression, and prompt engineering does not fix
a corpus that moved.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any, Mapping, Sequence

from rights_agent.config import PRICING, PRICING_AS_OF, cost_usd, price_for
from rights_agent.metrics import percentile

#: Smoothing floor for categories absent from a window.  PSI divides by the
#: baseline probability, so an unseen category sends the formula to infinity;
#: every implementation smooths, which keeps the number finite and makes it
#: depend on this constant.  It is therefore named, reported alongside the
#: figure it produced, and never quoted on its own.
PSI_EPSILON = 1e-4

#: Interpretation bands from credit-risk modelling, where PSI comes from.
#: A useful convention, not a law of nature -- and not calibrated on intent
#: distributions of a few dozen requests.
PSI_STABLE = 0.10
PSI_SIGNIFICANT = 0.25


def _floats(rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    return [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
    ]


def _scores(rows: Sequence[dict[str, Any]], name: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = (row.get("scores") or {}).get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append(float(value))
    return out


def _window_stats(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {
        "requests": float(len(rows)),
        "ttft_p95": round(percentile(_floats(rows, "ttft_ms"), 0.95), 2),
        "e2e_p95": round(percentile(_floats(rows, "e2e_ms"), 0.95), 2),
        "itl_p95": round(percentile(_floats(rows, "itl_ms_mean"), 0.95), 2),
        "groundedness_mean": round(mean(_scores(rows, "groundedness")), 4)
        if _scores(rows, "groundedness")
        else 0.0,
        "citation_coverage_mean": round(mean(_scores(rows, "citation_coverage")), 4)
        if _scores(rows, "citation_coverage")
        else 0.0,
        "citation_coverage_p10": round(percentile(_scores(rows, "citation_coverage"), 0.10), 4),
        "refusal_rate": round(sum(1 for row in rows if row.get("refused")) / len(rows), 4)
        if rows
        else 0.0,
        "degraded_share": round(sum(1 for row in rows if row.get("degraded")) / len(rows), 4)
        if rows
        else 0.0,
        "cost_per_request_usd": round(mean(_floats(rows, "cost_usd")), 6)
        if _floats(rows, "cost_usd")
        else 0.0,
    }


def _intent_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        intent = str(row.get("intent") or "unknown")
        counts[intent] = counts.get(intent, 0) + 1
    return dict(sorted(counts.items()))


def _intent_mix(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    counts = _intent_counts(rows)
    total = sum(counts.values()) or 1
    return {intent: round(count / total, 4) for intent, count in counts.items()}


def _distribution(counts: Mapping[str, int]) -> dict[str, float]:
    total = sum(counts.values()) or 1
    return {key: value / total for key, value in counts.items()}


def psi(
    earlier: Mapping[str, float], later: Mapping[str, float], epsilon: float = PSI_EPSILON
) -> float:
    """Population Stability Index between two categorical distributions.

    ``sum((p_later - p_earlier) * ln(p_later / p_earlier))`` over the union of
    categories, with absent categories floored at ``epsilon``.  The floor is why
    a single new category can dominate the result: see :func:`psi_report`.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive: PSI divides by the baseline probability")
    total = 0.0
    for key in set(earlier) | set(later):
        p_earlier = max(float(earlier.get(key, 0.0)), epsilon)
        p_later = max(float(later.get(key, 0.0)), epsilon)
        total += (p_later - p_earlier) * math.log(p_later / p_earlier)
    return round(total, 6)


def psi_band(value: float) -> str:
    if value < PSI_STABLE:
        return "stable"
    if value < PSI_SIGNIFICANT:
        return "moderate"
    return "significant"


@dataclass(frozen=True, slots=True)
class PsiReport:
    """Two numbers and a list, deliberately.

    Quoting one PSI figure for a window that contains new categories quotes your
    smoothing constant as much as your data.  So this reports:

    * ``psi_known`` -- the shift among intents present in **both** windows,
      renormalised over that shared support.  Epsilon-free, and comparable
      between runs.
    * ``psi_with_unseen`` -- the whole union, smoothed.  Reported *with* the
      epsilon that produced it, because it moves when the epsilon moves.
    * ``new_intents`` / ``vanished_intents`` -- the explicit lists.  Usually the
      more actionable finding, and they need no threshold at all: a topic that
      did not exist last week is interesting regardless of what any index says.

    And the framing that keeps the report useful: this is **not a bug report**.
    It says the questions changed. Paging someone with "PSI alert" sends them to
    look at the model, and the model is fine.
    """

    psi_known: float
    psi_known_band: str
    psi_with_unseen: float
    psi_with_unseen_band: str
    epsilon: float
    shared_intents: list[str]
    new_intents: list[str]
    vanished_intents: list[str]
    earlier: dict[str, float]
    later: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        lines = [
            f"PSI over intents in BOTH windows   {self.psi_known:>8.4f}  ({self.psi_known_band})",
            f"PSI including unseen categories    {self.psi_with_unseen:>8.4f}  "
            f"({self.psi_with_unseen_band}, epsilon={self.epsilon:g})",
        ]
        if self.new_intents:
            lines.append(f"NEW intents                        {', '.join(self.new_intents)}")
        else:
            lines.append("NEW intents                        (none)")
        if self.vanished_intents:
            lines.append(f"intents that stopped appearing     {', '.join(self.vanished_intents)}")
        lines += [
            "",
            "The two figures differ because PSI divides by the baseline probability: a "
            "category absent from the baseline sends it to infinity, so implementations "
            "smooth with an epsilon. That keeps the number finite and makes it depend on "
            "the constant — a single new intent can dominate the score.",
            "",
            "Use the first figure for the shift among known intents, and the new-intent "
            "list for everything else. The list needs no threshold and is usually the more "
            "actionable finding: take it to whoever owns the corpus.",
            "",
            f"Bands (<{PSI_STABLE} stable, >{PSI_SIGNIFICANT} significant) come from "
            "credit-risk modelling. A useful convention, not a law of nature.",
        ]
        return "\n".join(lines)


def psi_report(
    earlier_counts: Mapping[str, int],
    later_counts: Mapping[str, int],
    epsilon: float = PSI_EPSILON,
) -> PsiReport:
    """PSI split into the epsilon-free part and the epsilon-dependent part."""
    earlier = _distribution(earlier_counts)
    later = _distribution(later_counts)

    shared = sorted(set(earlier) & set(later))
    new = sorted(set(later) - set(earlier))
    vanished = sorted(set(earlier) - set(later))

    # Renormalise over the shared support so the epsilon plays no part at all.
    earlier_shared = _distribution({key: earlier_counts[key] for key in shared})
    later_shared = _distribution({key: later_counts[key] for key in shared})

    known = psi(earlier_shared, later_shared, epsilon) if shared else 0.0
    with_unseen = psi(earlier, later, epsilon)
    return PsiReport(
        psi_known=known,
        psi_known_band=psi_band(known),
        psi_with_unseen=with_unseen,
        psi_with_unseen_band=psi_band(with_unseen),
        epsilon=epsilon,
        shared_intents=shared,
        new_intents=new,
        vanished_intents=vanished,
        earlier={k: round(v, 4) for k, v in sorted(earlier.items())},
        later={k: round(v, 4) for k, v in sorted(later.items())},
    )


@dataclass
class DriftReport:
    """Earlier window against later window.

    What two correlated series can tell you: *when* something changed, and that
    the symptoms share a cause.  What they cannot tell you: which model
    answered.  Only the trace knows that, and saying so is part of reading the
    report honestly.
    """

    earlier: dict[str, float]
    later: dict[str, float]
    deltas: dict[str, float]
    earlier_intents: dict[str, float]
    later_intents: dict[str, float]
    intent_shift: dict[str, float]
    psi: PsiReport | None = None
    signals: list[str] = field(default_factory=list)
    caveat: str = (
        "Correlated latency and quality movements identify a window and a shared cause. "
        "They do not identify which model answered -- open the trace for that."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "earlier": self.earlier,
            "later": self.later,
            "deltas": self.deltas,
            "earlier_intents": self.earlier_intents,
            "later_intents": self.later_intents,
            "intent_shift": self.intent_shift,
            "psi": self.psi.to_dict() if self.psi else None,
            "signals": self.signals,
            "caveat": self.caveat,
        }

    def render(self) -> str:
        keys = [
            "requests",
            "ttft_p95",
            "e2e_p95",
            "citation_coverage_mean",
            "citation_coverage_p10",
            "groundedness_mean",
            "refusal_rate",
            "degraded_share",
            "cost_per_request_usd",
        ]
        lines = [f"{'metric':<26}{'earlier':>12}{'later':>12}{'delta':>12}"]
        for key in keys:
            lines.append(
                f"{key:<26}{self.earlier.get(key, 0):>12.4f}{self.later.get(key, 0):>12.4f}"
                f"{self.deltas.get(key, 0):>+12.4f}"
            )
        if self.intent_shift:
            lines += ["", f"{'intent':<26}{'earlier':>12}{'later':>12}{'shift':>12}"]
            for intent, shift in sorted(self.intent_shift.items(), key=lambda kv: -abs(kv[1])):
                lines.append(
                    f"{intent:<26}{self.earlier_intents.get(intent, 0.0):>12.4f}"
                    f"{self.later_intents.get(intent, 0.0):>12.4f}{shift:>+12.4f}"
                )
        if self.psi is not None:
            lines += ["", self.psi.render()]
        lines += ["", "signals:"]
        lines += [f"  - {signal}" for signal in self.signals] or ["  (none)"]
        lines += ["", self.caveat]
        return "\n".join(lines)


def drift_report(rows: Sequence[dict[str, Any]], split: float = 0.5) -> DriftReport:
    """Split the recorded rows in two and compare the halves."""
    if len(rows) < 4:
        return DriftReport(
            earlier={},
            later={},
            deltas={},
            earlier_intents={},
            later_intents={},
            intent_shift={},
            signals=[f"only {len(rows)} rows recorded; generate some traffic first"],
        )
    boundary = max(1, int(len(rows) * split))
    earlier_rows, later_rows = rows[:boundary], rows[boundary:]
    earlier, later = _window_stats(earlier_rows), _window_stats(later_rows)
    deltas = {key: round(later[key] - earlier[key], 4) for key in earlier}

    earlier_intents, later_intents = _intent_mix(earlier_rows), _intent_mix(later_rows)
    intent_shift = {
        intent: round(later_intents.get(intent, 0.0) - earlier_intents.get(intent, 0.0), 4)
        for intent in set(earlier_intents) | set(later_intents)
    }
    stability = psi_report(_intent_counts(earlier_rows), _intent_counts(later_rows))

    signals: list[str] = []
    if deltas["ttft_p95"] > 0.25 * max(earlier["ttft_p95"], 1e-9):
        signals.append(
            f"TTFT p95 rose {deltas['ttft_p95']:+.1f} ms "
            f"({earlier['ttft_p95']:.1f} → {later['ttft_p95']:.1f})"
        )
    if deltas["citation_coverage_mean"] < -0.05:
        signals.append(
            f"citation coverage fell {deltas['citation_coverage_mean']:+.3f} "
            f"({earlier['citation_coverage_mean']:.3f} → {later['citation_coverage_mean']:.3f})"
        )
    if deltas["ttft_p95"] > 0 and deltas["citation_coverage_mean"] < -0.05:
        signals.append(
            "latency up and citation coverage down in the same window: the signature of a "
            "primary model failing over to a weaker fallback"
        )
    if deltas["refusal_rate"] > 0.1:
        signals.append(f"refusal rate rose {deltas['refusal_rate']:+.3f}")
    big_shift = {intent: shift for intent, shift in intent_shift.items() if abs(shift) >= 0.15}
    if big_shift:
        signals.append(
            "intent mix moved: "
            + ", ".join(f"{intent} {shift:+.2f}" for intent, shift in sorted(big_shift.items()))
            + " — quality changes may be a change in the questions, not the system"
        )
    if stability.new_intents:
        signals.append(
            "new intents with no baseline: "
            + ", ".join(stability.new_intents)
            + " — a content investigation, not a bug report"
        )
    quality_fell = (
        deltas["citation_coverage_mean"] < -0.05 or deltas["groundedness_mean"] < -0.05
    )
    if quality_fell and (big_shift or stability.new_intents):
        signals.append(
            "intent mix moved AND quality fell in the same window: suspect a coverage gap "
            "rather than a regression — users started asking about something the index "
            "covers badly. Fix content, not temperature."
        )
    if not signals:
        signals.append("no movement above the reporting thresholds")

    return DriftReport(
        earlier=earlier,
        later=later,
        deltas=deltas,
        earlier_intents=earlier_intents,
        later_intents=later_intents,
        intent_shift=intent_shift,
        psi=stability,
        signals=signals,
    )


@dataclass
class Repricing:
    """The same traffic under a different price row.

    Change the table and every figure moves with it.  That property is the
    difference between a cost model and a spreadsheet.
    """

    target_model: str
    requests: int
    tokens: dict[str, int]
    actual_usd: float
    repriced_usd: float
    ratio: float
    components_usd: dict[str, float]
    monthly_projection_usd: float
    requests_per_day: int
    pricing_as_of: str = PRICING_AS_OF
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_model": self.target_model,
            "requests": self.requests,
            "tokens": self.tokens,
            "actual_usd": self.actual_usd,
            "repriced_usd": self.repriced_usd,
            "ratio": self.ratio,
            "components_usd": self.components_usd,
            "monthly_projection_usd": self.monthly_projection_usd,
            "requests_per_day": self.requests_per_day,
            "pricing_as_of": self.pricing_as_of,
            "note": self.note,
        }

    def render(self) -> str:
        lines = [
            f"target model       {self.target_model}",
            f"requests           {self.requests}",
            f"tokens             prompt {self.tokens['prompt']:,} "
            f"(cached {self.tokens['cached']:,}) · completion {self.tokens['completion']:,}",
            f"as recorded        ${self.actual_usd:.6f}",
            f"repriced           ${self.repriced_usd:.6f}  ({self.ratio:.2f}x)",
            "components         "
            + " · ".join(f"{key} ${value:.6f}" for key, value in sorted(self.components_usd.items())),
            f"monthly projection ${self.monthly_projection_usd:,.2f} "
            f"at {self.requests_per_day:,} requests/day",
            f"prices as of       {self.pricing_as_of}",
        ]
        if self.note:
            lines.append(f"note               {self.note}")
        return "\n".join(lines)


def reprice(
    rows: Sequence[dict[str, Any]], target_model: str, requests_per_day: int = 1_000
) -> Repricing:
    """Recompute recorded traffic under ``target_model``'s prices."""
    if target_model not in PRICING:
        raise ValueError(
            f"unknown model {target_model!r}; the pricing table knows {sorted(PRICING)}"
        )
    tokens = {
        "prompt": sum(int(row.get("prompt_tokens") or 0) for row in rows),
        "completion": sum(int(row.get("completion_tokens") or 0) for row in rows),
        "cached": sum(int(row.get("cached_tokens") or 0) for row in rows),
    }
    actual = round(sum(float(row.get("cost_usd") or 0.0) for row in rows), 6)
    total, components = cost_usd(
        target_model, tokens["prompt"], tokens["completion"], tokens["cached"]
    )
    price = price_for(target_model)
    per_request = total / len(rows) if rows else 0.0
    return Repricing(
        target_model=target_model,
        requests=len(rows),
        tokens=tokens,
        actual_usd=actual,
        repriced_usd=round(total, 6),
        ratio=round(total / actual, 3) if actual else 0.0,
        components_usd=components,
        monthly_projection_usd=round(per_request * requests_per_day * 30, 2),
        requests_per_day=requests_per_day,
        note=price.label(target_model) if price.is_reference else "",
    )
