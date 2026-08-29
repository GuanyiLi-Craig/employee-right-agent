"""The whole bill, not just the tokens.

A cost model that stops at model tokens sends teams optimising the wrong term.
Five components, each derived from the one dated table in
:mod:`rights_agent.config`:

======================  ====================================================
``generation_input``    the retrieved context you chose to send — every call,
                        forever.  In *this* workload it is the largest model
                        component; a system with short contexts and long
                        answers inverts that.
``generation_output``   materially dearer per token than input on many current
                        models, which makes brevity a cost control rather than
                        a style preference.
``judge``               an LLM call like any other.  Scoring every request can
                        materially increase the bill, and *that* — money, not
                        statistical elegance — is the real reason production
                        evaluation is sampled.
``trace_storage``       cheap per request, genuinely large in aggregate; the
                        multiplier is the retention period.
``infrastructure``      compute, orchestration and the vector store.
======================  ====================================================

Be honest about one artefact of the offline path: with the stub model priced as
a reference, infrastructure and trace storage look dominant.  Pricing any real
model moves generation to the top. :func:`reprice_components` shows that, and
the contrast is worth naming out loud rather than hiding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from rights_agent.config import (
    INFRA_USD_PER_REQUEST,
    JUDGE_MODEL,
    JUDGE_OUTPUT_TOKENS,
    JUDGE_SAMPLE_RATE,
    PRICING_AS_OF,
    TRACE_OVERHEAD_BYTES,
    TRACE_RETENTION_MONTHS,
    TRACE_STORAGE_USD_PER_GB_MONTH,
    cost_usd,
    price_for,
)

COMPONENTS = (
    "generation_input",
    "generation_output",
    "judge",
    "trace_storage",
    "infrastructure",
)

#: What to *do* about each component when it turns out to be the dominant one.
#:
#: Keyed rather than written into one sentence: which component dominates is a
#: property of the workload and the price list, not a constant. On a model with a
#: steep prompt-cache discount most of the input cost disappears and output leads
#: instead -- and a report that names one component while describing another is
#: worse than no report.
COMPONENT_LEVERS: dict[str, str] = {
    "generation_input": (
        "the retrieved context you chose to send, on every call, forever. The lever is "
        "top-k and chunk size -- usually a bigger one than the model."
    ),
    "generation_output": (
        "the answer you generate. The lever is brevity: instruct for shorter answers and "
        "cap max tokens. Worth checking the prompt cache is working first -- output only "
        "leads once input is discounted."
    ),
    "judge": (
        "your own evaluation. The lever is the sample rate, and possibly a cheaper judge "
        "model. This is why production evaluation is sampled: money, not statistics."
    ),
    "trace_storage": (
        "retained spans. The lever is the retention period and what you attach to a span "
        "-- and retention has a floor if the trace is also your audit trail."
    ),
    "infrastructure": (
        "compute, orchestration and the vector store. It leads only when the model is "
        "cheap or free, which is worth naming rather than reading as a real result."
    ),
}

_BYTES_PER_GB = 1_000_000_000


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Per-request cost, component by component."""

    total_usd: float
    components: dict[str, float]
    model: str
    judge_model: str
    judge_sample_rate: float
    judge_incurred: bool
    trace_bytes: int
    pricing_as_of: str = PRICING_AS_OF
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def dominant(self) -> str:
        if not self.components:
            return ""
        return max(self.components.items(), key=lambda item: item[1])[0]


def trace_bytes_for(context_chars: int, answer_chars: int, question_chars: int = 0) -> int:
    """Span payload size for one request.

    Measured from what the spans actually carry -- the retrieved context, the
    answer, the question -- plus a fixed allowance for attributes, ids and the
    exporter's framing.  Estimating this from a constant alone would make the
    component insensitive to top-k, which is the lever that actually moves it.
    """
    return max(0, context_chars) + max(0, answer_chars) + max(0, question_chars) + TRACE_OVERHEAD_BYTES


def judge_model_for(serving_model: str, override: str = "") -> str:
    """Which model to price the judge line against.

    An explicit override wins. Otherwise the serving model, because a run on one
    provider should not be costed with a competitor's judge -- except for the
    offline stub, which is free and would make the judge line vanish along with
    the point it illustrates.

    Using the generator as its own judge has a known self-preference bias. That
    is a reason to calibrate it (see :mod:`rights_agent.judges`), not a reason to
    price it as something it is not.
    """
    if override:
        return override
    if serving_model and not price_for(serving_model).is_reference:
        return serving_model
    return JUDGE_MODEL


def judge_cost(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    sample_rate: float = JUDGE_SAMPLE_RATE,
    judge_model: str = JUDGE_MODEL,
) -> float:
    """Expected judge cost per request at ``sample_rate``.

    A model-graded judge reads the question, the context and the answer, so its
    input is close to the generation prompt; it emits only a few tokens of JSON.
    Reported as an expectation per request, because that is the number that
    changes when you change the sample rate.
    """
    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError(f"sample_rate must be in [0, 1]; got {sample_rate}")
    judged_input = prompt_tokens + completion_tokens
    total, _ = cost_usd(judge_model, judged_input, JUDGE_OUTPUT_TOKENS)
    return round(total * sample_rate, 10)


def trace_storage_cost(trace_bytes: int, months: float = TRACE_RETENTION_MONTHS) -> float:
    """Storage cost for one request's spans over the retention period."""
    return round(trace_bytes / _BYTES_PER_GB * TRACE_STORAGE_USD_PER_GB_MONTH * months, 10)


def breakdown(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    trace_bytes: int = TRACE_OVERHEAD_BYTES,
    judge_sample_rate: float = JUDGE_SAMPLE_RATE,
    judge_model: str = "",
    judge_incurred: bool = False,
    infrastructure_usd: float = INFRA_USD_PER_REQUEST,
) -> CostBreakdown:
    """Full per-request cost.

    ``judge_incurred`` distinguishes a judge that actually ran and billed from
    one that is *modelled*: the offline heuristic judge makes no model call, so
    its line is what a model-graded judge would add at this sample rate.  A cost
    panel that cannot tell those apart is a cost panel that will be disbelieved.
    """
    judge_model = judge_model_for(model, judge_model)
    _, model_parts = cost_usd(model, prompt_tokens, completion_tokens, cached_tokens)
    components = {
        "generation_input": round(
            model_parts["input_usd"] + model_parts["cached_input_usd"], 10
        ),
        "generation_output": round(model_parts["output_usd"], 10),
        "judge": judge_cost(
            prompt_tokens,
            completion_tokens,
            sample_rate=judge_sample_rate,
            judge_model=judge_model,
        ),
        "trace_storage": trace_storage_cost(trace_bytes),
        "infrastructure": round(infrastructure_usd, 10),
    }
    notes: list[str] = []
    price = price_for(model)
    if price.is_reference:
        notes.append(
            f"{model} costs nothing to run; it is priced as {price.reference_of} so the "
            "mechanics are visible offline. Infrastructure and trace storage therefore "
            "look larger than they would against a real model."
        )
    if not judge_incurred:
        notes.append(
            f"judge line is modelled, not incurred: the offline judge makes no model call. "
            f"It is what a {judge_model} judge would add at a {judge_sample_rate:.0%} "
            "sample rate."
        )
    return CostBreakdown(
        total_usd=round(sum(components.values()), 10),
        components=components,
        model=model,
        judge_model=judge_model,
        judge_sample_rate=judge_sample_rate,
        judge_incurred=judge_incurred,
        trace_bytes=trace_bytes,
        notes=notes,
    )


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Sum the component breakdowns already recorded on metrics rows."""
    totals: dict[str, float] = {name: 0.0 for name in COMPONENTS}
    for row in rows:
        for name, value in (row.get("cost_components") or {}).items():
            if name in totals:
                totals[name] = round(totals[name] + float(value), 10)
    return totals


@dataclass(frozen=True, slots=True)
class Repricing:
    """The same recorded traffic under a different model's prices.

    Nothing is re-run: the token counts are already on the rows.  Recorded
    tokens plus a price table *is* a cost model, with no new instrumentation --
    and it makes routing easy questions to the cheap model an arithmetic
    question rather than a taste one.
    """

    model: str
    requests: int
    tokens: dict[str, int]
    components_usd: dict[str, float]
    total_usd: float
    per_request_usd: float
    monthly_projection_usd: float
    requests_per_day: int
    dominant_component: str
    pricing_as_of: str = PRICING_AS_OF
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reprice_components(
    rows: Sequence[Mapping[str, Any]],
    model: str,
    *,
    requests_per_day: int,
    judge_sample_rate: float = JUDGE_SAMPLE_RATE,
    judge_model: str = "",
) -> Repricing:
    """Reprice recorded traffic under ``model``, component by component."""
    tokens = {
        "prompt": sum(int(row.get("prompt_tokens") or 0) for row in rows),
        "completion": sum(int(row.get("completion_tokens") or 0) for row in rows),
        "cached": sum(int(row.get("cached_tokens") or 0) for row in rows),
    }
    trace_total = sum(int(row.get("trace_bytes") or TRACE_OVERHEAD_BYTES) for row in rows)
    parts = breakdown(
        model=model,
        prompt_tokens=tokens["prompt"],
        completion_tokens=tokens["completion"],
        cached_tokens=tokens["cached"],
        trace_bytes=trace_total,
        judge_sample_rate=judge_sample_rate,
        judge_model=judge_model,
        infrastructure_usd=INFRA_USD_PER_REQUEST * max(1, len(rows)),
    )
    count = max(1, len(rows))
    per_request = parts.total_usd / count
    return Repricing(
        model=model,
        requests=len(rows),
        tokens=tokens,
        components_usd=parts.components,
        total_usd=round(parts.total_usd, 6),
        per_request_usd=round(per_request, 8),
        monthly_projection_usd=round(per_request * requests_per_day * 30, 2),
        requests_per_day=requests_per_day,
        dominant_component=parts.dominant,
        notes=parts.notes,
    )


def render_repricing(left: Repricing, right: Repricing) -> str:
    """Two models side by side, with the ratio and the dominant component."""
    lines = [
        f"{'component':<22}{left.model:>18}{right.model:>18}",
    ]
    for name in COMPONENTS:
        lines.append(
            f"{name:<22}{left.components_usd.get(name, 0.0):>18.6f}"
            f"{right.components_usd.get(name, 0.0):>18.6f}"
        )
    lines.append("-" * 58)
    lines.append(f"{'total':<22}{left.total_usd:>18.6f}{right.total_usd:>18.6f}")
    lines.append(f"{'per request':<22}{left.per_request_usd:>18.6f}{right.per_request_usd:>18.6f}")
    lines.append(
        f"{'monthly projection':<22}{left.monthly_projection_usd:>18,.2f}"
        f"{right.monthly_projection_usd:>18,.2f}"
    )
    ratio = right.total_usd / left.total_usd if left.total_usd else 0.0
    cached = left.tokens.get("cached", 0)
    lines += [
        "",
        f"{right.model} is {ratio:.2f}x {left.model} on identical recorded tokens — "
        f"{left.requests} requests, {left.tokens['prompt']:,} prompt "
        f"({cached:,} cached) and {left.tokens['completion']:,} completion tokens. "
        "Nothing was re-run.",
        f"projection assumes {left.requests_per_day:,} requests/day · prices as of {PRICING_AS_OF}",
        "",
        f"In this workload the dominant component is {right.dominant_component}: "
        + COMPONENT_LEVERS.get(right.dominant_component, "no lever recorded for this component."),
    ]
    if right.dominant_component != left.dominant_component:
        lines.append(
            f"Note that it differs between the two: {left.model} is dominated by "
            f"{left.dominant_component}. Which component leads is a property of the price "
            "list as much as the workload."
        )
    for note in right.notes:
        lines.append(f"note: {note}")
    return "\n".join(lines)
