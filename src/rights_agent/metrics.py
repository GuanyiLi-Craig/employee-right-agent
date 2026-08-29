"""One row per request, appended to ``runs/metrics.jsonl``.

This file *is* the dashboard.  Three rules shape it:

* ``index_version`` on **every** row.  Without it you cannot answer "which
  index produced this answer" six months later.
* The retrieved *context* stays out.  It belongs in the trace, under its own
  retention rules, not duplicated into a third store.
* One row per request, appended, never mutated.

Percentiles, not means, everywhere latency is reported: a mean hides the tail,
and the tail is what users feel.  Quality is reported with a **p10** alongside
the mean for the same reason -- a mean groundedness of 0.8 is consistent with
one answer in ten being unsupported.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

from rights_agent.config import PRICING_AS_OF, Settings
from rights_agent.log import get_logger

log = get_logger("metrics")

#: Assumed daily request volume for the monthly cost projection.  A projection
#: is only as good as its volume assumption, so the assumption is reported
#: alongside the number rather than buried.
DEFAULT_REQUESTS_PER_DAY = 50_000


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile.

    Deliberately not interpolated: with the sample sizes a demo produces,
    interpolation invents precision that is not there.
    """
    if not values:
        return 0.0
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return float(ordered[min(rank, len(ordered)) - 1])


@dataclass(slots=True)
class RequestMetrics:
    """The row.  Field order follows §7.3 of the specification."""

    request_id: str
    session_id: str = ""
    user_id: str = ""
    tenant: str = "default"
    ts: str = ""

    question: str = ""
    rewritten_query: str = ""
    route: str = ""
    intent: str = ""

    ttft_ms: float = 0.0
    itl_ms_mean: float = 0.0
    itl_ms_p95: float = 0.0
    e2e_ms: float = 0.0
    stage_ms: dict[str, float] = field(default_factory=dict)

    index_version: str = ""
    embedding_model: str = ""
    retrieved_ids: list[str] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)

    citations: list[str] = field(default_factory=list)
    attempts: int = 0
    sufficiency: float = 0.0
    refused: bool = False

    #: The model that actually served this request. Empty on a refusal, because
    #: no model was called and saying otherwise would be a guess.
    model: str = ""
    #: The model that was configured. Equal to ``model`` unless a fallback
    #: happened, and recording both is what makes the fallback visible.
    requested_model: str = ""
    fallback: bool = False
    prompt_version: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    cost_usd: float = 0.0
    cost_breakdown: dict[str, float] = field(default_factory=dict)
    #: The whole bill, not just the tokens: generation input and output, the
    #: judge, trace storage, infrastructure.  See :mod:`rights_agent.costs`.
    cost_components: dict[str, float] = field(default_factory=dict)
    cost_total_usd: float = 0.0
    trace_bytes: int = 0
    pricing_as_of: str = PRICING_AS_OF

    scores: dict[str, float] = field(default_factory=dict)
    trace_span_id: str = ""
    trace_id: str = ""
    degraded: bool = False
    #: Set when a follow-up borrowed terms from an earlier question in the
    #: session.  On the row because "the retriever saw more than the user typed"
    #: is exactly the kind of thing you want visible six months later.
    history_used: bool = False
    #: Where this decision sits in the hash-chained audit log.
    audit_sequence: int = -1
    audit_record_hash: str = ""
    error: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    def stage_total_ms(self) -> float:
        """Sum of every timed stage.  ``e2e`` must be at least this."""
        return round(sum(float(value) for value in self.stage_ms.values()), 3)

    def non_generation_ms(self) -> float:
        """End-to-end latency minus the measured generation stage.

        Retrieval, scoring, orchestration, serialisation.  On a thin pipeline it
        is small; on a system with three sequential model calls it is frequently
        the largest single term, and almost nobody measures it.
        """
        generation = float(self.stage_ms.get("generate", 0.0))
        return round(max(0.0, self.e2e_ms - generation), 3)

    def orchestration_ms(self) -> float:
        """Time inside the request that no stage claimed."""
        return round(max(0.0, self.e2e_ms - self.stage_total_ms()), 3)

    def formula_gap_ms(self) -> float:
        """``e2e`` minus ``TTFT + ITL x (tokens - 1)``.

        Kept as a diagnostic rather than a headline: ITL is measured per stream
        *chunk* while the multiplier counts *tokens*, so the identity is an
        approximation on both sides.  A large positive gap means real
        non-generation work; a negative one means chunks carried several tokens
        each.  Either way the measured numbers are the ones on the row.
        """
        predicted = self.ttft_ms + self.itl_ms_mean * max(0, self.completion_tokens - 1)
        return round(self.e2e_ms - predicted, 3)


class MetricsSink:
    """Append-only JSONL writer.

    Locked because the demo server serves requests from a thread pool, and two
    interleaved partial lines are two lost rows.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def append(self, row: RequestMetrics) -> None:
        line = row.to_json()
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError as exc:
                # Losing a metrics row must not lose the answer the user asked
                # for; it is logged loudly instead.
                log.error("could not append metrics row to %s: %s", self.path, exc)

    def read(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Read rows, newest last.  Malformed lines are skipped and counted."""
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        skipped = 0
        with self._lock, self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    skipped += 1
        if skipped:
            log.warning("skipped %d malformed metrics rows in %s", skipped, self.path)
        return rows[-limit:] if limit else rows

    def clear(self) -> None:
        """Truncate the file (the dashboard's reset button)."""
        with self._lock:
            if self.path.exists():
                self.path.unlink()


def sink_for(settings: Settings) -> MetricsSink:
    return MetricsSink(settings.metrics_path)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _numbers(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            out.append(float(value))
    return out


def _score_values(rows: Iterable[dict[str, Any]], name: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        scores = row.get("scores") or {}
        value = scores.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append(float(value))
    return out


def latency_summary(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """p50 / p95 / p99 for TTFT, ITL and end-to-end.  No means, by design."""
    out: dict[str, dict[str, float]] = {}
    for label, key in (("ttft_ms", "ttft_ms"), ("itl_ms", "itl_ms_mean"), ("e2e_ms", "e2e_ms")):
        values = _numbers(rows, key)
        out[label] = {
            "p50": round(percentile(values, 0.50), 2),
            "p95": round(percentile(values, 0.95), 2),
            "p99": round(percentile(values, 0.99), 2),
            "n": len(values),
        }
    return out


def quality_summary(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean and p10 for each judged dimension."""
    out: dict[str, dict[str, float]] = {}
    for name in ("groundedness", "citation_coverage", "context_relevance", "answer_relevance"):
        values = _score_values(rows, name)
        out[name] = {
            "mean": round(mean(values), 4) if values else 0.0,
            "p10": round(percentile(values, 0.10), 4),
            "n": len(values),
        }
    return out


def cost_summary(
    rows: Sequence[dict[str, Any]], requests_per_day: int = DEFAULT_REQUESTS_PER_DAY
) -> dict[str, Any]:
    """Per request, per conversation, and a projection with its assumption."""
    costs = [float(row.get("cost_usd") or 0.0) for row in rows]
    per_request = mean(costs) if costs else 0.0

    by_session: dict[str, float] = {}
    for row in rows:
        session = str(row.get("session_id") or "unknown")
        by_session[session] = by_session.get(session, 0.0) + float(row.get("cost_usd") or 0.0)
    per_conversation = mean(list(by_session.values())) if by_session else 0.0

    components: dict[str, float] = {}
    for row in rows:
        for key, value in (row.get("cost_breakdown") or {}).items():
            components[key] = round(components.get(key, 0.0) + float(value), 8)

    # The five-component model: tokens are not the whole bill, and a panel that
    # stops at tokens sends teams optimising the wrong term.
    from rights_agent.costs import aggregate as aggregate_components

    modelled = aggregate_components(rows)
    modelled_total = round(sum(modelled.values()), 8)

    tokens = {
        "prompt": sum(int(row.get("prompt_tokens") or 0) for row in rows),
        "completion": sum(int(row.get("completion_tokens") or 0) for row in rows),
        "cached": sum(int(row.get("cached_tokens") or 0) for row in rows),
    }
    return {
        "requests": len(rows),
        "total_usd": round(sum(costs), 6),
        "per_request_usd": round(per_request, 6),
        "per_conversation_usd": round(per_conversation, 6),
        "conversations": len(by_session),
        "monthly_projection_usd": round(per_request * requests_per_day * 30, 2),
        "projection_assumes_requests_per_day": requests_per_day,
        "components_usd": components,
        "modelled_components_usd": modelled,
        "modelled_total_usd": modelled_total,
        "modelled_per_request_usd": round(modelled_total / len(rows), 8) if rows else 0.0,
        "modelled_monthly_projection_usd": round(
            (modelled_total / len(rows) if rows else 0.0) * requests_per_day * 30, 2
        ),
        "dominant_component": max(modelled.items(), key=lambda item: item[1])[0]
        if any(modelled.values())
        else "",
        "tokens": tokens,
        "pricing_as_of": PRICING_AS_OF,
    }


def refusal_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    refused = sum(1 for row in rows if row.get("refused"))
    degraded = sum(1 for row in rows if row.get("degraded"))
    errors = sum(1 for row in rows if row.get("error"))
    return {
        "requests": len(rows),
        "refused": refused,
        "refusal_rate": round(refused / len(rows), 4) if rows else 0.0,
        "degraded": degraded,
        "errors": errors,
        "index_versions": sorted({str(row.get("index_version") or "") for row in rows} - {""}),
    }


#: Requests the latency and quality panels look at.
#:
#: Windowed, because a cumulative percentile is not a monitoring panel: once a
#: few hundred healthy requests are in the denominator, an incident of twenty
#: cannot move the median, and the panel that should have caught it stays green.
#: Cost and traffic stay cumulative -- those are totals, and a windowed total is
#: just a smaller total.
DEFAULT_PANEL_WINDOW = 20


def summarise(
    rows: Sequence[dict[str, Any]],
    requests_per_day: int = DEFAULT_REQUESTS_PER_DAY,
    window: int | None = DEFAULT_PANEL_WINDOW,
) -> dict[str, Any]:
    """Everything the dashboard's panels need, from one pass over the rows."""
    recent = list(rows[-window:]) if window else list(rows)
    return {
        "latency": latency_summary(recent),
        "quality": quality_summary(recent),
        "cost": cost_summary(rows, requests_per_day),
        "traffic": refusal_summary(rows),
        "window": {
            "requests": len(recent),
            "of": len(rows),
            "size": window or 0,
            "note": (
                "Latency and quality are over the most recent requests; cost and traffic "
                "are cumulative. A cumulative percentile cannot move during an incident."
            ),
        },
    }
