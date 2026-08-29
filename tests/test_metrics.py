"""The metrics row, its sink and its aggregations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rights_agent.metrics import (
    MetricsSink,
    RequestMetrics,
    cost_summary,
    latency_summary,
    percentile,
    quality_summary,
    refusal_summary,
    summarise,
)


def row(**overrides: object) -> dict[str, object]:
    base = {
        "request_id": "r1",
        "session_id": "s1",
        "index_version": "parser-3+hashing-bow-512+abcd1234",
        "ttft_ms": 10.0,
        "itl_ms_mean": 2.0,
        "e2e_ms": 100.0,
        "prompt_tokens": 800,
        "completion_tokens": 120,
        "cached_tokens": 0,
        "cost_usd": 0.001,
        "cost_breakdown": {"input_usd": 0.0006, "output_usd": 0.0004},
        "scores": {"groundedness": 1.0, "citation_coverage": 1.0},
        "refused": False,
        "degraded": False,
        "intent": "leave",
        "error": "",
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# Percentiles
# --------------------------------------------------------------------------- #
def test_percentile_is_nearest_rank_not_interpolated() -> None:
    """With demo-sized samples, interpolation invents precision."""
    values = list(range(1, 11))
    assert percentile(values, 0.50) == 5
    assert percentile(values, 0.95) == 10
    assert percentile(values, 1.0) == 10
    assert percentile([7], 0.5) == 7
    assert percentile([], 0.5) == 0.0


def test_percentile_rejects_a_fraction_outside_the_range() -> None:
    with pytest.raises(ValueError):
        percentile([1, 2], 0.0)
    with pytest.raises(ValueError):
        percentile([1, 2], 1.5)


# --------------------------------------------------------------------------- #
# Latency accounting
# --------------------------------------------------------------------------- #
def test_stage_total_and_orchestration_split_the_request() -> None:
    metrics = RequestMetrics(
        request_id="r",
        e2e_ms=200.0,
        stage_ms={"retrieve": 20.0, "generate": 150.0, "score": 5.0},
        ttft_ms=5.0,
        itl_ms_mean=1.0,
        completion_tokens=100,
    )
    assert metrics.stage_total_ms() == 175.0
    assert metrics.orchestration_ms() == 25.0
    assert metrics.non_generation_ms() == 50.0


def test_orchestration_never_goes_negative() -> None:
    metrics = RequestMetrics(request_id="r", e2e_ms=10.0, stage_ms={"generate": 99.0})
    assert metrics.orchestration_ms() == 0.0


def test_formula_gap_may_be_negative_and_that_is_informative() -> None:
    """Chunks carry several tokens each, so the identity overshoots."""
    metrics = RequestMetrics(
        request_id="r",
        e2e_ms=100.0,
        ttft_ms=5.0,
        itl_ms_mean=2.0,
        completion_tokens=200,
        stage_ms={"generate": 90.0},
    )
    assert metrics.formula_gap_ms() < 0


# --------------------------------------------------------------------------- #
# The sink
# --------------------------------------------------------------------------- #
def test_sink_appends_one_json_line_per_request(tmp_path: Path) -> None:
    sink = MetricsSink(tmp_path / "metrics.jsonl")
    sink.append(RequestMetrics(request_id="a"))
    sink.append(RequestMetrics(request_id="b"))
    lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    assert [json.loads(line)["request_id"] for line in lines] == ["a", "b"]


def test_sink_reads_newest_last_and_honours_a_limit(tmp_path: Path) -> None:
    sink = MetricsSink(tmp_path / "metrics.jsonl")
    for index in range(5):
        sink.append(RequestMetrics(request_id=f"r{index}"))
    assert [r["request_id"] for r in sink.read(limit=2)] == ["r3", "r4"]


def test_sink_skips_malformed_lines_rather_than_failing(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    sink = MetricsSink(path)
    sink.append(RequestMetrics(request_id="good"))
    with path.open("a") as handle:
        handle.write("{not json\n")
    assert [r["request_id"] for r in sink.read()] == ["good"]


def test_sink_reports_no_rows_before_anything_is_written(tmp_path: Path) -> None:
    assert MetricsSink(tmp_path / "missing.jsonl").read() == []


def test_the_row_never_carries_the_retrieved_context(tmp_path: Path) -> None:
    """Context belongs in the trace, under its own retention rules."""
    payload = json.loads(RequestMetrics(request_id="r").to_json())
    assert "context" not in payload and "docs" not in payload
    assert "retrieved_ids" in payload, "ids are kept; the text is not"


# --------------------------------------------------------------------------- #
# Aggregations
# --------------------------------------------------------------------------- #
def test_latency_summary_reports_percentiles_and_no_mean() -> None:
    rows = [row(e2e_ms=float(value)) for value in (50, 100, 150, 900)]
    summary = latency_summary(rows)
    assert set(summary["e2e_ms"]) == {"p50", "p95", "p99", "n"}
    assert "mean" not in summary["e2e_ms"]
    assert summary["e2e_ms"]["p95"] == 900.0


def test_quality_summary_reports_the_p10_alongside_the_mean() -> None:
    """A mean of 0.9 is consistent with one answer in ten being unsupported."""
    rows = [row(scores={"groundedness": g}) for g in [1.0] * 9 + [0.0]]
    summary = quality_summary(rows)
    assert summary["groundedness"]["mean"] == 0.9
    assert summary["groundedness"]["p10"] == 0.0


def test_cost_summary_states_the_assumption_behind_the_projection() -> None:
    summary = cost_summary([row(), row(session_id="s2")], requests_per_day=500)
    assert summary["projection_assumes_requests_per_day"] == 500
    assert summary["monthly_projection_usd"] == pytest.approx(0.001 * 500 * 30, rel=1e-6)
    assert summary["conversations"] == 2
    assert summary["components_usd"]["input_usd"] == pytest.approx(0.0012)


def test_cost_summary_averages_per_conversation_not_per_row() -> None:
    rows = [row(session_id="s1"), row(session_id="s1"), row(session_id="s2")]
    summary = cost_summary(rows)
    assert summary["per_conversation_usd"] == pytest.approx(0.0015)


def test_refusal_summary_lists_every_index_version_seen() -> None:
    rows = [row(), row(index_version="parser-3+onnx+abcd1234", refused=True)]
    summary = refusal_summary(rows)
    assert summary["refused"] == 1 and summary["refusal_rate"] == 0.5
    assert len(summary["index_versions"]) == 2


def test_summarise_covers_every_panel() -> None:
    assert set(summarise([row()])) == {"latency", "quality", "cost", "traffic", "window"}


def test_latency_and_quality_are_windowed_but_totals_are_not() -> None:
    """A cumulative percentile cannot move during an incident, which is exactly
    when the panel needs to."""
    healthy = [row(ttft_ms=2.0) for _ in range(40)]
    incident = [row(ttft_ms=90.0) for _ in range(10)]
    summary = summarise(healthy + incident, window=10)
    assert summary["latency"]["ttft_ms"]["p50"] == 90.0, "the window did not follow the incident"
    assert summary["window"] == {
        "requests": 10,
        "of": 50,
        "size": 10,
        "note": summary["window"]["note"],
    }
    # Totals stay cumulative: a windowed total is just a smaller total.
    assert summary["cost"]["requests"] == 50
    assert summary["traffic"]["requests"] == 50


def test_an_unwindowed_summary_uses_every_row() -> None:
    summary = summarise([row() for _ in range(5)], window=None)
    assert summary["window"]["requests"] == 5 and summary["window"]["size"] == 0


def test_aggregations_survive_an_empty_input() -> None:
    summary = summarise([])
    assert summary["latency"]["e2e_ms"]["p95"] == 0.0
    assert summary["cost"]["requests"] == 0
    assert summary["traffic"]["refusal_rate"] == 0.0


# --------------------------------------------------------------------------- #
# Which model actually answered
# --------------------------------------------------------------------------- #
def test_the_row_distinguishes_the_served_model_from_the_requested_one() -> None:
    """A row naming only the configured model cannot answer "which model
    produced this" after a silent failover."""
    metrics = RequestMetrics(
        request_id="r", model="stub-local", requested_model="deepseek-v4-flash", fallback=True
    )
    payload = json.loads(metrics.to_json())
    assert payload["model"] == "stub-local"
    assert payload["requested_model"] == "deepseek-v4-flash"
    assert payload["fallback"] is True


def test_a_row_with_no_fallback_has_both_fields_equal() -> None:
    metrics = RequestMetrics(
        request_id="r", model="deepseek-v4-flash", requested_model="deepseek-v4-flash"
    )
    assert not metrics.fallback
