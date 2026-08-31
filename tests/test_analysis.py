"""Drift detection and repricing."""

from __future__ import annotations

import pytest

from rights_agent.analysis import drift_report, reprice


def row(**overrides: object) -> dict[str, object]:
    base = {
        "ttft_ms": 2.0,
        "itl_ms_mean": 2.0,
        "e2e_ms": 150.0,
        "prompt_tokens": 800,
        "completion_tokens": 120,
        "cached_tokens": 0,
        "cost_usd": 0.001,
        "scores": {"groundedness": 1.0, "citation_coverage": 1.0},
        "refused": False,
        "degraded": False,
        "intent": "leave",
    }
    base.update(overrides)
    return base


def test_too_few_rows_says_so_rather_than_inventing_a_trend() -> None:
    report = drift_report([row(), row()])
    assert "generate some traffic" in report.signals[0]
    assert report.deltas == {}


def test_the_degraded_signature_is_reported_as_a_shared_cause() -> None:
    """Latency up and citation coverage down in the same window."""
    healthy = [row() for _ in range(6)]
    degraded = [
        row(ttft_ms=20.0, e2e_ms=1_400.0, scores={"groundedness": 1.0, "citation_coverage": 0.0},
            degraded=True)
        for _ in range(6)
    ]
    report = drift_report(healthy + degraded)
    joined = " ".join(report.signals)
    assert "TTFT p95 rose" in joined
    assert "citation coverage fell" in joined
    assert "weaker fallback" in joined


def test_the_report_says_what_it_cannot_prove() -> None:
    report = drift_report([row() for _ in range(8)])
    assert "do not identify which model answered" in report.caveat


def test_an_intent_shift_is_flagged_as_a_confound() -> None:
    """A quality change can be a change in the questions."""
    earlier = [row(intent="leave") for _ in range(6)]
    later = [row(intent="enforcement") for _ in range(6)]
    report = drift_report(earlier + later)
    assert report.intent_shift["enforcement"] == pytest.approx(1.0)
    assert any("intent mix moved" in signal for signal in report.signals)


def test_a_quiet_window_reports_no_movement() -> None:
    report = drift_report([row() for _ in range(10)])
    assert report.signals == ["no movement above the reporting thresholds"]


def test_repricing_scales_with_the_table_not_the_recorded_cost() -> None:
    rows = [row(prompt_tokens=1_000_000, completion_tokens=0, cost_usd=0.8)]
    haiku = reprice(rows, "claude-haiku-4-5")
    opus = reprice(rows, "claude-opus-5")
    assert opus.repriced_usd > haiku.repriced_usd * 10
    assert haiku.actual_usd == 0.8


def test_repricing_states_its_projection_assumption() -> None:
    result = reprice([row()], "claude-sonnet-5", requests_per_day=250)
    assert result.requests_per_day == 250
    assert result.monthly_projection_usd == pytest.approx(
        result.repriced_usd / 1 * 250 * 30, rel=1e-6
    )


def test_repricing_labels_a_reference_price() -> None:
    """The offline stub costs nothing; saying otherwise without a label is a lie."""
    result = reprice([row()], "stub-local")
    assert "reference" in result.note


def test_repricing_an_unknown_model_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        reprice([row()], "gpt-9")


def test_cached_tokens_lower_the_repriced_total() -> None:
    fresh = reprice([row(prompt_tokens=100_000, cached_tokens=0)], "claude-sonnet-5")
    cached = reprice([row(prompt_tokens=100_000, cached_tokens=100_000)], "claude-sonnet-5")
    assert cached.repriced_usd < fresh.repriced_usd
