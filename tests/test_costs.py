"""The five-component cost model."""

from __future__ import annotations

import pytest

from rights_agent.config import JUDGE_MODEL, PRICING
from rights_agent.costs import (
    COMPONENTS,
    aggregate,
    breakdown,
    judge_cost,
    render_repricing,
    reprice_components,
    trace_bytes_for,
    trace_storage_cost,
)


def row(**overrides: object) -> dict[str, object]:
    base = {
        "prompt_tokens": 855,
        "completion_tokens": 142,
        "cached_tokens": 0,
        "trace_bytes": 10_000,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
def test_every_component_is_reported() -> None:
    """A cost model that stops at tokens sends teams optimising the wrong term."""
    parts = breakdown(model="claude-haiku-4-5", prompt_tokens=1_000, completion_tokens=100)
    assert set(parts.components) == set(COMPONENTS)
    assert parts.total_usd == pytest.approx(sum(parts.components.values()))


def test_generation_input_dominates_this_workload() -> None:
    """Long retrieved context, short answer: the chunks you chose to send."""
    parts = breakdown(model="claude-sonnet-5", prompt_tokens=855, completion_tokens=142)
    assert parts.dominant == "generation_input"


def test_a_short_context_and_long_answer_inverts_it() -> None:
    """Which is why the claim is scoped to the workload, not stated as a law."""
    parts = breakdown(model="claude-sonnet-5", prompt_tokens=100, completion_tokens=2_000)
    assert parts.dominant == "generation_output"


def test_output_is_dearer_per_token_than_input() -> None:
    same = breakdown(model="claude-sonnet-5", prompt_tokens=1_000, completion_tokens=1_000)
    assert same.components["generation_output"] > same.components["generation_input"]


def test_cached_input_is_folded_into_generation_input() -> None:
    fresh = breakdown(model="claude-sonnet-5", prompt_tokens=10_000, completion_tokens=0)
    cached = breakdown(
        model="claude-sonnet-5", prompt_tokens=10_000, completion_tokens=0, cached_tokens=10_000
    )
    assert cached.components["generation_input"] < fresh.components["generation_input"]


# --------------------------------------------------------------------------- #
# The judge is on the list
# --------------------------------------------------------------------------- #
def test_the_judge_is_a_cost_line() -> None:
    """It is an LLM call. Scoring every request can materially increase the bill."""
    parts = breakdown(model="stub-local", prompt_tokens=855, completion_tokens=142)
    assert parts.components["judge"] > 0


def test_judge_cost_scales_with_the_sample_rate() -> None:
    """Money, not statistical elegance, is why production evaluation is sampled."""
    full = judge_cost(1_000, 100, sample_rate=1.0)
    sampled = judge_cost(1_000, 100, sample_rate=0.1)
    assert sampled == pytest.approx(full / 10, rel=1e-6)
    assert judge_cost(1_000, 100, sample_rate=0.0) == 0.0


def test_an_invalid_sample_rate_is_rejected() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        judge_cost(10, 10, sample_rate=1.5)


def test_a_modelled_judge_is_labelled_as_modelled() -> None:
    """A cost panel that cannot tell incurred from modelled gets disbelieved."""
    modelled = breakdown(model="stub-local", prompt_tokens=100, completion_tokens=10)
    assert not modelled.judge_incurred
    assert any("modelled, not incurred" in note for note in modelled.notes)
    incurred = breakdown(
        model="claude-sonnet-5", prompt_tokens=100, completion_tokens=10, judge_incurred=True
    )
    assert not any("modelled" in note for note in incurred.notes)


def test_a_reference_priced_model_says_so() -> None:
    parts = breakdown(model="stub-local", prompt_tokens=100, completion_tokens=10)
    assert any("costs nothing to run" in note for note in parts.notes)
    assert JUDGE_MODEL in PRICING


# --------------------------------------------------------------------------- #
# Trace storage
# --------------------------------------------------------------------------- #
def test_trace_bytes_grow_with_the_retrieved_context() -> None:
    """Estimating this from a constant would make the component insensitive to
    top-k, which is the lever that actually moves it."""
    small = trace_bytes_for(1_000, 500)
    large = trace_bytes_for(20_000, 500)
    assert large > small
    assert large - small == 19_000


def test_trace_storage_is_cheap_per_request_and_priced_over_retention() -> None:
    single = trace_storage_cost(10_000)
    assert 0 < single < 0.0001
    assert trace_storage_cost(10_000, months=12) == pytest.approx(single * 2, rel=1e-9)


def test_trace_bytes_never_go_negative() -> None:
    assert trace_bytes_for(-5, -5, -5) > 0


# --------------------------------------------------------------------------- #
# Repricing
# --------------------------------------------------------------------------- #
def test_sonnet_costs_about_three_times_haiku_on_the_model_lines() -> None:
    rows = [row()] * 20
    haiku = reprice_components(rows, "claude-haiku-4-5", requests_per_day=50_000)
    sonnet = reprice_components(rows, "claude-sonnet-5", requests_per_day=50_000)
    for line in ("generation_input", "generation_output"):
        assert sonnet.components_usd[line] == pytest.approx(
            haiku.components_usd[line] * 3, rel=1e-6
        )
    assert 2.5 < sonnet.total_usd / haiku.total_usd < 3.0, (
        "model-independent components dilute the ratio, which is itself worth showing"
    )


def test_repricing_uses_recorded_tokens_and_reruns_nothing() -> None:
    rows = [row(), row(prompt_tokens=100, completion_tokens=10)]
    priced = reprice_components(rows, "claude-sonnet-5", requests_per_day=1_000)
    assert priced.tokens["prompt"] == 955
    assert priced.tokens["completion"] == 152
    assert priced.requests == 2


def test_the_projection_states_its_volume_assumption() -> None:
    priced = reprice_components([row()], "claude-haiku-4-5", requests_per_day=50_000)
    assert priced.requests_per_day == 50_000
    assert priced.monthly_projection_usd == pytest.approx(
        priced.per_request_usd * 50_000 * 30, rel=1e-6
    )


def test_the_side_by_side_render_names_the_dominant_component() -> None:
    rows = [row()] * 5
    rendered = render_repricing(
        reprice_components(rows, "claude-haiku-4-5", requests_per_day=50_000),
        reprice_components(rows, "claude-sonnet-5", requests_per_day=50_000),
    )
    assert "generation_input" in rendered
    assert "Nothing was re-run" in rendered
    assert "top-k and chunk size" in rendered
    assert "50,000 requests/day" in rendered


def test_aggregate_sums_recorded_component_breakdowns() -> None:
    rows = [
        {"cost_components": {"generation_input": 0.001, "judge": 0.0001}},
        {"cost_components": {"generation_input": 0.002, "unknown_line": 9.0}},
    ]
    totals = aggregate(rows)
    assert totals["generation_input"] == pytest.approx(0.003)
    assert totals["judge"] == pytest.approx(0.0001)
    assert "unknown_line" not in totals


def test_aggregate_of_nothing_is_zero_not_missing() -> None:
    assert aggregate([]) == dict.fromkeys(COMPONENTS, 0.0)


# --------------------------------------------------------------------------- #
# Which component leads is a property of the price list, not a constant
# --------------------------------------------------------------------------- #
def test_a_steep_cache_discount_moves_the_dominant_component() -> None:
    """On DeepSeek most of the prompt is billed at ~1/31, so input stops leading
    even though the prompt is six times the answer."""
    parts = breakdown(
        model="deepseek-v4-flash",
        prompt_tokens=900,
        completion_tokens=140,
        cached_tokens=768,
    )
    assert parts.dominant == "generation_output"


def test_the_same_traffic_uncached_is_led_by_input() -> None:
    parts = breakdown(
        model="deepseek-v4-flash", prompt_tokens=900, completion_tokens=140, cached_tokens=0
    )
    assert parts.dominant == "generation_input"


def test_the_report_describes_the_component_it_names() -> None:
    """A report that names one component while describing another is worse than
    no report."""
    from rights_agent.costs import COMPONENT_LEVERS

    rows = [row(prompt_tokens=900, completion_tokens=140, cached_tokens=768)] * 10
    rendered = render_repricing(
        reprice_components(rows, "deepseek-v4-flash-offpeak", requests_per_day=50_000),
        reprice_components(rows, "deepseek-v4-flash", requests_per_day=50_000),
    )
    assert "dominant component is generation_output" in rendered
    assert COMPONENT_LEVERS["generation_output"] in rendered
    assert COMPONENT_LEVERS["generation_input"] not in rendered


def test_every_component_has_a_lever_recorded() -> None:
    from rights_agent.costs import COMPONENT_LEVERS

    assert set(COMPONENT_LEVERS) == set(COMPONENTS)


def test_the_cached_share_is_reported_because_it_explains_the_shift() -> None:
    rows = [row(prompt_tokens=900, completion_tokens=140, cached_tokens=768)] * 10
    rendered = render_repricing(
        reprice_components(rows, "deepseek-v4-flash-offpeak", requests_per_day=1_000),
        reprice_components(rows, "deepseek-v4-flash", requests_per_day=1_000),
    )
    assert "7,680 cached" in rendered


# --------------------------------------------------------------------------- #
# Judge pricing follows what you actually serve
# --------------------------------------------------------------------------- #
def test_the_judge_is_priced_as_the_serving_model_by_default() -> None:
    """A DeepSeek run should not be costed with a competitor's judge."""
    from rights_agent.costs import judge_model_for

    assert judge_model_for("deepseek-v4-flash") == "deepseek-v4-flash"
    assert judge_model_for("claude-sonnet-5") == "claude-sonnet-5"


def test_the_offline_stub_keeps_a_real_judge_price() -> None:
    """It is free, so pricing the judge as the stub would make the line vanish
    along with the point it illustrates."""
    from rights_agent.costs import JUDGE_MODEL, judge_model_for

    assert judge_model_for("stub-local") == JUDGE_MODEL


def test_an_explicit_judge_model_wins() -> None:
    """Judging with a cheaper model than you serve with is the common pattern."""
    from rights_agent.costs import judge_model_for

    assert judge_model_for("deepseek-v4-pro", "deepseek-v4-flash") == "deepseek-v4-flash"


def test_the_judge_note_names_the_model_it_priced() -> None:
    parts = breakdown(model="stub-local", prompt_tokens=100, completion_tokens=10,
                      judge_model="deepseek-v4-flash")
    assert any("deepseek-v4-flash judge" in note for note in parts.notes)
