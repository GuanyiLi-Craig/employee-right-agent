"""Aggregate quality thresholds.

Model output is a distribution.  Asserting that *every* answer clears a bar
produces a flaky suite, and a flaky suite gets deleted -- so this file gates
aggregates, and the one thing it asserts per-row is that nothing is missing.

Two aggregates, always together:

* the **mean**, which says whether the system is broadly working;
* the **p10**, which says whether one answer in ten is unsupported.  A mean of
  0.9 is perfectly consistent with a tenth of your answers being groundless,
  and the mean will stay green the whole time.

And before any of that: **the judge's kappa.**  Gate the instrument before you
gate with it.  If the kappa assertion fails, every other threshold in this file
is measuring nothing, so it runs first and its failure message says so.

Thresholds live in ``evals/datasets/<embedder>/baseline.json``, set *below* observed values on a
green build.  Ratchet them upward as the system improves; **never** downward to
fix a red build.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean
from typing import Any

import pytest

from conftest import GoldenResult
from rights_agent.judges import HeuristicJudge, JudgeScores, calibrate
from rights_agent.metrics import percentile


#: The dimension pair that decides whether an answer is shippable: supported
#: *and* attributable.  Calibrating against groundedness alone marks an
#: accurate-but-uncited answer as fine, which is how uncitable answers ship.
def shippable(scores: JudgeScores) -> float:
    return min(scores.groundedness, scores.citation_coverage)


@pytest.fixture(scope="module")
def observed(answerable_results: Sequence[GoldenResult]) -> dict[str, list[float]]:
    """Per-dimension score samples over the answerable golden rows."""
    samples: dict[str, list[float]] = {
        "groundedness": [],
        "citation_coverage": [],
        "context_relevance": [],
        "answer_relevance": [],
    }
    for result in answerable_results:
        for name in samples:
            samples[name].append(float(result.answer.scores[name]))
    assert samples["groundedness"], "no answerable golden rows produced scores"
    return samples


@pytest.fixture(scope="module")
def thresholds(baseline: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in baseline["quality_thresholds"].items()}


# --------------------------------------------------------------------------- #
# Gate the instrument first
# --------------------------------------------------------------------------- #
def test_judge_kappa_clears_the_floor(
    calibration_rows: Sequence[dict[str, Any]], thresholds: dict[str, float]
) -> None:
    calibration = calibrate(HeuristicJudge(), calibration_rows, scorer=shippable, threshold=0.7)
    floor = thresholds["judge_kappa"]
    assert calibration.n >= 12, "the calibration set is too small to say anything"
    assert calibration.kappa >= floor, (
        f"judge kappa {calibration.kappa:.3f} is below {floor:.2f} "
        f"(agreement {calibration.agreement:.3f}, {calibration.confusion}). "
        "Every other threshold in this file is measuring nothing until this passes. "
        f"Disagreements: {[d['id'] for d in calibration.disagreements]}"
    )


def test_raw_agreement_alone_would_have_flattered_the_judge(
    calibration_rows: Sequence[dict[str, Any]]
) -> None:
    """Why kappa, and not accuracy.

    A judge that returns the majority label every time scores high raw
    agreement and has a kappa of zero.  This asserts the calibration set is
    still capable of showing that -- if it ever became one-sided, the kappa gate
    above would quietly stop meaning anything.
    """
    labels = [int(row["human_label"]) for row in calibration_rows]
    majority = max(labels.count(0), labels.count(1)) / len(labels)
    assert 0.35 <= majority <= 0.80, (
        f"the calibration set is {majority:.0%} one label; a constant judge would score "
        "that as agreement, so the set needs rebalancing"
    )


def test_hard_cases_lower_the_kappa(calibration_rows: Sequence[dict[str, Any]]) -> None:
    """The hard cases must actually be hard.

    Clean examples alone yield a near-perfect kappa.  If the full set scores the
    same as the clean subset, the hard cases have stopped doing their job and
    the reported kappa is optimistic.
    """
    judge = HeuristicJudge()
    clean = [row for row in calibration_rows if str(row["id"]).startswith("c")]
    hard = [row for row in calibration_rows if not str(row["id"]).startswith("c")]
    assert len(hard) >= 4, "the calibration set should keep at least four hard cases"
    clean_kappa = calibrate(judge, clean, scorer=shippable, threshold=0.7).kappa
    full_kappa = calibrate(judge, calibration_rows, scorer=shippable, threshold=0.7).kappa
    assert full_kappa < clean_kappa, (
        f"hard cases did not lower the kappa ({full_kappa:.3f} vs clean {clean_kappa:.3f}); "
        "the judge did not get better, the test got easier"
    )


# --------------------------------------------------------------------------- #
# Aggregate quality
# --------------------------------------------------------------------------- #
def test_mean_groundedness(observed: dict[str, list[float]], thresholds: dict[str, float]) -> None:
    value = mean(observed["groundedness"])
    assert value >= thresholds["groundedness_mean"], (
        f"mean groundedness {value:.3f} below {thresholds['groundedness_mean']:.2f}"
    )


def test_p10_groundedness(observed: dict[str, list[float]], thresholds: dict[str, float]) -> None:
    """The mean stays green while one answer in ten is unsupported."""
    value = percentile(observed["groundedness"], 0.10)
    assert value >= thresholds["groundedness_p10"], (
        f"p10 groundedness {value:.3f} below {thresholds['groundedness_p10']:.2f}: "
        "the tail is unsupported even if the mean looks fine"
    )


def test_mean_citation_coverage(
    observed: dict[str, list[float]], thresholds: dict[str, float]
) -> None:
    value = mean(observed["citation_coverage"])
    assert value >= thresholds["citation_coverage_mean"], (
        f"mean citation coverage {value:.3f} below {thresholds['citation_coverage_mean']:.2f}"
    )


def test_p10_citation_coverage(
    observed: dict[str, list[float]], thresholds: dict[str, float]
) -> None:
    """Its own threshold. Reading ``groundedness_p10`` here made the two tails
    move together by accident: tightening one silently tightened the other, and
    the failure message named a threshold the test was not checking."""
    value = percentile(observed["citation_coverage"], 0.10)
    assert value >= thresholds["citation_coverage_p10"], (
        f"p10 citation coverage {value:.3f} below {thresholds['citation_coverage_p10']:.2f}"
    )


def test_mean_context_relevance(
    observed: dict[str, list[float]], thresholds: dict[str, float]
) -> None:
    """Blames retrieval, not generation -- which is the point of separating them."""
    value = mean(observed["context_relevance"])
    assert value >= thresholds["context_relevance_mean"], (
        f"mean context relevance {value:.3f} below "
        f"{thresholds['context_relevance_mean']:.2f}: retrieval regressed"
    )


def test_mean_answer_relevance(
    observed: dict[str, list[float]], thresholds: dict[str, float]
) -> None:
    value = mean(observed["answer_relevance"])
    assert value >= thresholds["answer_relevance_mean"], (
        f"mean answer relevance {value:.3f} below {thresholds['answer_relevance_mean']:.2f}"
    )


def test_every_answerable_row_is_scored(
    answerable_results: Sequence[GoldenResult]
) -> None:
    """A missing score is not a passing score."""
    for result in answerable_results:
        scores = result.answer.scores
        missing = {
            "groundedness",
            "citation_coverage",
            "context_relevance",
            "answer_relevance",
        } - set(scores)
        assert not missing, f"{result.id} is missing scores: {sorted(missing)}"


def test_degraded_mode_is_visible_in_the_scores(eval_agent) -> None:
    """The degraded-fallback simulation must actually degrade something.

    Two correlated series -- latency up, citation coverage down -- tell you
    *when* something changed and that the symptoms share a cause.  They do not
    prove which model answered; only the trace does.  This asserts the signal
    exists, not that it identifies the culprit.
    """
    question = "What does the document say about bereavement leave?"
    healthy = eval_agent.ask(question, session_id="degraded-control", record=False)
    eval_agent.set_degraded(True)
    try:
        degraded = eval_agent.ask(question, session_id="degraded-test", record=False)
    finally:
        eval_agent.set_degraded(False)

    assert healthy.scores["citation_coverage"] > degraded.scores["citation_coverage"]
    assert not degraded.citations, "degraded mode should drop citations"
    assert degraded.metrics.ttft_ms > healthy.metrics.ttft_ms, "degraded mode should be slower"
