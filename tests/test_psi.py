"""PSI, and the smoothing constant that makes one figure misleading."""

from __future__ import annotations

import pytest

from rights_agent.analysis import (
    PSI_EPSILON,
    PSI_SIGNIFICANT,
    PSI_STABLE,
    psi,
    psi_band,
    psi_report,
)


def test_identical_distributions_score_zero() -> None:
    counts = {"leave": 5, "pay": 5}
    report = psi_report(counts, counts)
    assert report.psi_known == 0.0
    assert report.psi_with_unseen == 0.0
    assert report.new_intents == []


def test_psi_grows_with_the_size_of_the_shift() -> None:
    small = psi_report({"a": 5, "b": 5}, {"a": 6, "b": 4}).psi_known
    large = psi_report({"a": 5, "b": 5}, {"a": 9, "b": 1}).psi_known
    assert 0 < small < large


def test_psi_is_symmetric_in_the_pair() -> None:
    left = {"a": 0.7, "b": 0.3}
    right = {"a": 0.3, "b": 0.7}
    assert psi(left, right) == pytest.approx(psi(right, left), rel=1e-9)


def test_bands_are_the_credit_risk_convention() -> None:
    assert psi_band(PSI_STABLE - 0.01) == "stable"
    assert psi_band((PSI_STABLE + PSI_SIGNIFICANT) / 2) == "moderate"
    assert psi_band(PSI_SIGNIFICANT + 0.01) == "significant"


# --------------------------------------------------------------------------- #
# The epsilon gotcha
# --------------------------------------------------------------------------- #
def test_a_new_category_dominates_the_smoothed_figure() -> None:
    """PSI divides by the baseline probability, so an unseen category sends the
    formula to infinity; smoothing keeps it finite and makes it depend on the
    constant."""
    earlier = {"leave": 10, "pay": 10}
    later = {"leave": 9, "pay": 9, "enforcement": 2}
    report = psi_report(earlier, later)
    assert report.psi_known < PSI_STABLE, "the known intents barely moved"
    assert report.psi_with_unseen > PSI_SIGNIFICANT
    assert report.new_intents == ["enforcement"]


def test_the_smoothed_figure_moves_with_the_epsilon() -> None:
    """Which is why it is never reported without the epsilon beside it."""
    earlier = {"leave": 10}
    later = {"leave": 9, "enforcement": 1}
    tight = psi_report(earlier, later, epsilon=1e-6).psi_with_unseen
    loose = psi_report(earlier, later, epsilon=1e-2).psi_with_unseen
    assert tight > loose * 2, "the figure is dominated by the smoothing constant"


def test_the_known_only_figure_is_epsilon_free() -> None:
    earlier = {"leave": 10, "pay": 6}
    later = {"leave": 6, "pay": 10, "enforcement": 4}
    tight = psi_report(earlier, later, epsilon=1e-8).psi_known
    loose = psi_report(earlier, later, epsilon=1e-2).psi_known
    assert tight == loose, "the shared-support figure must not depend on the epsilon"


def test_the_report_names_its_epsilon() -> None:
    report = psi_report({"a": 1}, {"a": 1, "b": 1})
    assert report.epsilon == PSI_EPSILON
    assert f"epsilon={PSI_EPSILON:g}" in report.render()


def test_a_zero_epsilon_is_refused() -> None:
    with pytest.raises(ValueError, match="divides by the baseline"):
        psi({"a": 1.0}, {"b": 1.0}, epsilon=0.0)


# --------------------------------------------------------------------------- #
# The list, which needs no threshold
# --------------------------------------------------------------------------- #
def test_new_and_vanished_intents_are_listed_explicitly() -> None:
    report = psi_report({"leave": 5, "hours": 5}, {"leave": 5, "enforcement": 5})
    assert report.new_intents == ["enforcement"]
    assert report.vanished_intents == ["hours"]
    assert report.shared_intents == ["leave"]


def test_the_render_leads_with_the_actionable_finding() -> None:
    rendered = psi_report({"leave": 5}, {"leave": 4, "unions": 3}).render()
    assert "NEW intents" in rendered
    assert "unions" in rendered
    assert "not a law of nature" in rendered
    assert "whoever owns the corpus" in rendered


def test_shared_support_is_renormalised_not_truncated() -> None:
    """Otherwise the known-only figure would report a shift that is only the
    missing mass of the new categories."""
    earlier = {"leave": 5, "pay": 5}
    later = {"leave": 5, "pay": 5, "new": 10}
    assert psi_report(earlier, later).psi_known == 0.0


def test_an_empty_window_does_not_raise() -> None:
    report = psi_report({}, {"leave": 3})
    assert report.psi_known == 0.0
    assert report.new_intents == ["leave"]
