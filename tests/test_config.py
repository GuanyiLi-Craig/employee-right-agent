"""Settings parsing, validation and the pricing table."""

from __future__ import annotations

import pytest

from rights_agent.config import (
    PRICING,
    PRICING_AS_OF,
    ConfigError,
    Settings,
    cost_usd,
    price_for,
    reload_settings,
)


def test_defaults_work_with_no_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "RIGHTS_MODEL",
        "RIGHTS_TOP_K",
        "RIGHTS_SUFFICIENCY",
        "RIGHTS_MAX_ATTEMPTS",
        "RIGHTS_EMBEDDER",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = reload_settings()
    assert settings.model == "stub-local"
    assert settings.top_k == 6
    assert settings.sufficiency_threshold == 0.45
    assert settings.max_attempts == 2
    assert settings.embedder == "auto"


@pytest.mark.parametrize(
    ("name", "value", "fragment"),
    [
        ("RIGHTS_TOP_K", "0", "outside supported range"),
        ("RIGHTS_TOP_K", "banana", "not an integer"),
        ("RIGHTS_SUFFICIENCY", "1.5", "outside supported range"),
        ("RIGHTS_SUFFICIENCY", "high", "not a number"),
        ("RIGHTS_EMBEDDER", "word2vec", "must be one of"),
        ("RIGHTS_LOG_LEVEL", "chatty", "not a logging level"),
        ("RIGHTS_DEGRADED", "sometimes", "not a boolean"),
    ],
)
def test_bad_values_fail_loudly(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, fragment: str
) -> None:
    """A misconfigured gate that silently never fires is worse than a crash."""
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError, match=fragment):
        reload_settings()


def test_context_budget_must_exceed_minimum_block(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIGHTS_CONTEXT_BUDGET", "600")
    monkeypatch.setenv("RIGHTS_MIN_BLOCK_CHARS", "800")
    with pytest.raises(ConfigError, match="must be smaller than"):
        reload_settings()


def test_derived_paths_hang_off_runs_dir(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("RIGHTS_RUNS_DIR", str(tmp_path / "artefacts"))
    settings = reload_settings()
    assert settings.chroma_dir == settings.runs_dir / "chroma"
    assert settings.manifest_path == settings.runs_dir / "index_manifest.json"
    assert settings.metrics_path == settings.runs_dir / "metrics.jsonl"


def test_with_overrides_revalidates() -> None:
    settings = Settings.from_env()
    assert settings.with_overrides(top_k=3).top_k == 3
    with pytest.raises(ConfigError):
        settings.with_overrides(context_budget_chars=100, min_block_chars=400)


# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #
def test_cost_is_derived_entirely_from_the_table() -> None:
    total, breakdown = cost_usd("claude-sonnet-5", prompt_tokens=1_000_000, completion_tokens=0)
    assert breakdown["input_usd"] == pytest.approx(PRICING["claude-sonnet-5"].input_per_mtok)
    assert total == pytest.approx(breakdown["input_usd"])


def test_cached_input_is_billed_at_the_cached_rate() -> None:
    """Cached input is roughly a tenth: that ratio is what makes layout a lever."""
    fresh, _ = cost_usd("claude-sonnet-5", prompt_tokens=100_000, completion_tokens=0)
    cached, breakdown = cost_usd(
        "claude-sonnet-5", prompt_tokens=100_000, completion_tokens=0, cached_tokens=100_000
    )
    assert breakdown["input_usd"] == 0.0
    assert cached < fresh
    assert cached == pytest.approx(fresh / 10, rel=0.01)


def test_output_costs_several_times_input() -> None:
    for name, price in PRICING.items():
        assert price.output_per_mtok > price.input_per_mtok, name
        assert price.cached_input_per_mtok < price.input_per_mtok, name


def test_unknown_model_falls_back_rather_than_costing_zero() -> None:
    total, _ = cost_usd("some-new-model", 1_000, 1_000)
    assert total > 0


def test_cached_tokens_cannot_exceed_prompt_tokens() -> None:
    total, breakdown = cost_usd("gpt-4.1", prompt_tokens=100, completion_tokens=0, cached_tokens=500)
    assert breakdown["input_usd"] == 0.0
    assert total == pytest.approx(breakdown["cached_input_usd"])


def test_negative_tokens_are_rejected() -> None:
    with pytest.raises(ValueError):
        cost_usd("gpt-4.1", -1, 0)


def test_reference_prices_are_labelled_as_such() -> None:
    """The offline stub costs nothing; its cost figure must say so."""
    price = price_for("stub-local")
    assert price.is_reference
    label = price.label("stub-local")
    assert "reference" in label and price.reference_of in label
    assert PRICING_AS_OF


# --------------------------------------------------------------------------- #
# Audit and cost settings
# --------------------------------------------------------------------------- #
def test_retention_below_the_statutory_floor_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Six months is a floor, and the error says where it comes from."""
    monkeypatch.setenv("RIGHTS_RETENTION_DAYS", "30")
    with pytest.raises(ConfigError) as excinfo:
        reload_settings()
    message = str(excinfo.value)
    assert "Articles 19 and 26(6)" in message
    assert "at least six months" in message


def test_a_longer_retention_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIGHTS_RETENTION_DAYS", "2555")
    assert reload_settings().retention_days == 2555


def test_retention_is_not_checked_when_auditing_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """No audit log, no retention duty over one."""
    monkeypatch.setenv("RIGHTS_AUDIT", "false")
    monkeypatch.setenv("RIGHTS_RETENTION_DAYS", "1")
    assert reload_settings().retention_days == 1


def test_the_projection_volume_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIGHTS_PROJECTION_RPD", "1000")
    assert reload_settings().projection_requests_per_day == 1_000


def test_the_judge_sample_rate_is_a_fraction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RIGHTS_JUDGE_SAMPLE_RATE", "1.4")
    with pytest.raises(ConfigError, match="outside supported range"):
        reload_settings()


def test_audit_paths_hang_off_runs_dir(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("RIGHTS_RUNS_DIR", str(tmp_path))
    settings = reload_settings()
    assert settings.audit_path == settings.runs_dir / "audit.jsonl"
    assert settings.audit_checkpoint_path == settings.runs_dir / "audit_checkpoint.json"


def test_the_embedder_choices_match_the_embedding_module() -> None:
    """config.py duplicates the list to avoid importing chromadb. The
    duplication is only safe while something checks it."""
    from rights_agent.config import EMBEDDER_CHOICES
    from rights_agent.embedding import KNOWN_EMBEDDERS, PREFERENCE_ALIASES

    buildable = set(KNOWN_EMBEDDERS) | set(PREFERENCE_ALIASES) | {"auto", "hashing", "onnx"}
    assert EMBEDDER_CHOICES == buildable, (
        "RIGHTS_EMBEDDER accepts a value the embedding module cannot build, or "
        f"rejects one it can: {sorted(EMBEDDER_CHOICES ^ buildable)}"
    )


def test_every_gate_path_pins_the_same_model() -> None:
    """A merge gate that depends on a provider's availability, latency and price
    list is not a gate. Three call sites pinned this independently; the CLI's had
    drifted to no pin at all, so ``evaluate --gate`` graded a non-deterministic
    hosted model against thresholds measured on the stub."""
    import re
    from pathlib import Path

    from rights_agent.demo.jobs import GATE_MODEL as jobs_model
    from rights_agent.tools.evaluate import GATE_MODEL as evaluate_model

    assert jobs_model == evaluate_model == "stub-local"

    root = Path(__file__).resolve().parents[1]
    for relative in ("evals/conftest.py", "src/rights_agent/demo/jobs.py"):
        source = (root / relative).read_text()
        assert not re.search(r'^GATE_MODEL\s*=\s*"', source, re.M), (
            f"{relative} defines its own GATE_MODEL instead of importing the one"
        )
