"""Generation: the offline stub, latency measurement, token counting."""

from __future__ import annotations

import time

import pytest

from rights_agent.config import Settings
from rights_agent.llm import (
    DEGRADED_TTFT_PENALTY_S,
    STUB_MODEL,
    SYSTEM_PROMPT,
    LLMResult,
    StreamTimer,
    StubClient,
    build_prompt,
    count_tokens,
    extract_citations,
    generate,
    make_client,
    parse_context,
)

CONTEXT = (
    "[s.19] Act > Part 1 > s.19 Right to bereavement leave\n"
    "(1) A bereaved person is entitled to leave. (2) The leave must be taken within 56 days.\n\n"
    "[s.20] Act > Part 1 > s.20 Bereavement leave: length\n"
    "(1) The leave period is two weeks.\n\n"
    "[s.99] Act > Part 4 > s.99 Access agreements\n"
    "(1) An access agreement must be in writing."
)
QUESTION = "What does the document say about bereavement leave?"


@pytest.fixture
def settings() -> Settings:
    return Settings.from_env().with_overrides(model=STUB_MODEL, tracing_enabled=False)


# --------------------------------------------------------------------------- #
# Context parsing and prompts
# --------------------------------------------------------------------------- #
def test_parse_context_recovers_the_blocks() -> None:
    """The stub and the judges score exactly what the model was shown."""
    blocks = parse_context(CONTEXT)
    assert [block.citation for block in blocks] == ["s.19", "s.20", "s.99"]
    assert blocks[0].breadcrumb.startswith("Act > Part 1")
    assert blocks[0].text.startswith("(1) A bereaved person")


def test_parse_context_ignores_blocks_without_a_citation_header() -> None:
    assert parse_context("just some prose\n\nmore prose") == []


def test_prompt_says_what_to_do_with_an_empty_context() -> None:
    prompt = build_prompt(QUESTION, "")
    assert "(no provisions retrieved)" in prompt
    assert QUESTION in prompt


def test_extract_citations_deduplicates_and_preserves_order() -> None:
    assert extract_citations("[s.2] then [s.1] then [s.2] again") == ["s.2", "s.1"]


# --------------------------------------------------------------------------- #
# The stub
# --------------------------------------------------------------------------- #
def test_stub_is_extractive_and_cites_what_it_quotes() -> None:
    """Extractive, so a groundedness judge should score it high -- and if it does
    not, the judge is the thing that is wrong."""
    result = generate(QUESTION, CONTEXT, Settings.from_env().with_overrides(
        model=STUB_MODEL, tracing_enabled=False))
    assert result.citations, "the stub must cite"
    for citation in result.citations:
        assert f"[{citation}]" in CONTEXT, f"{citation} is not a context citation"
    blocks = {block.citation: block.text for block in parse_context(CONTEXT)}
    for citation in result.citations:
        first_sentence = blocks[citation].split(". ")[0].lstrip()
        assert first_sentence[:40] in result.text, "the quote is not verbatim"


def test_stub_is_deterministic(settings: Settings) -> None:
    """The offline path must produce the same output for the same input."""
    first = generate(QUESTION, CONTEXT, settings).text
    second = generate(QUESTION, CONTEXT, settings).text
    assert first == second


def test_stub_prefers_evidence_that_matches_the_question(settings: Settings) -> None:
    result = generate(QUESTION, CONTEXT, settings)
    assert "s.19" in result.citations
    assert result.citations[0] in {"s.19", "s.20"}, result.citations


def test_stub_says_so_when_the_context_is_empty(settings: Settings) -> None:
    result = generate(QUESTION, "", settings)
    assert "does not contain" in result.text
    assert not result.citations


def test_degraded_mode_drops_citations_and_slows_down(settings: Settings) -> None:
    """The signature of a primary model failing over to a weaker fallback."""
    healthy = generate(QUESTION, CONTEXT, settings, degraded=False)
    degraded = generate(QUESTION, CONTEXT, settings, degraded=True)
    assert healthy.citations and not degraded.citations
    assert degraded.ttft_ms > healthy.ttft_ms
    assert degraded.itl_ms_mean > healthy.itl_ms_mean
    assert degraded.degraded and not healthy.degraded


def test_the_degraded_first_token_penalty_survives_end_to_end_noise(
    settings: Settings,
) -> None:
    """Multiplying only the inter-token delay bought about 10ms of extra TTFT --
    real, and smaller than the retrieval jitter around it, so the whole-request
    ``ttft_ms`` a panel plots went *down* on a degraded run as often as up.  A
    failover costs a first token before it costs anything else."""
    healthy = generate(QUESTION, CONTEXT, settings, degraded=False)
    degraded = generate(QUESTION, CONTEXT, settings, degraded=True)
    charged_ms = degraded.ttft_ms - healthy.ttft_ms
    assert charged_ms >= DEGRADED_TTFT_PENALTY_S * 1000 * 0.9, (
        f"only {charged_ms:.0f}ms charged; the penalty has to be large enough "
        "to read on a latency panel next to a real model's TTFT"
    )


def test_degraded_mode_leads_with_lower_ranked_evidence(settings: Settings) -> None:
    """Degraded mode inverts the ranking, so the weakest evidence answers first."""
    healthy = generate(QUESTION, CONTEXT, settings, degraded=False)
    degraded = generate(QUESTION, CONTEXT, settings, degraded=True)
    # s.99 (access agreements) is irrelevant to the question; s.19 is the answer.
    assert healthy.text.lower().index("bereaved") < healthy.text.lower().index("access agreement")
    assert degraded.text.lower().index("access agreement") < degraded.text.lower().index("bereaved")


def test_prefix_cache_is_observed_not_invented() -> None:
    """Report a cache hit only for a prefix this client has actually seen."""
    client = StubClient()
    settings = Settings.from_env().with_overrides(model=STUB_MODEL, tracing_enabled=False)
    first = generate(QUESTION, CONTEXT, settings, client=client)
    second = generate(QUESTION, CONTEXT, settings, client=client)
    assert first.cached_tokens == 0, "nothing was cached on the first call"
    assert second.cached_tokens == count_tokens(SYSTEM_PROMPT)
    # A fresh client has an empty cache: the number is per-process, not global.
    assert generate(QUESTION, CONTEXT, settings, client=StubClient()).cached_tokens == 0


# --------------------------------------------------------------------------- #
# Latency
# --------------------------------------------------------------------------- #
def test_stream_timer_separates_ttft_from_inter_token_gaps() -> None:
    timer = StreamTimer()
    time.sleep(0.02)
    timer.record()
    for _ in range(3):
        time.sleep(0.01)
        timer.record()
    assert timer.ttft_ms >= 15, timer.ttft_ms
    assert 5 <= timer.itl_mean_ms <= 60, timer.itl_mean_ms
    assert timer.itl_p95_ms >= timer.itl_mean_ms * 0.5
    assert timer.tokens == 4
    assert len(timer.gaps_ms) == 3, "the first token has no preceding gap"


def test_stream_timer_reports_zero_when_nothing_streamed() -> None:
    timer = StreamTimer()
    assert timer.ttft_ms == 0.0 and timer.itl_mean_ms == 0.0 and timer.itl_p95_ms == 0.0


def test_generate_records_measurable_latency(settings: Settings) -> None:
    result = generate(QUESTION, CONTEXT, settings)
    assert result.ttft_ms > 0, "TTFT must be measured, not estimated"
    assert result.generation_ms >= result.ttft_ms
    assert result.completion_tokens > 0


def test_formula_and_measurement_are_reported_separately(settings: Settings) -> None:
    """The identity is an approximation; both sides are kept."""
    result = generate(QUESTION, CONTEXT, settings)
    assert result.predicted_e2e_ms() > 0
    assert result.generation_ms > 0


# --------------------------------------------------------------------------- #
# Tokens, cost and client selection
# --------------------------------------------------------------------------- #
def test_token_counting_is_deterministic_and_non_zero() -> None:
    assert count_tokens("") == 0
    assert count_tokens("hello world") == count_tokens("hello world")
    assert count_tokens("a" * 400) > 10


def test_cost_comes_from_the_result_not_the_client() -> None:
    result = LLMResult(
        text="x",
        model="claude-sonnet-5",
        prompt_tokens=1_000,
        completion_tokens=100,
        cached_tokens=0,
        ttft_ms=1.0,
        itl_ms_mean=1.0,
        itl_ms_p95=1.0,
        generation_ms=2.0,
        degraded=False,
    )
    total, breakdown = result.cost
    assert total > 0
    assert set(breakdown) == {"input_usd", "cached_input_usd", "output_usd"}


def test_missing_api_key_falls_back_to_the_stub_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = make_client(Settings.from_env().with_overrides(model="claude-sonnet-5"))
    assert isinstance(client, StubClient)


def test_unrecognised_model_id_falls_back_to_the_stub() -> None:
    client = make_client(Settings.from_env().with_overrides(model="llama-9b-instruct"))
    assert isinstance(client, StubClient)


def test_a_generation_failure_becomes_a_result_not_an_exception(settings: Settings) -> None:
    """A failed generation must still produce a row with an error on it."""

    class Exploding:
        model = "exploding"
        degraded = False

        def stream(self, system: str, prompt: str):
            raise RuntimeError("upstream 503")
            yield  # pragma: no cover

        def usage(self):
            return None

    result = generate(QUESTION, CONTEXT, settings, client=Exploding())  # type: ignore[arg-type]
    assert result.text == ""
    assert "upstream 503" in result.error


# --------------------------------------------------------------------------- #
# The degraded fallback works for any provider
# --------------------------------------------------------------------------- #
class _Recording:
    """A stand-in hosted client that reports what it was asked."""

    model = "some-hosted-model"

    def __init__(self, text: str = "[s.19] provides: a bereaved person is entitled to leave.") -> None:
        self.text = text
        self.systems: list[str] = []
        self.prompts: list[str] = []

    def stream(self, system: str, prompt: str):
        self.systems.append(system)
        self.prompts.append(prompt)
        for word in self.text.split(" "):
            yield word + " "

    def usage(self):
        return (100, 20, 0)


def test_degradation_wraps_any_client_not_just_the_stub(settings: Settings) -> None:
    """Otherwise the incident demo silently shows nothing whenever a hosted model
    is configured -- which is exactly when it is being demonstrated."""
    from rights_agent.llm import DegradedClient

    inner = _Recording()
    result = generate(QUESTION, CONTEXT, settings, client=DegradedClient(inner))
    assert result.degraded
    assert result.model == inner.model, "the real model answered; the record must say so"


def test_degradation_removes_citations(settings: Settings) -> None:
    from rights_agent.llm import DegradedClient

    healthy = generate(QUESTION, CONTEXT, settings, client=_Recording())
    degraded = generate(QUESTION, CONTEXT, settings, client=DegradedClient(_Recording()))
    assert healthy.citations and not degraded.citations
    assert "s.19" not in degraded.text or "[" not in degraded.text


def test_degradation_leads_with_lower_ranked_evidence(settings: Settings) -> None:
    from rights_agent.llm import DegradedClient

    inner = _Recording()
    generate(QUESTION, CONTEXT, settings, client=DegradedClient(inner))
    sent = inner.prompts[0]
    assert sent.index("s.99") < sent.index("s.19"), "context was not reversed"


def test_degradation_is_slower(settings: Settings) -> None:
    from rights_agent.llm import DegradedClient

    healthy = generate(QUESTION, CONTEXT, settings, client=_Recording())
    degraded = generate(QUESTION, CONTEXT, settings, client=DegradedClient(_Recording()))
    assert degraded.ttft_ms > healthy.ttft_ms
    assert degraded.generation_ms > healthy.generation_ms


def test_degradation_keeps_the_text_grounded(settings: Settings) -> None:
    """The honest caveat: a lexical groundedness check stays happy while the
    answers become uncitable. That is the argument for a family of signals."""
    from rights_agent.judges import HeuristicJudge
    from rights_agent.llm import DegradedClient

    degraded = generate(QUESTION, CONTEXT, settings, client=DegradedClient(_Recording()))
    scores = HeuristicJudge().score(QUESTION, CONTEXT, degraded.text, degraded.citations)
    assert scores.citation_coverage == 0.0
    assert scores.groundedness > 0.5, "the fallback still quotes the retrieved text"


def test_usage_passes_through_the_wrapper(settings: Settings) -> None:
    from rights_agent.llm import DegradedClient

    result = generate(QUESTION, CONTEXT, settings, client=DegradedClient(_Recording()))
    assert (result.prompt_tokens, result.completion_tokens) == (100, 20)


def test_citation_stripping_handles_split_and_unterminated_brackets() -> None:
    from rights_agent.llm import _without_citations

    assert "".join(_without_citations(iter(["a ", "[s.1", "9] b"]))) == "a  b"
    assert "".join(_without_citations(iter(["a [s.19"]))) == "a s.19"
    assert "".join(_without_citations(iter(["plain text"]))) == "plain text"
