"""DeepSeek over its OpenAI-compatible endpoint.

These tests stand a fake SDK in front of the client, so they assert what goes on
the wire and what comes back off it without needing a key or the network. The
two things worth pinning are the ones that fail silently: the ``thinking``
parameter, whose provider default is *enabled*, and reasoning text, which
arrives in its own field and must never reach the answer.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any, Iterator

import pytest

from rights_agent.config import PRICING, Settings, price_for
from rights_agent.llm import STUB_MODEL, DeepSeekClient, StubClient, make_client


# --------------------------------------------------------------------------- #
# A fake OpenAI SDK
# --------------------------------------------------------------------------- #
@dataclass
class _Delta:
    content: str | None = None
    reasoning_content: str | None = None


@dataclass
class _Choice:
    delta: _Delta


@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_cache_hit_tokens: int | None = None
    prompt_tokens_details: Any = None


@dataclass
class _Chunk:
    choices: list[_Choice] = field(default_factory=list)
    usage: _Usage | None = None


class _Recorder:
    """Captures the request and replays a scripted stream."""

    def __init__(self, chunks: list[_Chunk]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, Any]] = []
        self.base_urls: list[Any] = []
        self.api_keys: list[Any] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = self

        class FakeCompletions:
            def create(self, **kwargs: Any) -> Iterator[_Chunk]:
                recorder.calls.append(kwargs)
                return iter(recorder.chunks)

        class FakeChat:
            completions = FakeCompletions()

        class FakeOpenAI:
            def __init__(self, api_key: Any = None, base_url: Any = None, **_: Any) -> None:
                # Captured, not swallowed: the real SDK falls back to
                # OPENAI_API_KEY when no key is passed, so a fake that ignores
                # the argument cannot catch a provider getting no credentials.
                recorder.api_keys.append(api_key)
                recorder.base_urls.append(base_url)
                self.chat = FakeChat()

        module = types.ModuleType("openai")
        module.OpenAI = FakeOpenAI  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "openai", module)


def _text(content: str) -> _Chunk:
    return _Chunk(choices=[_Choice(_Delta(content=content))])


def _reasoning(content: str) -> _Chunk:
    return _Chunk(choices=[_Choice(_Delta(reasoning_content=content))])


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder(
        [
            _reasoning("Let me consider section 19 first."),
            _text("[s.19] provides: "),
            _text("a bereaved person is entitled to leave."),
            _Chunk(usage=_Usage(prompt_tokens=900, completion_tokens=40, prompt_cache_hit_tokens=768)),
        ]
    )
    rec.install(monkeypatch)
    return rec


# --------------------------------------------------------------------------- #
# The request
# --------------------------------------------------------------------------- #
def test_thinking_is_disabled_explicitly(recorder: _Recorder) -> None:
    """The provider default is ``enabled``. Omitting the parameter buys reasoning
    tokens and a much slower first token without asking."""
    client = DeepSeekClient("deepseek-v4-flash", thinking=False)
    list(client.stream("system", "prompt"))
    assert recorder.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}


def test_thinking_can_be_turned_on_deliberately(recorder: _Recorder) -> None:
    client = DeepSeekClient("deepseek-v4-flash", thinking=True)
    list(client.stream("system", "prompt"))
    assert recorder.calls[0]["extra_body"] == {"thinking": {"type": "enabled"}}


def test_the_request_is_deterministic_and_streamed(recorder: _Recorder) -> None:
    list(DeepSeekClient("deepseek-v4-flash").stream("system", "prompt"))
    call = recorder.calls[0]
    assert call["temperature"] == 0
    assert call["stream"] is True
    assert call["stream_options"] == {"include_usage": True}
    assert call["model"] == "deepseek-v4-flash"
    assert [m["role"] for m in call["messages"]] == ["system", "user"]


def test_it_points_at_deepseek_not_openai(recorder: _Recorder) -> None:
    DeepSeekClient("deepseek-v4-flash")
    assert recorder.base_urls == ["https://api.deepseek.com"]


def test_the_deepseek_key_is_passed_explicitly(
    recorder: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Left to the SDK it reads OPENAI_API_KEY, so DeepSeek would get no
    credentials and fail naming the wrong variable."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-should-not-be-used")
    DeepSeekClient("deepseek-v4-flash")
    assert recorder.api_keys == ["sk-deepseek"]


def test_the_openai_client_still_reads_its_own_variable(
    recorder: _Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rights_agent.llm import OpenAIClient

    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    OpenAIClient("gpt-4.1-mini")
    assert recorder.api_keys == ["sk-openai"]


def test_a_custom_base_url_is_honoured(recorder: _Recorder) -> None:
    """For a proxy or a self-hosted gateway."""
    DeepSeekClient("deepseek-v4-flash", base_url="https://gateway.internal/v1")
    assert recorder.base_urls == ["https://gateway.internal/v1"]


# --------------------------------------------------------------------------- #
# The response
# --------------------------------------------------------------------------- #
def test_reasoning_never_reaches_the_answer(recorder: _Recorder) -> None:
    """It arrives in its own field; an answer must never quietly contain
    chain-of-thought."""
    client = DeepSeekClient("deepseek-v4-flash")
    answer = "".join(client.stream("system", "prompt"))
    assert answer == "[s.19] provides: a bereaved person is entitled to leave."
    assert "Let me consider" not in answer


def test_reasoning_is_counted_rather_than_silently_dropped(recorder: _Recorder) -> None:
    """If it arrives while thinking is disabled, that is a fact about the
    provider worth surfacing -- not something to swallow."""
    client = DeepSeekClient("deepseek-v4-flash", thinking=False)
    list(client.stream("system", "prompt"))
    assert client.reasoning_chars == len("Let me consider section 19 first.")


def test_deepseeks_own_cache_field_is_authoritative(recorder: _Recorder) -> None:
    client = DeepSeekClient("deepseek-v4-flash")
    list(client.stream("system", "prompt"))
    assert client.usage() == (900, 40, 768)


def test_it_falls_back_to_the_openai_compatible_cache_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """So this keeps working if the providers converge on one spelling."""
    details = types.SimpleNamespace(cached_tokens=512)
    rec = _Recorder(
        [
            _text("answer"),
            _Chunk(usage=_Usage(prompt_tokens=800, completion_tokens=10, prompt_tokens_details=details)),
        ]
    )
    rec.install(monkeypatch)
    client = DeepSeekClient("deepseek-v4-flash")
    list(client.stream("system", "prompt"))
    assert client.usage() == (800, 10, 512)


def test_usage_is_none_when_the_provider_reports_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caller then counts tokens itself rather than recording a zero cost."""
    rec = _Recorder([_text("answer")])
    rec.install(monkeypatch)
    client = DeepSeekClient("deepseek-v4-flash")
    list(client.stream("system", "prompt"))
    assert client.usage() is None


def test_a_chunk_with_no_delta_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _Recorder([_Chunk(choices=[_Choice(delta=None)]), _text("answer")])  # type: ignore[arg-type]
    rec.install(monkeypatch)
    assert "".join(DeepSeekClient("deepseek-v4-flash").stream("s", "p")) == "answer"


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def test_a_deepseek_model_id_selects_the_deepseek_client(
    monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client = make_client(Settings.from_env().with_overrides(model="deepseek-v4-flash"))
    assert isinstance(client, DeepSeekClient)
    assert client.thinking is False, "thinking must be off unless asked for"


def test_the_thinking_setting_reaches_the_client(
    monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("RIGHTS_THINKING", "true")
    client = make_client(Settings.from_env().with_overrides(model="deepseek-v4-flash"))
    assert isinstance(client, DeepSeekClient) and client.thinking is True


def test_a_missing_key_falls_back_to_the_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """The demo runs with no key and no network; that is a fallback, not a crash."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = make_client(Settings.from_env().with_overrides(model="deepseek-v4-flash"))
    assert isinstance(client, StubClient)


def test_a_pricing_only_row_is_refused_with_the_reason(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Nobody should discover that off-peak is not a model id from a 400."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    with caplog.at_level("WARNING"):
        client = make_client(
            Settings.from_env().with_overrides(model="deepseek-v4-flash-offpeak")
        )
    assert isinstance(client, StubClient)
    assert "is a pricing row" in caplog.text


# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #
def test_the_published_deepseek_prices_are_in_the_table() -> None:
    flash = price_for("deepseek-v4-flash")
    assert (flash.input_per_mtok, flash.output_per_mtok, flash.cached_input_per_mtok) == (
        0.44,
        1.32,
        0.014,
    )


def test_off_peak_is_exactly_half_of_peak() -> None:
    peak = price_for("deepseek-v4-flash")
    off_peak = price_for("deepseek-v4-flash-offpeak")
    for attribute in ("input_per_mtok", "output_per_mtok", "cached_input_per_mtok"):
        assert getattr(off_peak, attribute) == pytest.approx(
            getattr(peak, attribute) / 2, rel=1e-9
        )


def test_the_cache_ratio_is_far_steeper_than_the_usual_tenth() -> None:
    """Which makes prompt layout the dominant cost lever on this model rather
    than one lever among several."""
    flash = price_for("deepseek-v4-flash")
    assert flash.input_per_mtok / flash.cached_input_per_mtok > 25


def test_flash_is_cheaper_than_pro_on_every_line() -> None:
    flash, pro = price_for("deepseek-v4-flash"), price_for("deepseek-v4-pro")
    for attribute in ("input_per_mtok", "output_per_mtok", "cached_input_per_mtok"):
        assert getattr(flash, attribute) < getattr(pro, attribute)


def test_pricing_rows_are_labelled_with_their_rate() -> None:
    assert "peak rate" in price_for("deepseek-v4-flash").label("deepseek-v4-flash")
    assert "off-peak" in price_for("deepseek-v4-flash-offpeak").label("x")


def test_only_deliberate_rows_are_pricing_only() -> None:
    pricing_only = {name for name, row in PRICING.items() if row.pricing_only}
    assert pricing_only == {"deepseek-v4-flash-offpeak"}
    assert STUB_MODEL not in pricing_only
