"""Generation: streaming clients, the offline stub, and latency measurement.

Latency is **measured**, not estimated.  The streaming loop timestamps the first
token (TTFT) and the gap before every subsequent one (ITL), and keeps the mean
and the p95 of those gaps.  The identity

    e2e ~= TTFT + ITL x (output_tokens - 1)

then has two measured sides, and the difference between them is
non-generation overhead -- orchestration, retrieval, serialisation, client.
That difference is the number almost nobody measures.

Three clients behind one protocol:

* :class:`StubClient` -- offline, deterministic, extractive.  The default.
* :class:`OpenAIClient` / :class:`AnthropicClient` -- real streaming, used when
  ``RIGHTS_MODEL`` names a hosted model and its key is present.

There is no demo-only path: the stub is a client like any other, and the tests
and the dashboard call the same :func:`generate`.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Protocol

from rights_agent.config import PROMPT_VERSION, Settings, cost_usd, price_for
from rights_agent.log import get_logger
from rights_agent.metrics import percentile
from rights_agent.telemetry import LLM, SEMCONV, span

log = get_logger("llm")

STUB_MODEL = "stub-local"

SYSTEM_PROMPT = (
    "You answer questions about a single legal document, using only the provisions "
    "quoted in the context. Report what the document says; do not advise, and do not "
    "generalise beyond the text. Cite the provision for every statement, in square "
    "brackets, as it appears in the context header. If the context does not answer the "
    "question, say so plainly.\n"
    # Brevity is a cost control, not a style preference: output tokens are dearer
    # per token than input on every model in the pricing table. It also keeps the
    # answer readable on a projector.
    "Be brief: at most four sentences, or four short bullets. Do not restate the "
    "question, and do not enumerate every provision you were given -- answer the one "
    "thing that was asked."
)

#: Per-token delay used by the offline stub, in seconds.  The stub has no model
#: to wait for, so without a cadence every inter-token latency would be zero and
#: the latency panel would have nothing to measure.  This is a *simulated
#: cadence for the offline path*, not a claim about any model's speed -- and it
#: is a property of the stub client itself, used identically by the tests and
#: the dashboard, not a demo-only branch.
STUB_TOKEN_DELAY_S = 0.0015

#: Degraded mode multiplies the cadence: slower answers, and no citations.
DEGRADED_DELAY_MULTIPLIER = 8

#: Sized against a *hosted* model's baseline, not the stub's. The offline stub
#: answers in single-digit milliseconds, so any penalty looks dramatic; a real
#: model already takes most of a second, and a +0.6s penalty moved a TTFT p50
#: from 1.01s to 1.37s -- true, and far too subtle to point at on a panel. A
#: real failover to an overloaded or weaker endpoint routinely doubles TTFT,
#: which is what this reproduces.
DEGRADED_TTFT_PENALTY_S = 1.5
DEGRADED_CHUNK_DELAY_S = 0.012

#: Where streamed tokens go, when someone is watching.
#:
#: A context variable rather than a parameter threaded through the graph: the
#: graph's node signatures belong to the workflow, not to whichever transport
#: happens to be rendering the answer. It is set for the duration of one
#: request and is therefore correct under the demo server's thread pool.
_token_sink: ContextVar[Callable[[str], None] | None] = ContextVar("token_sink", default=None)


@contextmanager
def stream_tokens_to(sink: Callable[[str], None] | None) -> Iterator[None]:
    """Send every generated chunk to ``sink`` for the duration of the block."""
    token = _token_sink.set(sink)
    try:
        yield
    finally:
        _token_sink.reset(token)


_SENTENCE_RE = re.compile(r"(?<=[.;])\s+")
#: A citation has to fit inside this to be recognised at all. 80 was sized
#: against ``[s.152]``-shaped citations and silently dropped 122 of the real
#: Act's 2141 leaf citations, the longest being 97 characters:
#: ``[Trade Union and Labour Relations (Consolidation) Act 1992 s.146B (as
#: inserted by Sch. 6 para. 63)]``. An unrecognised citation is not a formatting
#: nit -- the sentence carrying it reads as uncited, so citation coverage scores
#: a correct answer at zero.
MAX_CITATION_CHARS = 160
CITATION_RE = re.compile(rf"\[([^\]\n]{{2,{MAX_CITATION_CHARS}}})\]")
_CITATION_RE = CITATION_RE
_WORD_RE = re.compile(r"[a-z0-9£][a-z0-9£'\-]*")


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class LLMResult:
    """A completed generation, with everything needed to price and time it."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    ttft_ms: float
    itl_ms_mean: float
    itl_ms_p95: float
    generation_ms: float
    degraded: bool
    prompt_version: str = PROMPT_VERSION
    citations: tuple[str, ...] = ()
    error: str = ""

    @property
    def cost(self) -> tuple[float, dict[str, float]]:
        return cost_usd(self.model, self.prompt_tokens, self.completion_tokens, self.cached_tokens)

    def predicted_e2e_ms(self) -> float:
        """``TTFT + ITL x (tokens - 1)`` -- generation time only.

        Expect this to disagree with :attr:`generation_ms`, and for a reason
        worth knowing: ITL is measured per *chunk*, because a chunk is what a
        streaming API actually delivers, while the multiplier is a *token*
        count.  Most providers pack several tokens into a chunk, so the identity
        is an approximation on both sides -- which is why the measured number
        is the one that goes on the metrics row.
        """
        return round(self.ttft_ms + self.itl_ms_mean * max(0, self.completion_tokens - 1), 3)


@dataclass
class StreamTimer:
    """Collects TTFT and inter-token latencies from a stream."""

    started: float = field(default_factory=time.perf_counter)
    first_token_at: float | None = None
    last_token_at: float | None = None
    gaps_ms: list[float] = field(default_factory=list)
    tokens: int = 0

    def record(self) -> None:
        now = time.perf_counter()
        self.tokens += 1
        if self.first_token_at is None:
            self.first_token_at = now
        elif self.last_token_at is not None:
            self.gaps_ms.append((now - self.last_token_at) * 1_000)
        self.last_token_at = now

    @property
    def ttft_ms(self) -> float:
        if self.first_token_at is None:
            return 0.0
        return round((self.first_token_at - self.started) * 1_000, 3)

    @property
    def itl_mean_ms(self) -> float:
        return round(mean(self.gaps_ms), 3) if self.gaps_ms else 0.0

    @property
    def itl_p95_ms(self) -> float:
        return round(percentile(self.gaps_ms, 0.95), 3) if self.gaps_ms else 0.0

    @property
    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self.started) * 1_000, 3)


# --------------------------------------------------------------------------- #
# Token counting
# --------------------------------------------------------------------------- #
_encoder: object | None = None
_encoder_tried = False


def count_tokens(text: str, model: str = STUB_MODEL) -> int:
    """Token count, exact where possible and clearly approximate otherwise.

    ``tiktoken`` is used when installed; otherwise a 4-characters-per-token
    approximation.  The approximation is deterministic, which matters more for
    the offline path than being exactly right: a cost number that changes
    between runs cannot gate anything.
    """
    global _encoder, _encoder_tried
    if not text:
        return 0
    if not _encoder_tried:
        _encoder_tried = True
        try:
            import tiktoken

            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:  # noqa: BLE001 - optional dependency
            log.debug("tiktoken unavailable, approximating token counts: %s", exc)
            _encoder = None
    if _encoder is not None:
        try:
            return len(_encoder.encode(text))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001, S110 - falls through to the estimate
            pass
    return max(1, (len(text) + 3) // 4)


# --------------------------------------------------------------------------- #
# Client protocol
# --------------------------------------------------------------------------- #
class LLMClient(Protocol):
    """A streaming text generator."""

    model: str

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        """Yield chunks of the answer, in order."""

    def usage(self) -> tuple[int, int, int] | None:
        """``(prompt, completion, cached)`` tokens if the provider reported them."""


# --------------------------------------------------------------------------- #
# Offline stub
# --------------------------------------------------------------------------- #
@dataclass
class _Block:
    citation: str
    breadcrumb: str
    text: str


def parse_context(context: str) -> list[_Block]:
    """Split an assembled context back into blocks.

    The stub and the judges both need the blocks, and re-parsing the assembled
    string keeps a single source of truth: whatever the model was shown is
    exactly what gets scored.
    """
    blocks: list[_Block] = []
    for chunk in context.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        header, _, body = chunk.partition("\n")
        match = _CITATION_RE.match(header.strip())
        if match is None:
            continue
        blocks.append(
            _Block(
                citation=match.group(1).strip(),
                breadcrumb=header.strip()[match.end() :].strip(),
                text=body.strip(),
            )
        )
    return blocks


class StubClient:
    """Deterministic, offline, extractive generator.

    It quotes the retrieved provisions rather than paraphrasing them, which is
    honest about what it is: no model is running.  Because it is extractive, a
    groundedness judge should score it high -- and if it does not, the judge is
    the thing that is wrong.

    Prompt caching is *observed*, not simulated: the client keeps a set of
    hashed prompt prefixes it has already been given in this process, and
    reports the system-prompt prefix as cached on a repeat.  That is a real
    prefix cache; it is simply an in-process one rather than a vendor's.
    """

    def __init__(
        self,
        *,
        degraded: bool = False,
        max_chars: int = 2_000,
        token_delay_s: float = STUB_TOKEN_DELAY_S,
    ) -> None:
        self.model = STUB_MODEL
        self.degraded = degraded
        self.max_chars = max_chars
        self.token_delay_s = token_delay_s * (DEGRADED_DELAY_MULTIPLIER if degraded else 1)
        self._prefix_cache: set[str] = set()
        self._usage: tuple[int, int, int] | None = None

    # ---- selection --------------------------------------------------------
    @staticmethod
    def _overlap(question: str, block: _Block) -> int:
        terms = {word for word in _WORD_RE.findall(question.lower()) if len(word) >= 4}
        haystack = set(_WORD_RE.findall(f"{block.breadcrumb} {block.text}".lower()))
        return len(terms & haystack)

    def _select(self, question: str, blocks: Sequence[_Block]) -> list[_Block]:
        if not blocks:
            return []
        ranked = sorted(
            enumerate(blocks),
            key=lambda pair: (-self._overlap(question, pair[1]), pair[0]),
        )
        ordered = [block for _, block in ranked]
        if self.degraded:
            # Degraded mode picks lower-ranked evidence: the signature of a
            # primary model failing over to a weaker fallback.
            ordered.reverse()
        return ordered[:3]

    def _compose(self, question: str, context: str) -> str:
        blocks = parse_context(context)
        chosen = self._select(question, blocks)
        if not chosen:
            return "The retrieved context does not contain a provision that answers this question."

        pieces: list[str] = []
        for block in chosen:
            sentences = [s.strip() for s in _SENTENCE_RE.split(block.text) if s.strip()]
            # One sentence per block, so every sentence in the answer has
            # exactly one citation attached to it.  A citation that covers two
            # sentences is a citation that covers one of them.
            quote = sentences[0][: self.max_chars // 3].strip() if sentences else ""
            if not quote:
                continue
            if self.degraded:
                # Citations dropped: the other half of the degraded signature.
                pieces.append(f"The document provides that {quote}")
            else:
                pieces.append(f"[{block.citation}] provides: {quote}")
        if not pieces:
            return "The retrieved context does not contain a provision that answers this question."
        answer = " ".join(pieces)
        return answer[: self.max_chars].rstrip()

    # ---- streaming --------------------------------------------------------
    def stream(self, system: str, prompt: str) -> Iterator[str]:
        question, context = _split_prompt(prompt)
        answer = self._compose(question, context)

        prefix = hashlib.sha256(system.encode("utf-8")).hexdigest()
        cached = count_tokens(system) if prefix in self._prefix_cache else 0
        self._prefix_cache.add(prefix)
        self._usage = (
            count_tokens(f"{system}\n{prompt}"),
            count_tokens(answer),
            cached,
        )

        # A failover costs a first token before it costs anything else: the
        # primary has to fail, or time out, before the fallback is asked. Paying
        # that only in inter-token delay left ttft_ms indistinguishable from
        # jitter (16ms degraded against 28ms healthy on one run), so the one
        # number the incident demo points at moved the wrong way. Charged here so
        # the stub degrades the same way DegradedClient does for hosted models.
        if self.degraded:
            time.sleep(DEGRADED_TTFT_PENALTY_S)

        # Stream word by word so TTFT and ITL are measured against something
        # real rather than a single blob.
        for index, word in enumerate(answer.split(" ")):
            if self.token_delay_s > 0:
                time.sleep(self.token_delay_s)
            yield word if index == 0 else f" {word}"

    def usage(self) -> tuple[int, int, int] | None:
        return self._usage


def _split_prompt(prompt: str) -> tuple[str, str]:
    """Recover ``(question, context)`` from an assembled prompt."""
    question = ""
    context = ""
    if "Question:" in prompt:
        _, _, tail = prompt.partition("Question:")
        question, _, rest = tail.partition("\n")
        question = question.strip()
        context = rest.partition("Context:")[2].strip() if "Context:" in rest else rest.strip()
    elif "Context:" in prompt:
        context = prompt.partition("Context:")[2].strip()
    return question, context


# --------------------------------------------------------------------------- #
# Hosted clients
# --------------------------------------------------------------------------- #
class OpenAIClient:
    """Streaming chat completions over an OpenAI-compatible endpoint.

    Imported lazily and optionally, so the offline path never needs the SDK.
    Subclassed rather than copied for other providers that speak the same
    protocol -- see :class:`DeepSeekClient`.
    """

    #: ``None`` means the SDK's own default (the OpenAI API).
    base_url: str | None = None
    #: Environment variable holding the key, named in the error when it is unset.
    api_key_env: str = "OPENAI_API_KEY"

    def __init__(self, model: str, *, base_url: str | None = None) -> None:
        from openai import OpenAI  # imported here so the offline path never needs it

        self.model = model
        # The key must be passed explicitly. Left to the SDK, it reads
        # OPENAI_API_KEY -- so a provider whose key lives anywhere else silently
        # gets no credentials at all, and the failure surfaces as "Missing
        # credentials" naming the wrong variable.
        self._client = OpenAI(
            api_key=os.environ.get(self.api_key_env),
            base_url=base_url or self.base_url or None,
        )
        self._usage: tuple[int, int, int] | None = None

    # ---- provider hooks ---------------------------------------------------
    def _extra_body(self) -> dict[str, Any]:
        """Provider-specific request fields the OpenAI schema does not cover."""
        return {}

    def _is_reasoning_delta(self, delta: Any) -> str:
        """Reasoning text in this chunk, if the provider streams it separately."""
        return ""

    def _cached_tokens(self, usage: Any) -> int:
        details = getattr(usage, "prompt_tokens_details", None)
        return int(getattr(details, "cached_tokens", 0) or 0)

    # ---- streaming --------------------------------------------------------
    def stream(self, system: str, prompt: str) -> Iterator[str]:
        self._usage = None
        self.reasoning_chars = 0
        extra_body = self._extra_body()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            stream=True,
            stream_options={"include_usage": True},
            temperature=0,
            **({"extra_body": extra_body} if extra_body else {}),
        )
        for chunk in response:
            if chunk.usage is not None:
                self._usage = (
                    chunk.usage.prompt_tokens,
                    chunk.usage.completion_tokens,
                    self._cached_tokens(chunk.usage),
                )
            for choice in chunk.choices or ():
                delta = choice.delta
                if delta is None:
                    continue
                # Reasoning is never part of the answer. Counted rather than
                # silently dropped: if it arrives while we asked for it to be
                # off, that is a fact about the provider worth surfacing.
                self.reasoning_chars += len(self._is_reasoning_delta(delta))
                if delta.content:
                    yield delta.content

    def usage(self) -> tuple[int, int, int] | None:
        return self._usage


class DeepSeekClient(OpenAIClient):
    """DeepSeek over its OpenAI-compatible endpoint.

    Two things this must get right, neither of them optional:

    * **``thinking`` defaults to ``enabled``.**  Omitting it buys reasoning
      tokens on every request and a far slower first token, without asking.  So
      the mode is always sent explicitly, in both directions.
    * **Reasoning text arrives in its own field** (``reasoning_content``) in the
      streaming delta.  It is excluded from the answer and counted, so an answer
      can never quietly contain chain-of-thought -- and if any shows up while
      thinking is disabled, the count says so rather than the text appearing.

    Cache accounting differs too: DeepSeek reports ``prompt_cache_hit_tokens``
    alongside the OpenAI-compatible field, and its cache-hit input is roughly a
    thirty-first of cache-miss rather than the usual tenth, which makes prompt
    layout the dominant cost lever on this model.
    """

    base_url = "https://api.deepseek.com"
    api_key_env = "DEEPSEEK_API_KEY"

    def __init__(
        self, model: str, *, base_url: str | None = None, thinking: bool = False
    ) -> None:
        super().__init__(model, base_url=base_url)
        self.thinking = thinking
        self.reasoning_chars = 0

    def _extra_body(self) -> dict[str, Any]:
        return {"thinking": {"type": "enabled" if self.thinking else "disabled"}}

    def _is_reasoning_delta(self, delta: Any) -> str:
        return str(getattr(delta, "reasoning_content", "") or "")

    def _cached_tokens(self, usage: Any) -> int:
        # DeepSeek's own field is authoritative; fall back to the
        # OpenAI-compatible one so this keeps working if they converge.
        native = getattr(usage, "prompt_cache_hit_tokens", None)
        if native is not None:
            return int(native or 0)
        return super()._cached_tokens(usage)


class AnthropicClient:
    """Streaming Anthropic messages.  Imported lazily and optionally."""

    def __init__(self, model: str) -> None:
        from anthropic import Anthropic

        self.model = model
        self._client = Anthropic()
        self._usage: tuple[int, int, int] | None = None

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        self._usage = None
        with self._client.messages.stream(
            model=self.model,
            max_tokens=1_024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        ) as stream:
            yield from stream.text_stream
            final = stream.get_final_message()
            usage = final.usage
            self._usage = (
                usage.input_tokens,
                usage.output_tokens,
                getattr(usage, "cache_read_input_tokens", 0) or 0,
            )

    def usage(self) -> tuple[int, int, int] | None:
        return self._usage


# --------------------------------------------------------------------------- #
# The degraded fallback, for any provider
# --------------------------------------------------------------------------- #
#: Extra delay before the first chunk, and between chunks, in degraded mode.
#:
class DegradedClient:
    """Wraps any client to reproduce the signature of a weaker fallback.

    Three changes, each mirroring something a real failover does:

    * **Leads with lower-ranked evidence** -- the context blocks are reversed, so
      the model answers from the weakest retrieval rather than the strongest.
    * **Produces no citations** -- bracketed citations are removed from the
      output. Asking a model not to cite does not work: DeepSeek kept citing
      through an explicit instruction, and again after the citation was stripped
      from the context header, because the breadcrumb still names the section.
      A capable model is *hard to make careless*, so the behaviour being
      simulated -- a weaker model that ignores the citation contract -- is
      simulated directly rather than hoped for.
    * **Answers slower** -- a delay before the first chunk and between chunks.

    A wrapper rather than a flag on each client, because otherwise every provider
    needs its own degradation and the one that matters during a demo is whichever
    is configured that day. :class:`StubClient` keeps its own built-in version
    for one honest reason: it is extractive and ignores prompts entirely, so
    rewriting the prompt cannot change what it says -- its selection has to
    change instead.

    ``model`` reports the **inner** model, because the real model did answer.
    The record says what served and that it was degraded; inventing a different
    model name here would be the exact silent-failover confusion the metrics row
    exists to prevent.
    """

    degraded = True

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner

    @property
    def model(self) -> str:
        return self._inner.model

    def usage(self) -> tuple[int, int, int] | None:
        return self._inner.usage()

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        question, context = _split_prompt(prompt)
        blocks = parse_context(context)
        if blocks:
            # Reversed *and* stripped of the bracketed citation: lower-ranked
            # evidence first, and no provision id to attribute a claim to.
            degraded_context = "\n\n".join(
                f"{block.breadcrumb}\n{block.text}".strip() for block in reversed(blocks)
            )
            prompt = build_prompt(question, degraded_context)
        degraded_system = (
            system.replace(
                "Cite the provision for every statement, in square brackets, as it "
                "appears in the context header.",
                "Do not cite provisions and do not use square brackets.",
            )
            + "\nAnswer in general terms. Do not name specific provisions."
        )
        time.sleep(DEGRADED_TTFT_PENALTY_S)
        for chunk in _without_citations(self._inner.stream(degraded_system, prompt)):
            time.sleep(DEGRADED_CHUNK_DELAY_S)
            yield chunk


def _without_citations(chunks: Iterator[str]) -> Iterator[str]:
    """Drop ``[...]`` markers from a stream, across chunk boundaries.

    Buffered rather than applied per chunk: a citation routinely arrives split
    across several chunks, and a per-chunk regex would let half of one through.
    """
    held = ""
    for chunk in chunks:
        held += chunk
        emit: list[str] = []
        while "[" in held:
            head, _, rest = held.partition("[")
            if "]" not in rest:
                # An unterminated bracket: hold it until the closer arrives.
                emit.append(head)
                held = "[" + rest
                break
            emit.append(head)
            held = rest.partition("]")[2]
        else:
            emit.append(held)
            held = ""
        text = "".join(emit)
        if text:
            yield text
    if held:
        yield held.replace("[", "").replace("]", "")


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def make_client(settings: Settings, *, degraded: bool | None = None) -> LLMClient:
    """Pick a client for ``settings.model``, falling back to the stub.

    A missing key or a missing SDK is a fallback, not a crash: the demo must
    run with no API key and no network.
    """
    degraded = settings.degraded if degraded is None else degraded
    model = settings.model
    if model == STUB_MODEL:
        return StubClient(degraded=degraded, max_chars=settings.max_answer_chars)

    try:
        price = price_for(model)
        if price.pricing_only:
            raise RuntimeError(
                f"{model!r} is a pricing row, not a model id the API accepts "
                f"({price.note}). Set RIGHTS_MODEL to the model itself and reprice "
                "against this row instead."
            )
        hosted: LLMClient
        if model.startswith("deepseek"):
            if not os.environ.get(DeepSeekClient.api_key_env):
                raise RuntimeError(f"{DeepSeekClient.api_key_env} is not set")
            hosted = DeepSeekClient(
                model, base_url=settings.deepseek_base_url, thinking=settings.thinking
            )
        elif model.startswith("claude"):
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            hosted = AnthropicClient(model)
        elif model.startswith(("gpt", "o1", "o3", "o4")):
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set")
            hosted = OpenAIClient(model)
        else:
            raise RuntimeError(f"unrecognised model id {model!r}")
        # Degradation has to work for whichever provider is configured, or the
        # incident demo silently shows nothing on the day.
        return DegradedClient(hosted) if degraded else hosted
    except Exception as exc:  # noqa: BLE001 - any failure means "use the stub"
        # A fallback, not a crash: the demo must run with no key and no network.
        # Logged at warning because silently answering from the stub while the
        # operator believes a hosted model is serving is its own kind of wrong.
        log.warning("falling back to %s: %s", STUB_MODEL, exc)
        return StubClient(degraded=degraded, max_chars=settings.max_answer_chars)


def build_prompt(question: str, context: str) -> str:
    """The user-side prompt.  Layout is stable, so prompt caching can bite."""
    return (
        f"Question: {question}\n\n"
        f"Answer using only the provisions below. Cite each one in square brackets.\n\n"
        f"Context:\n{context if context.strip() else '(no provisions retrieved)'}"
    )


def extract_citations(answer: str) -> list[str]:
    """Citations in the order they appear, de-duplicated."""
    seen: list[str] = []
    for match in _CITATION_RE.finditer(answer):
        citation = match.group(1).strip()
        if citation and citation not in seen:
            seen.append(citation)
    return seen


def generate(
    question: str,
    context: str,
    settings: Settings,
    *,
    client: LLMClient | None = None,
    degraded: bool | None = None,
    on_token: Callable[[str], None] | None = None,
) -> LLMResult:
    """Stream an answer and return it with measured latency and token counts."""
    client = client or make_client(settings, degraded=degraded)
    on_token = on_token or _token_sink.get()
    is_degraded = bool(getattr(client, "degraded", False))
    prompt = build_prompt(question, context)
    timer = StreamTimer()
    chunks: list[str] = []
    error = ""

    with span(
        "rag.generate",
        LLM,
        **{
            SEMCONV.LLM_MODEL_NAME: client.model,
            SEMCONV.INPUT_VALUE: question,
            "metadata.prompt_version": PROMPT_VERSION,
            "metadata.degraded": is_degraded,
            "llm.prompt_chars": len(prompt),
        },
    ) as current:
        try:
            for chunk in client.stream(SYSTEM_PROMPT, prompt):
                timer.record()
                chunks.append(chunk)
                if on_token is not None:
                    on_token(chunk)
        except Exception as exc:  # noqa: BLE001 - a generation failure is a result
            error = f"{type(exc).__name__}: {exc}"
            log.error("generation failed: %s", error)
            current.record_exception(exc)

        answer = "".join(chunks).strip()
        reported = client.usage()
        if reported is not None:
            prompt_tokens, completion_tokens, cached_tokens = reported
        else:
            prompt_tokens = count_tokens(f"{SYSTEM_PROMPT}\n{prompt}", client.model)
            completion_tokens = count_tokens(answer, client.model)
            cached_tokens = 0

        result = LLMResult(
            text=answer,
            model=client.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            ttft_ms=timer.ttft_ms,
            itl_ms_mean=timer.itl_mean_ms,
            itl_ms_p95=timer.itl_p95_ms,
            generation_ms=timer.elapsed_ms,
            degraded=is_degraded,
            citations=tuple(extract_citations(answer)),
            error=error,
        )
        total_cost, _breakdown = result.cost
        price = price_for(result.model)
        current.set_attributes(
            {
                SEMCONV.OUTPUT_VALUE: answer,
                SEMCONV.LLM_TOKEN_COUNT_PROMPT: prompt_tokens,
                SEMCONV.LLM_TOKEN_COUNT_COMPLETION: completion_tokens,
                SEMCONV.LLM_TOKEN_COUNT_TOTAL: prompt_tokens + completion_tokens,
                "llm.token_count.prompt_details.cache_read": cached_tokens,
                "llm.latency.ttft_ms": result.ttft_ms,
                "llm.latency.itl_ms_mean": result.itl_ms_mean,
                "llm.latency.itl_ms_p95": result.itl_ms_p95,
                "llm.cost.total_usd": total_cost,
                "llm.cost.is_reference_price": price.is_reference,
                "llm.citations": list(result.citations),
            }
        )
    return result
