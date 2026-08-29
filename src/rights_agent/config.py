"""Settings, paths and the dated pricing table.

Every value here has a default that works with no ``.env``, no API key and no
network (constraint 1 of the specification).  Settings are read from the
environment exactly once per process and validated eagerly: a bad
``RIGHTS_SUFFICIENCY`` should fail at startup with a readable message, not produce
a gate that silently never fires.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Versions that participate in ``index_version`` / experiment tagging.
# Bump PARSER_VERSION whenever the tree shape changes; bump PROMPT_VERSION
# whenever the generation prompt changes.  Both end up on every metrics row.
# --------------------------------------------------------------------------- #
#: Bumped whenever the tree shape changes, because ``index_version`` is a promise
#: that two indexes with the same string contain the same thing. parser-4 taught
#: the parser the real Act's layout: Schedule paragraphs opening with their first
#: subsection inline, inserted provisions written with a single space, schedule
#: markers carrying their authorising section, and a drop-capped enacting
#: formula. It took uncitable leaves from 39% to 0% and found 87 inserted
#: provisions where parser-3 found 20 -- a different index, and it must not claim
#: to be the same one.
#:
#: parser-5 finished the attribution job parser-4 started. Schedule paragraphs
#: are set one space in, and requiring column 0 dropped all 180 of them, which
#: stranded the provisions they insert: 47 of 87 inserted provisions cited "the
#: host Act", naming no statute a reader could look up. Three changes fixed it --
#: indent-tolerant paragraph recognition, parentheses allowed in statute short
#: titles ("Trade Union and Labour Relations (Consolidation) Act 1992"), and
#: following the ``SCHEDULE 5   Section 56`` cross-reference to the body section
#: that names the Act a schedule amends. Result: 0 of 87 hostless, 0 orphan
#: subsections, 2141 leaves against parser-4's 1930.
#:
#: parser-6 stopped reading wrapped cross-references as new subsections. A
#: reference that wraps across its own number --
#: ``...that comply with subsection`` / ``(2) of that section.`` -- produced a
#: subsection whose entire text was ``of that section.``: a chunk that says
#: nothing, sits in the searched index, and turned up as cited evidence in an
#: answer. 63 of the Act's leaves were fragments of that kind, and one of them
#: reached the committed calibration set.
PARSER_VERSION: Final[str] = "parser-6"
PROMPT_VERSION: Final[str] = "prompt-3"

#: Accepted ``RIGHTS_EMBEDDER`` values.
#:
#: Duplicated from :mod:`rights_agent.embedding` rather than imported: that
#: module pulls in chromadb, and config is imported by everything including the
#: CLI's ``--help``. ``tests/test_config.py`` asserts the two agree, so the
#: duplication cannot drift silently.
EMBEDDER_CHOICES: Final[frozenset[str]] = frozenset(
    {
        "auto",
        "hashing",
        "onnx",
        "openai",
        "openai-small",
        "openai-large",
        "hashing-bow-512",
        "onnx-all-MiniLM-L6-v2",
        "openai-text-embedding-3-small",
        "openai-text-embedding-3-large",
    }
)

#: Collection names.  ``leaves`` is searched; ``parents`` is only ever fetched
#: by id, to widen a leaf hit into its surrounding provision.
LEAF_COLLECTION: Final[str] = "corpus_leaves"
PARENT_COLLECTION: Final[str] = "corpus_parents"
SIMPLE_COLLECTION: Final[str] = "corpus_simple"

#: Fixed-window baseline (§8).  ~250 tokens with ~15% overlap.
CHUNK_CHARS: Final[int] = 1_000
OVERLAP_CHARS: Final[int] = 150


#: AI Act Articles 19/26(6) floor, in days.  Duplicated as a plain constant so
#: settings validation does not have to import the audit module.  See
#: ``rights_agent.audit`` for what it does and does not mean.
RETENTION_FLOOR_DAYS: Final[int] = 183


class ConfigError(RuntimeError):
    """Raised when the environment cannot be turned into valid settings."""


# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #
#: The single date on which every cost figure in this project is based.  Change
#: the table below and every cost number, projection and breakdown moves with
#: it -- that property is the difference between a cost model and a spreadsheet.
PRICING_AS_OF: Final[str] = "2026-08-28"


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """USD per **million** tokens.

    Two ratios survive any price change and are the reason this table is worth
    having: output is typically several times input, and cached input is
    roughly a tenth of fresh input.  Those are what make brevity and prompt
    layout into cost levers rather than style preferences.
    """

    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float
    #: True when this row prices a model that costs nothing to run locally.
    #: The offline stub is priced *as if* it were a small hosted model so the
    #: cost panel shows a real number offline.  Every surface that reports a
    #: reference cost must say so -- see :meth:`ModelPrice.label`.
    is_reference: bool = False
    reference_of: str = ""
    #: True for a row that exists to be *repriced against* but is not a model id
    #: the API will accept -- DeepSeek's off-peak rates, for instance, are the
    #: same model at a different time of day.  :func:`~rights_agent.llm.make_client`
    #: refuses these, so nobody discovers it from a 400.
    pricing_only: bool = False
    note: str = ""

    def label(self, model: str) -> str:
        if self.is_reference:
            return f"{model} (priced as {self.reference_of}, reference only)"
        if self.note:
            return f"{model} ({self.note})"
        return model


PRICING: Final[dict[str, ModelPrice]] = {
    # Offline stub -- costs nothing, priced as a small hosted model so that the
    # cost mechanics are demonstrable with the network off.
    "stub-local": ModelPrice(0.80, 4.00, 0.08, is_reference=True, reference_of="claude-haiku-4-5"),
    # ---- DeepSeek -------------------------------------------------------- #
    # Published list prices as of PRICING_AS_OF, from
    # https://api-docs.deepseek.com/quick_start/pricing
    #
    # DeepSeek prices by time of day: off-peak is half of peak, where peak is
    # 01:00-04:00 and 06:00-10:00 UTC on weekdays. The peak row is the one to
    # budget with; the off-peak row is here so the same recorded traffic can be
    # repriced against it. That comparison is unusual and worth showing -- the
    # only variable is *when you ran it*.
    #
    # Note the cache ratio: cache-hit input is about 1/31 of cache-miss, far
    # steeper than the ~1/10 typical elsewhere. On this model, prompt layout is
    # the dominant cost lever rather than one lever among several.
    "deepseek-v4-flash": ModelPrice(0.44, 1.32, 0.014, note="peak rate"),
    "deepseek-v4-flash-offpeak": ModelPrice(
        0.22, 0.66, 0.007, pricing_only=True, note="off-peak rate, same model"
    ),
    "deepseek-v4-pro": ModelPrice(1.32, 3.96, 0.044, note="peak rate"),
    # ---- Illustrative rows for the repricing comparison ------------------ #
    "claude-haiku-4-5": ModelPrice(1.00, 5.00, 0.10),
    "claude-sonnet-5": ModelPrice(3.00, 15.00, 0.30),
    "claude-opus-5": ModelPrice(15.00, 75.00, 1.50),
    "gpt-4.1-mini": ModelPrice(0.40, 1.60, 0.10),
    "gpt-4.1": ModelPrice(2.00, 8.00, 0.50),
}

#: Used when an unknown model id shows up, so a cost column is never silently 0.
FALLBACK_PRICE: Final[ModelPrice] = PRICING["claude-haiku-4-5"]

# --------------------------------------------------------------------------- #
# The rest of the bill.
#
# Model tokens are not the whole cost, and a cost model that stops at tokens
# sends teams optimising the wrong term.  These are **illustrative, dated
# assumptions** from the same table as everything else: change them and every
# figure downstream moves.  Each is named so it can be argued with.
# --------------------------------------------------------------------------- #

#: Compute, orchestration and the vector store, amortised per request.  Small
#: and roughly fixed; it dominates the panel only because the offline stub is
#: priced at zero -- pricing any real model moves generation to the top, which
#: is a contrast worth naming out loud rather than hiding.
INFRA_USD_PER_REQUEST: Final[float] = 0.00008

#: Object-storage price for retained spans, per GB-month.
TRACE_STORAGE_USD_PER_GB_MONTH: Final[float] = 0.023

#: How long spans are kept.  Trace storage is cheap per request and genuinely
#: large in aggregate, and the multiplier is the retention period.
TRACE_RETENTION_MONTHS: Final[float] = 6.0

#: Bytes of span payload beyond the context and answer we can measure: span
#: names, attributes, ids, timestamps, and the exporter's framing.
TRACE_OVERHEAD_BYTES: Final[int] = 6_000

#: The judge is an LLM call, so scoring *every* request can materially increase
#: the bill.  That -- money, not statistical elegance -- is the real reason
#: production evaluation is sampled.
JUDGE_SAMPLE_RATE: Final[float] = 0.10
JUDGE_MODEL: Final[str] = "claude-haiku-4-5"

#: A judge reads the question, the context and the answer, and emits a few
#: tokens of JSON. Its input is therefore close to the generation prompt.
JUDGE_OUTPUT_TOKENS: Final[int] = 40


def price_for(model: str) -> ModelPrice:
    """Price row for ``model``, falling back to a documented default."""
    return PRICING.get(model, FALLBACK_PRICE)


def cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> tuple[float, dict[str, float]]:
    """Total cost and its component breakdown, derived only from :data:`PRICING`.

    ``cached_tokens`` is the subset of ``prompt_tokens`` that was served from a
    prompt cache; it is billed at the cached rate and removed from the fresh
    input count.
    """
    if prompt_tokens < 0 or completion_tokens < 0 or cached_tokens < 0:
        raise ValueError("token counts must be non-negative")
    cached = min(cached_tokens, prompt_tokens)
    fresh = prompt_tokens - cached
    p = price_for(model)
    breakdown = {
        "input_usd": round(fresh * p.input_per_mtok / 1e6, 8),
        "cached_input_usd": round(cached * p.cached_input_per_mtok / 1e6, 8),
        "output_usd": round(completion_tokens * p.output_per_mtok / 1e6, 8),
    }
    return round(sum(breakdown.values()), 8), breakdown


# --------------------------------------------------------------------------- #
# Environment parsing helpers
# --------------------------------------------------------------------------- #
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _raw(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int, *, low: int, high: int) -> int:
    raw = _raw(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not an integer") from exc
    if not low <= value <= high:
        raise ConfigError(f"{name}={value} outside supported range [{low}, {high}]")
    return value


def _env_float(name: str, default: float, *, low: float, high: float) -> float:
    raw = _raw(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not a number") from exc
    if not low <= value <= high:
        raise ConfigError(f"{name}={value} outside supported range [{low}, {high}]")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = _raw(name, "true" if default else "false").lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ConfigError(f"{name}={raw!r} is not a boolean (use one of {sorted(_TRUE | _FALSE)})")


def _env_path(name: str, default: str) -> Path:
    return Path(_raw(name, default)).expanduser().resolve()


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable, validated process configuration."""

    # -- corpus and artefacts ------------------------------------------------
    data_dir: Path
    runs_dir: Path
    corpus_path: Path

    # -- embedding -----------------------------------------------------------
    #: ``auto`` prefers Chroma's ONNX MiniLM and falls back to the offline
    #: hashing embedder; every other value forces one exactly. ``openai`` and
    #: ``openai-large`` bill per token, so ``auto`` never selects them.
    embedder: str

    # -- retrieval -----------------------------------------------------------
    top_k: int
    sufficiency_threshold: float
    max_attempts: int
    max_parent_chars: int
    context_budget_chars: int
    min_block_chars: int

    # -- generation ----------------------------------------------------------
    model: str
    max_answer_chars: int
    degraded: bool
    #: Whether to let a model that supports it think before answering.
    #:
    #: Off by default, and sent explicitly rather than left to the provider:
    #: DeepSeek's ``thinking`` parameter defaults to *enabled*, so omitting it
    #: buys reasoning tokens and a much slower first token without asking.
    thinking: bool
    #: Override the OpenAI-compatible endpoint (a proxy, a self-hosted gateway).
    deepseek_base_url: str

    # -- telemetry -----------------------------------------------------------
    tracing_enabled: bool
    phoenix_endpoint: str
    phoenix_project: str

    # -- audit ---------------------------------------------------------------
    audit_enabled: bool
    retention_days: int
    tenant: str
    role: str
    lawful_basis: str

    # -- cost reporting ------------------------------------------------------
    #: Daily request volume the monthly projection assumes.  A projection is
    #: only as good as its volume assumption, so the assumption is reported
    #: next to the number rather than buried in it.
    projection_requests_per_day: int
    judge_sample_rate: float
    #: Model the judge line is priced against.  Empty means the serving model --
    #: judging with a cheaper model than you serve with is the common pattern, so
    #: it is worth being able to say so.
    judge_model: str
    #: Requests the latency and quality panels average over.
    panel_window: int
    #: Push the judged scores to Phoenix as span *annotations* as well as span
    #: attributes. Annotations aggregate and filter across a project, so
    #: "every generated answer that scored under 0.7" becomes a question rather
    #: than a script. One extra HTTP call per recorded request, on the request
    #: path -- so it follows tracing rather than being on by itself.
    annotate_traces: bool
    #: Send one throwaway request at startup, before the port accepts traffic
    #: from anyone else. The first request to a hosted model pays for TLS setup
    #: and an empty prompt cache -- 10.5s TTFT against 0.8s warm, on identical
    #: settings -- and the first request after `docker compose up` is the one an
    #: audience watches. Off for tests and offline runs.
    warmup: bool

    # -- demo server ---------------------------------------------------------
    demo_host: str
    demo_port: int
    #: Shared secret required for mutating endpoints and the audit log.
    #:
    #: Empty by default: on loopback there is nothing to protect against, and a
    #: password prompt in front of a local demo is friction with no benefit. It
    #: becomes necessary the moment the port is reachable from anywhere else --
    #: which is why the server refuses to serve a non-loopback address without
    #: one unless explicitly told to.
    demo_token: str
    #: Serve a non-loopback address with no token. Off, and the server says why.
    demo_allow_insecure: bool
    #: Concurrent chat requests. Each one is a thread and a billable model call,
    #: so an unbounded endpoint is an unbounded invoice.
    demo_max_concurrent_chats: int

    # -- logging -------------------------------------------------------------
    log_level: str
    log_json: bool

    # ---- derived paths ----------------------------------------------------
    @property
    def chroma_dir(self) -> Path:
        return self.runs_dir / "chroma"

    @property
    def manifest_path(self) -> Path:
        return self.runs_dir / "index_manifest.json"

    @property
    def simple_manifest_path(self) -> Path:
        return self.runs_dir / "simple_manifest.json"

    @property
    def audit_path(self) -> Path:
        return self.runs_dir / "audit.jsonl"

    @property
    def audit_checkpoint_path(self) -> Path:
        return self.runs_dir / "audit_checkpoint.json"

    @property
    def metrics_path(self) -> Path:
        return self.runs_dir / "metrics.jsonl"

    @property
    def demo_is_loopback(self) -> bool:
        """Whether the bind address is reachable only from this machine.

        ``0.0.0.0`` is the one that catches people out: it is the default inside
        a container, and with a published port it means "the whole network".
        """
        return self.demo_host in {"127.0.0.1", "::1", "localhost"}

    @property
    def evals_dir(self) -> Path:
        return self.data_dir.parent / "evals"

    # ---- construction ------------------------------------------------------
    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = _env_path("RIGHTS_DATA_DIR", "./data")
        runs_dir = _env_path("RIGHTS_RUNS_DIR", "./runs")
        corpus = _raw("RIGHTS_CORPUS", "")
        corpus_path = (
            Path(corpus).expanduser().resolve() if corpus else data_dir / "corpus.layout.txt"
        )

        embedder = _raw("RIGHTS_EMBEDDER", "auto").lower()
        # Validated here rather than at first use: an unrecognised embedder is an
        # operator typo, and finding out about it after a 300-page parse is a
        # waste of everyone's time. The list comes from the embedding module so
        # the two cannot drift apart.
        if embedder not in EMBEDDER_CHOICES:
            raise ConfigError(
                f"RIGHTS_EMBEDDER={embedder!r} must be one of "
                f"{'|'.join(sorted(EMBEDDER_CHOICES))}"
            )

        log_level = _raw("RIGHTS_LOG_LEVEL", "INFO").upper()
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError(f"RIGHTS_LOG_LEVEL={log_level!r} is not a logging level")

        settings = cls(
            data_dir=data_dir,
            runs_dir=runs_dir,
            corpus_path=corpus_path,
            embedder=embedder,
            top_k=_env_int("RIGHTS_TOP_K", 6, low=1, high=50),
            sufficiency_threshold=_env_float("RIGHTS_SUFFICIENCY", 0.45, low=0.0, high=1.0),
            max_attempts=_env_int("RIGHTS_MAX_ATTEMPTS", 2, low=0, high=10),
            max_parent_chars=_env_int("RIGHTS_MAX_PARENT_CHARS", 4_000, low=200, high=200_000),
            context_budget_chars=_env_int("RIGHTS_CONTEXT_BUDGET", 6_000, low=500, high=400_000),
            min_block_chars=_env_int("RIGHTS_MIN_BLOCK_CHARS", 400, low=50, high=20_000),
            model=_raw("RIGHTS_MODEL", "stub-local"),
            max_answer_chars=_env_int("RIGHTS_MAX_ANSWER_CHARS", 2_000, low=200, high=40_000),
            degraded=_env_bool("RIGHTS_DEGRADED", False),
            thinking=_env_bool("RIGHTS_THINKING", False),
            deepseek_base_url=_raw("RIGHTS_DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            tracing_enabled=_env_bool("RIGHTS_TRACING", True),
            phoenix_endpoint=_raw(
                "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
            ).rstrip("/"),
            phoenix_project=_raw("PHOENIX_PROJECT_NAME", "rights-rag-agent"),
            audit_enabled=_env_bool("RIGHTS_AUDIT", True),
            retention_days=_env_int("RIGHTS_RETENTION_DAYS", 183, low=1, high=36_500),
            tenant=_raw("RIGHTS_TENANT", "default"),
            role=_raw("RIGHTS_ROLE", "reader"),
            lawful_basis=_raw("RIGHTS_LAWFUL_BASIS", "legitimate_interests"),
            projection_requests_per_day=_env_int(
                "RIGHTS_PROJECTION_RPD", 50_000, low=1, high=100_000_000
            ),
            judge_sample_rate=_env_float("RIGHTS_JUDGE_SAMPLE_RATE", JUDGE_SAMPLE_RATE, low=0.0, high=1.0),
            judge_model=_raw("RIGHTS_JUDGE_MODEL", ""),
            panel_window=_env_int("RIGHTS_PANEL_WINDOW", 20, low=1, high=100_000),
            warmup=_env_bool("RIGHTS_WARMUP", True),
            # Follows tracing: annotating spans that were never exported writes
            # feedback about nothing, and costs a failed HTTP call to find out.
            annotate_traces=_env_bool(
                "RIGHTS_ANNOTATE_TRACES", _env_bool("RIGHTS_TRACING", True)
            ),
            demo_host=_raw("RIGHTS_DEMO_HOST", "127.0.0.1"),
            demo_port=_env_int("RIGHTS_DEMO_PORT", 8000, low=1, high=65535),
            demo_token=_raw("RIGHTS_DEMO_TOKEN", ""),
            demo_allow_insecure=_env_bool("RIGHTS_DEMO_ALLOW_INSECURE", False),
            demo_max_concurrent_chats=_env_int(
                "RIGHTS_DEMO_MAX_CONCURRENT_CHATS", 4, low=1, high=64
            ),
            log_level=log_level,
            log_json=_env_bool("RIGHTS_LOG_JSON", False),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.min_block_chars >= self.context_budget_chars:
            raise ConfigError(
                f"RIGHTS_MIN_BLOCK_CHARS ({self.min_block_chars}) must be smaller than "
                f"RIGHTS_CONTEXT_BUDGET ({self.context_budget_chars})"
            )
        if self.audit_enabled and self.retention_days < RETENTION_FLOOR_DAYS:
            raise ConfigError(
                f"RIGHTS_RETENTION_DAYS={self.retention_days} is below the "
                f"{RETENTION_FLOOR_DAYS}-day floor. The AI Act's Articles 19 and 26(6) "
                "require logs to be kept for a period appropriate to the purpose and "
                "generally at least six months; shorten it only with a documented basis "
                "in other applicable law."
            )

    def with_overrides(self, **kwargs: object) -> "Settings":
        """Return a copy with fields replaced (used by the CLI and by tests)."""
        updated = replace(self, **kwargs)  # type: ignore[arg-type]
        updated.validate()
        return updated

    def ensure_runs_dir(self) -> Path:
        """Create the artefact directory, with a pointed error if we cannot.

        Chroma keeps its index in SQLite, which fails with ``disk I/O error``
        on some network and virtualised mounts.  Pointing ``RIGHTS_RUNS_DIR`` at
        local disk is the fix, and it is worth saying so before someone hits it
        during a live demo.
        """
        try:
            self.runs_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigError(
                f"cannot create RIGHTS_RUNS_DIR={self.runs_dir}: {exc}. "
                "Chroma stores its index in SQLite and needs local disk; "
                "point RIGHTS_RUNS_DIR at a local path."
            ) from exc
        return self.runs_dir


@lru_cache(maxsize=1)
def settings() -> Settings:
    """Process-wide settings, parsed once."""
    return Settings.from_env()


def reload_settings() -> Settings:
    """Drop the cache and re-read the environment (tests, and the demo's reset)."""
    settings.cache_clear()
    return settings()
