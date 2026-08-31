"""Phoenix / OpenTelemetry tracing.

Rule that shapes this whole module: **observability cannot break the app.**
Every import is optional, every export failure is swallowed, and when tracing
is off :func:`span` yields a no-op shim so that no call site anywhere has to
branch on whether tracing is available.  Killing Phoenix mid-demo must change
nothing except that spans stop appearing.

Note on naming: the ``openinference.span.kind`` values used here (CHAIN,
RETRIEVER, LLM, EVALUATOR) are what make a span render as a retriever or an LLM
call in the Phoenix UI.  They are a *different concept* from OpenTelemetry's own
``SpanKind`` (SERVER, CLIENT, INTERNAL), which describes a span's role in a
distributed call graph.  Conflating the two is a common and easily-corrected
error.

Two span sources appear in one trace, on purpose.  ``auto_instrument=True`` lets
the LangChain instrumentor emit a span per LangGraph node, named after the node
and carrying its state in and out.  This project's own spans are named
``rag.*`` and carry the OpenInference semantics -- the documents table, the
token counts, the gate decision.  The prefix is what keeps them apart in the UI:
same-named siblings from two instrumentation layers is a genuinely confusing
trace to read.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from rights_agent.log import get_logger

log = get_logger("telemetry")

#: OTel rejects attribute values that are neither primitives nor homogeneous
#: sequences of one primitive type, and a trace UI is unusable if a single
#: attribute is a megabyte of context, so non-primitives are JSON-encoded and
#: truncated.
MAX_ATTRIBUTE_CHARS = 4_096


# --------------------------------------------------------------------------- #
# Semantic conventions, with literal fallbacks
# --------------------------------------------------------------------------- #
class _Semconv:
    """OpenInference attribute names.

    Imported from ``openinference-semantic-conventions`` when installed, with
    literal-string fallbacks otherwise.  The literals are the wire format, so
    the trace renders identically either way -- the package is a convenience,
    not a requirement.
    """

    SPAN_KIND = "openinference.span.kind"
    INPUT_VALUE = "input.value"
    OUTPUT_VALUE = "output.value"
    LLM_MODEL_NAME = "llm.model_name"
    # These three name OpenInference token *counts*, not credentials.
    LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt"  # noqa: S105
    LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"  # noqa: S105
    LLM_TOKEN_COUNT_TOTAL = "llm.token_count.total"  # noqa: S105
    SESSION_ID = "session.id"
    USER_ID = "user.id"
    METADATA = "metadata"
    RETRIEVAL_DOCUMENTS = "retrieval.documents"
    DOCUMENT_ID = "document.id"
    DOCUMENT_SCORE = "document.score"
    DOCUMENT_CONTENT = "document.content"
    DOCUMENT_METADATA = "document.metadata"

    def __init__(self) -> None:
        try:
            from openinference.semconv.trace import (
                DocumentAttributes,
                SpanAttributes,
            )
        except ImportError:  # pragma: no cover - fallbacks are the point
            log.debug("openinference-semantic-conventions not installed; using literals")
            return
        self.SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
        self.INPUT_VALUE = SpanAttributes.INPUT_VALUE
        self.OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
        self.LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
        self.LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
        self.LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
        self.LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
        self.SESSION_ID = SpanAttributes.SESSION_ID
        self.USER_ID = SpanAttributes.USER_ID
        self.METADATA = SpanAttributes.METADATA
        self.RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS
        self.DOCUMENT_ID = DocumentAttributes.DOCUMENT_ID
        self.DOCUMENT_SCORE = DocumentAttributes.DOCUMENT_SCORE
        self.DOCUMENT_CONTENT = DocumentAttributes.DOCUMENT_CONTENT
        self.DOCUMENT_METADATA = DocumentAttributes.DOCUMENT_METADATA


SEMCONV = _Semconv()

#: ``openinference.span.kind`` values -- NOT OpenTelemetry SpanKind.
CHAIN = "CHAIN"
RETRIEVER = "RETRIEVER"
LLM = "LLM"
EVALUATOR = "EVALUATOR"
AGENT = "AGENT"


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
@dataclass
class TelemetryState:
    enabled: bool = False
    status: str = "not initialised"
    endpoint: str = ""
    project: str = ""


_state = TelemetryState()
_tracer: Any = None
_lock = threading.Lock()


def telemetry_status() -> TelemetryState:
    """Current tracing state, for the dashboard and the CLI banner."""
    return _state


def init_telemetry(
    project_name: str, endpoint: str, *, enabled: bool = True, batch: bool = True
) -> bool:
    """Bootstrap tracing.  Never raises.

    Returns whether spans will be exported.  A blanket ``except`` here is
    deliberate: a missing package, a bad endpoint, a version mismatch inside
    Phoenix -- none of them are reasons for the agent to stop answering.
    """
    global _tracer
    with _lock:
        if not enabled:
            _state.enabled = False
            _state.status = "disabled (RIGHTS_TRACING=false)"
            return False
        if _tracer is not None:
            return _state.enabled
        try:
            from phoenix.otel import register

            provider = register(
                project_name=project_name,
                endpoint=f"{endpoint}/v1/traces",
                auto_instrument=True,
                batch=batch,
                verbose=False,
                set_global_tracer_provider=True,
            )
            _tracer = provider.get_tracer("rights_agent")
            _state.enabled = True
            _state.endpoint = endpoint
            _state.project = project_name
            _state.status = f"exporting to {endpoint} (project {project_name})"
            log.info("tracing %s", _state.status)
        except Exception as exc:  # noqa: BLE001 - deliberate blanket catch
            _tracer = None
            _state.enabled = False
            _state.status = f"disabled ({type(exc).__name__}: {exc})"
            log.warning("tracing disabled: %s", exc)
        return _state.enabled


def flush_spans(timeout_millis: int = 3_000) -> None:
    """Export anything still batched. Never raises.

    Needed before annotating a span: the batch processor holds spans for seconds
    by default, and Phoenix rejects feedback for a span it has not received --
    ``404 Spans with IDs ... do not exist``, which reads like a bug in the id.
    """
    try:
        from opentelemetry import trace as otel_trace

        flush = getattr(otel_trace.get_tracer_provider(), "force_flush", None)
        if flush is not None:
            flush(timeout_millis)
    except Exception as exc:  # noqa: BLE001
        log.debug("span flush failed: %s", exc)


def shutdown_telemetry(timeout_millis: int = 3_000) -> None:
    """Flush pending spans on the way out.  Never raises."""
    try:
        from opentelemetry import trace as otel_trace

        provider = otel_trace.get_tracer_provider()
        flush = getattr(provider, "force_flush", None)
        if flush is not None:
            flush(timeout_millis)
    except Exception as exc:  # noqa: BLE001
        log.debug("span flush failed: %s", exc)


# --------------------------------------------------------------------------- #
# Attribute coercion
# --------------------------------------------------------------------------- #
def coerce_attribute(value: Any) -> Any:
    """Make ``value`` safe to set as an OTel attribute."""
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = list(value)
        if items and all(isinstance(item, str) for item in items):
            return [_truncate(item, MAX_ATTRIBUTE_CHARS // 4) for item in items]
        if items and all(isinstance(item, bool) for item in items):
            return items
        if items and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in items):
            return items
        return _truncate(json.dumps(items, default=str, ensure_ascii=False))
    if isinstance(value, Mapping):
        return _truncate(json.dumps(dict(value), default=str, ensure_ascii=False, sort_keys=True))
    return _truncate(str(value))


def _truncate(text: str, limit: int = MAX_ATTRIBUTE_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


# --------------------------------------------------------------------------- #
# Spans
# --------------------------------------------------------------------------- #
class SpanShim:
    """Uniform span surface whether or not tracing is on.

    The no-op variant means retrieval, generation and scoring code never has an
    ``if tracing_enabled`` branch -- the single most common way instrumentation
    ends up drifting out of sync with the code it describes.
    """

    __slots__ = ("_span",)

    def __init__(self, span: Any = None) -> None:
        self._span = span

    @property
    def recording(self) -> bool:
        return self._span is not None

    def set_attribute(self, key: str, value: Any) -> None:
        if self._span is None:
            return
        try:
            self._span.set_attribute(key, coerce_attribute(value))
        except Exception as exc:  # noqa: BLE001
            log.debug("could not set span attribute %s: %s", key, exc)

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        for key, value in attributes.items():
            if value is not None:
                self.set_attribute(key, value)

    def set_input(self, value: Any) -> None:
        self.set_attribute(SEMCONV.INPUT_VALUE, value)

    def set_output(self, value: Any) -> None:
        self.set_attribute(SEMCONV.OUTPUT_VALUE, value)

    def record_exception(self, exc: BaseException) -> None:
        if self._span is None:
            return
        try:
            from opentelemetry.trace import Status, StatusCode

            self._span.record_exception(exc)
            self._span.set_status(Status(StatusCode.ERROR, str(exc)))
        except Exception as inner:  # noqa: BLE001
            log.debug("could not record exception on span: %s", inner)

    def span_id(self) -> str:
        """Hex span id, so a metrics row can link to its trace."""
        if self._span is None:
            return ""
        try:
            return format(self._span.get_span_context().span_id, "016x")
        except Exception:  # noqa: BLE001
            return ""

    def trace_id(self) -> str:
        if self._span is None:
            return ""
        try:
            return format(self._span.get_span_context().trace_id, "032x")
        except Exception:  # noqa: BLE001
            return ""


_NOOP = SpanShim(None)


@contextmanager
def span(name: str, kind: str = CHAIN, **attributes: Any) -> Iterator[SpanShim]:
    """Open a span named ``name`` with an OpenInference kind.

    Yields a :class:`SpanShim`; when tracing is off, the shim is a no-op and the
    block runs exactly as it otherwise would.
    """
    if _tracer is None:
        yield _NOOP
        return
    try:
        context = _tracer.start_as_current_span(name)
    except Exception as exc:  # noqa: BLE001
        log.debug("could not start span %s: %s", name, exc)
        yield _NOOP
        return

    with context as raw:
        shim = SpanShim(raw)
        shim.set_attribute(SEMCONV.SPAN_KIND, kind)
        shim.set_attributes(attributes)
        try:
            yield shim
        except Exception as exc:
            shim.record_exception(exc)
            raise


def current_span() -> SpanShim:
    """The active span as a shim, or a no-op."""
    if _tracer is None:
        return _NOOP
    try:
        from opentelemetry import trace as otel_trace

        raw = otel_trace.get_current_span()
        return SpanShim(raw) if raw is not None else _NOOP
    except Exception:  # noqa: BLE001
        return _NOOP


def set_retrieval_documents(shim: SpanShim, documents: Sequence[Mapping[str, Any]]) -> None:
    """Write the flattened document list that renders as a documents table.

    Each entry needs ``id``, ``score``, ``content`` and optionally ``metadata``.
    """
    if not shim.recording:
        return
    base = SEMCONV.RETRIEVAL_DOCUMENTS
    for index, document in enumerate(documents):
        shim.set_attribute(f"{base}.{index}.{SEMCONV.DOCUMENT_ID}", document.get("id", ""))
        shim.set_attribute(
            f"{base}.{index}.{SEMCONV.DOCUMENT_SCORE}", float(document.get("score", 0.0))
        )
        shim.set_attribute(
            f"{base}.{index}.{SEMCONV.DOCUMENT_CONTENT}", document.get("content", "")
        )
        metadata = document.get("metadata")
        if metadata:
            shim.set_attribute(f"{base}.{index}.{SEMCONV.DOCUMENT_METADATA}", metadata)
