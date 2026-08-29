"""One request, end to end: root span, unique thread, one metrics row.

The invocation trap this exists to avoid: **a unique ``thread_id`` per request,
and every state field initialised.**  Reusing one ``thread_id`` across questions
makes the checkpointer replay prior state -- ``attempts`` accumulates, the
previous rewrite leaks into the next question, and everything refuses after the
third query.  It is invisible in logs and instantly obvious in a trace, which
makes it a great thing to demonstrate deliberately and a miserable thing to hit
by accident.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from rights_agent.audit import AuditError, AuditLog, fingerprint, redact
from rights_agent.config import PARSER_VERSION, PROMPT_VERSION, Settings, settings as load_settings
from rights_agent.conversation import Contextualisation, ConversationStore, Turn, contextualise
from rights_agent.costs import breakdown as cost_breakdown_model, trace_bytes_for
from rights_agent.annotations import annotate_span_later, judge_annotations
from rights_agent.llm import stream_tokens_to
from rights_agent.graph import AgentDeps, AgentState, build_graph, initial_state
from rights_agent.log import bind_request, get_logger
from rights_agent.metrics import MetricsSink, RequestMetrics, sink_for
from rights_agent.store import now_iso
from rights_agent.telemetry import CHAIN, SEMCONV, init_telemetry, span

log = get_logger("agent")


@dataclass(frozen=True, slots=True)
class AgentAnswer:
    """Everything a caller needs, and nothing it has to reach into state for."""

    request_id: str
    session_id: str
    question: str
    answer: str
    citations: list[str]
    refused: bool
    sufficiency: float
    intent: str
    route: str
    attempts: int
    docs: list[dict[str, Any]]
    context: str
    scores: dict[str, float]
    stage_ms: dict[str, float]
    llm_stats: dict[str, float]
    metrics: RequestMetrics
    contextualisation: Contextualisation | None = None
    error: str = ""

    @property
    def cost_usd(self) -> float:
        return self.metrics.cost_usd

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form for the dashboard API."""
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "refused": self.refused,
            "sufficiency": round(self.sufficiency, 4),
            "intent": self.intent,
            "route": self.route,
            "attempts": self.attempts,
            "documents": [
                {
                    "citation": doc.get("citation", ""),
                    "breadcrumb": doc.get("breadcrumb", ""),
                    "score": round(float(doc.get("score", 0.0)), 4),
                    "expanded": bool(doc.get("expanded", False)),
                    "chars": len(str(doc.get("text", ""))),
                }
                for doc in self.docs
            ],
            "scores": self.scores,
            "stage_ms": self.stage_ms,
            "llm_stats": self.llm_stats,
            "ttft_ms": self.metrics.ttft_ms,
            "itl_ms_mean": self.metrics.itl_ms_mean,
            "e2e_ms": self.metrics.e2e_ms,
            "non_generation_ms": self.metrics.non_generation_ms(),
            "orchestration_ms": self.metrics.orchestration_ms(),
            "stage_total_ms": self.metrics.stage_total_ms(),
            "formula_gap_ms": self.metrics.formula_gap_ms(),
            "prompt_tokens": self.metrics.prompt_tokens,
            "completion_tokens": self.metrics.completion_tokens,
            "cached_tokens": self.metrics.cached_tokens,
            "cost_usd": self.metrics.cost_usd,
            "cost_breakdown": self.metrics.cost_breakdown,
            "cost_components": self.metrics.cost_components,
            "cost_total_usd": self.metrics.cost_total_usd,
            "trace_bytes": self.metrics.trace_bytes,
            "history_used": self.metrics.history_used,
            "contextualisation": self.contextualisation.to_dict()
            if self.contextualisation
            else None,
            "audit_sequence": self.metrics.audit_sequence,
            "audit_record_hash": self.metrics.audit_record_hash,
            "model": self.metrics.model,
            "requested_model": self.metrics.requested_model,
            "fallback": self.metrics.fallback,
            "index_version": self.metrics.index_version,
            "degraded": self.metrics.degraded,
            "trace_id": self.metrics.trace_id,
            "error": self.error,
        }


class Agent:
    """A compiled graph, a metrics sink, and the request bookkeeping around them."""

    def __init__(
        self,
        settings: Settings | None = None,
        deps: AgentDeps | None = None,
        *,
        degraded: bool | None = None,
        sink: MetricsSink | None = None,
        audit: AuditLog | None = None,
        conversations: ConversationStore | None = None,
        init_tracing: bool = True,
    ) -> None:
        self.settings = settings or load_settings()
        if init_tracing:
            init_telemetry(
                self.settings.phoenix_project,
                self.settings.phoenix_endpoint,
                enabled=self.settings.tracing_enabled,
            )
        self.deps = deps or AgentDeps(settings=self.settings, degraded=degraded)
        self.graph = build_graph(self.deps)
        self.sink = sink or sink_for(self.settings)
        self.conversations = conversations or ConversationStore()
        self.audit = audit if audit is not None else AuditLog(self.settings.audit_path)
        self._lock = threading.Lock()

    # ---- properties -------------------------------------------------------
    @property
    def index_version(self) -> str:
        return self.deps.index_version

    @property
    def degraded(self) -> bool:
        return bool(self.deps.degraded or self.settings.degraded)

    def set_degraded(self, degraded: bool) -> None:
        """Flip the degraded-fallback simulation between requests."""
        with self._lock:
            self.deps.degraded = degraded
            self.deps.client = None

    # ---- the request ------------------------------------------------------
    def ask(
        self,
        question: str,
        *,
        session_id: str | None = None,
        user_id: str = "",
        tenant: str | None = None,
        record: bool = True,
        use_history: bool = True,
        remember: bool = True,
        on_token: Callable[[str], None] | None = None,
    ) -> AgentAnswer:
        """Run one question through the graph and record exactly one row."""
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        request_id = uuid.uuid4().hex[:12]
        session_id = session_id or f"sess-{uuid.uuid4().hex[:8]}"
        tenant = tenant or self.settings.tenant
        started = time.perf_counter()

        # Resolve a follow-up against the session's earlier questions. The
        # *original* question is still what the sufficiency gate scores: a
        # rewrite that retrieves beautifully for itself can still fail the user.
        resolved = (
            contextualise(question, self.conversations.questions(session_id))
            if use_history
            else Contextualisation(query=question, used_history=False, reason="history disabled")
        )

        with bind_request(request_id):
            with span(
                "rag-agent",
                CHAIN,
                **{
                    SEMCONV.INPUT_VALUE: question,
                    SEMCONV.SESSION_ID: session_id,
                    SEMCONV.USER_ID: user_id or "anonymous",
                    "metadata.index_version": self.index_version,
                    "metadata.prompt_version": PROMPT_VERSION,
                    "metadata.request_id": request_id,
                    "metadata.tenant": tenant,
                    "metadata.degraded": self.degraded,
                },
            ) as root:
                root.set_attributes(
                    {
                        "conversation.history_used": resolved.used_history,
                        "conversation.borrowed_terms": list(resolved.borrowed),
                        "conversation.turns": len(self.conversations.history(session_id)),
                    }
                )
                state: AgentState = initial_state(
                    question,
                    session_id=session_id,
                    user_id=user_id,
                    scored_question=resolved.query,
                )
                state["rewritten_query"] = resolved.query
                error = ""
                try:
                    with stream_tokens_to(on_token):
                        final = self.graph.invoke(
                            state,
                            config={
                                # Unique per request. See the module docstring.
                                "configurable": {"thread_id": f"{session_id}:{request_id}"},
                                "run_name": "rag-agent",
                                "metadata": {
                                    "session_id": session_id,
                                    "request_id": request_id,
                                    "prompt_version": PROMPT_VERSION,
                                    "index_version": self.index_version,
                                },
                            },
                        )
                except Exception as exc:  # noqa: BLE001 - the row must still be written
                    error = f"{type(exc).__name__}: {exc}"
                    log.exception("graph invocation failed")
                    root.record_exception(exc)
                    final = dict(state)

                e2e_ms = round((time.perf_counter() - started) * 1_000, 3)
                answer = self._build_answer(
                    request_id=request_id,
                    session_id=session_id,
                    user_id=user_id,
                    tenant=tenant,
                    question=question,
                    final=final,
                    e2e_ms=e2e_ms,
                    error=error,
                    root=root,
                    resolved=resolved,
                )
                root.set_output(answer.answer)
                root.set_attributes(
                    {
                        "gate.sufficiency": answer.sufficiency,
                        "gate.refused": answer.refused,
                        "llm.cost.total_usd": answer.metrics.cost_usd,
                        "latency.e2e_ms": e2e_ms,
                        "latency.non_generation_ms": answer.metrics.non_generation_ms(),
                        "latency.orchestration_ms": answer.metrics.orchestration_ms(),
                        "latency.formula_gap_ms": answer.metrics.formula_gap_ms(),
                    }
                )

        if record:
            self.sink.append(answer.metrics)
            self._audit(answer)
            self._annotate(answer)
        if remember:
            # Synthetic load is not a conversation. Remembering it would fill a
            # bounded transcript store with generated traffic and evict the
            # conversation a person is actually having.
            self.conversations.add(
                session_id, Turn(role="user", content=question, request_id=request_id)
            )
            self.conversations.add(
                session_id,
                Turn(
                    role="agent",
                    content=answer.answer,
                    request_id=request_id,
                    citations=tuple(answer.citations),
                    refused=answer.refused,
                ),
            )
        return answer

    # ---- the audit record -------------------------------------------------
    def _annotate(self, answer: "AgentAnswer") -> None:
        """Attach the judged scores to the request's own trace, in Phoenix.

        The scores are already span *attributes*, which is enough to read one
        request. Annotations aggregate across a project and can be filtered, so
        "show me every generated answer that scored under 0.7 for groundedness"
        becomes a question you can ask rather than a script you have to write.

        Off by default in tests and whenever tracing is off; never fatal.
        """
        if not self.settings.annotate_traces or not answer.metrics.trace_span_id:
            return
        if not answer.scores:
            return
        annotate_span_later(
            answer.metrics.trace_span_id,
            judge_annotations(
                answer.scores,
                judge=getattr(self.deps.judge, "name", "heuristic-lexical"),
                sufficiency=answer.sufficiency,
                route=answer.metrics.route,
                index_version=answer.metrics.index_version,
            ),
            self.settings,
        )

    def _audit(self, answer: AgentAnswer) -> None:
        """Append one hash-chained record.  Never loses the user's answer.

        Sources are recorded as id, citation, version and hash -- never as a
        copy of the text. An id and a hash prove a source *changed*; the version
        is what lets you reconstruct what it said.
        """
        if not self.settings.audit_enabled:
            return
        metrics = answer.metrics
        manifest = self.deps.retriever.manifest if self.deps.retriever else None
        sources = [
            {
                "id": str(doc.get("id", "")),
                "citation": str(doc.get("citation", "")),
                "version": manifest.index_version if manifest else metrics.index_version,
                "sha256": fingerprint(str(doc.get("text", ""))),
            }
            for doc in answer.docs
        ]
        try:
            record = self.audit.append(
                ts=metrics.ts,
                request_id=metrics.request_id,
                session_id=metrics.session_id,
                actor=metrics.user_id or "anonymous",
                tenant=metrics.tenant,
                role=self.settings.role,
                lawful_basis=self.settings.lawful_basis,
                # Redacted at capture, with a fingerprint of the original: a
                # redaction applied on read has already been exported by anyone
                # with an API key.
                question=redact(metrics.question),
                question_sha256=fingerprint(metrics.question),
                answered=not answer.refused and not metrics.error,
                refused=answer.refused,
                refusal_reason=(
                    f"sufficiency {metrics.sufficiency:.3f} < threshold "
                    f"{self.settings.sufficiency_threshold:.2f}"
                    if answer.refused
                    else ""
                ),
                citations=list(answer.citations),
                index_version=metrics.index_version,
                embedding_model=metrics.embedding_model,
                parser_version=PARSER_VERSION,
                prompt_version=metrics.prompt_version,
                model=metrics.model,
                requested_model=metrics.requested_model,
                fallback=metrics.fallback,
                intent=metrics.intent,
                route=metrics.route,
                sufficiency=metrics.sufficiency,
                sufficiency_threshold=self.settings.sufficiency_threshold,
                attempts=metrics.attempts,
                sources=sources,
                prompt_tokens=metrics.prompt_tokens,
                completion_tokens=metrics.completion_tokens,
                cost_usd=metrics.cost_usd,
                e2e_ms=metrics.e2e_ms,
                degraded=metrics.degraded,
                error=metrics.error,
                trace_id=metrics.trace_id,
                trace_span_id=metrics.trace_span_id,
            )
        except AuditError as exc:
            # A missing audit record is a compliance event and is logged as an
            # error -- but it must not take the answer away from the user.
            log.error("audit record not written for %s: %s", metrics.request_id, exc)
            return
        metrics.audit_sequence = record.sequence
        metrics.audit_record_hash = record.record_hash

    def _build_answer(
        self,
        *,
        request_id: str,
        session_id: str,
        user_id: str,
        tenant: str,
        question: str,
        final: dict[str, Any],
        e2e_ms: float,
        error: str,
        root: Any,
        resolved: Contextualisation,
    ) -> AgentAnswer:
        llm_stats = dict(final.get("llm_stats") or {})
        docs = list(final.get("docs") or [])
        cost_breakdown = {
            key.removeprefix("cost_"): value
            for key, value in llm_stats.items()
            if key.startswith("cost_") and key != "cost_usd"
        }
        context = str(final.get("context") or "")
        answer_text = str(final.get("answer") or "")
        trace_bytes = trace_bytes_for(len(context), len(answer_text), len(question))
        # The model that actually served, never the configured one. Pricing the
        # components against a model that did not answer -- while ``cost_usd``
        # priced the one that did -- gave two figures for the same request that
        # disagreed about which model they were about.
        served_model = str(final.get("model") or "")
        requested_model = str(final.get("requested_model") or self.settings.model)
        components = cost_breakdown_model(
            model=served_model or requested_model,
            prompt_tokens=int(llm_stats.get("prompt_tokens", 0)),
            completion_tokens=int(llm_stats.get("completion_tokens", 0)),
            cached_tokens=int(llm_stats.get("cached_tokens", 0)),
            trace_bytes=trace_bytes,
            judge_sample_rate=self.settings.judge_sample_rate,
            judge_model=self.settings.judge_model,
            judge_incurred=False,
        )
        metrics = RequestMetrics(
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
            tenant=tenant,
            ts=now_iso(),
            question=question,
            rewritten_query=str(final.get("rewritten_query") or question),
            route=str(final.get("route") or ""),
            intent=str(final.get("intent") or ""),
            ttft_ms=float(llm_stats.get("ttft_ms", 0.0)),
            itl_ms_mean=float(llm_stats.get("itl_ms_mean", 0.0)),
            itl_ms_p95=float(llm_stats.get("itl_ms_p95", 0.0)),
            e2e_ms=e2e_ms,
            stage_ms=dict(final.get("stage_ms") or {}),
            index_version=str(final.get("index_version") or self.index_version),
            embedding_model=self.deps.retriever.embedder_name if self.deps.retriever else "",
            retrieved_ids=[str(doc.get("id", "")) for doc in docs],
            retrieval_scores=[round(float(doc.get("score", 0.0)), 6) for doc in docs],
            citations=list(final.get("citations") or []),
            attempts=int(final.get("attempts", 0)),
            sufficiency=round(float(final.get("sufficiency", 0.0)), 6),
            refused=bool(final.get("refused", False)),
            model=served_model,
            requested_model=requested_model,
            fallback=bool(served_model) and served_model != requested_model,
            prompt_version=PROMPT_VERSION,
            prompt_tokens=int(llm_stats.get("prompt_tokens", 0)),
            completion_tokens=int(llm_stats.get("completion_tokens", 0)),
            cached_tokens=int(llm_stats.get("cached_tokens", 0)),
            cost_usd=round(float(llm_stats.get("cost_usd", 0.0)), 8),
            cost_breakdown=cost_breakdown,
            cost_components=components.components,
            cost_total_usd=components.total_usd,
            trace_bytes=trace_bytes,
            history_used=resolved.used_history,
            scores=dict(final.get("scores") or {}),
            trace_span_id=root.span_id() if hasattr(root, "span_id") else "",
            trace_id=root.trace_id() if hasattr(root, "trace_id") else "",
            degraded=self.degraded,
            error=error or str(final.get("error") or ""),
        )
        return AgentAnswer(
            contextualisation=resolved,
            request_id=request_id,
            session_id=session_id,
            question=question,
            answer=str(final.get("answer") or ""),
            citations=list(final.get("citations") or []),
            refused=bool(final.get("refused", False)),
            sufficiency=float(final.get("sufficiency", 0.0)),
            intent=str(final.get("intent") or ""),
            route=str(final.get("route") or ""),
            attempts=int(final.get("attempts", 0)),
            docs=docs,
            context=str(final.get("context") or ""),
            scores=dict(final.get("scores") or {}),
            stage_ms=dict(final.get("stage_ms") or {}),
            llm_stats=llm_stats,
            metrics=metrics,
            error=metrics.error,
        )
