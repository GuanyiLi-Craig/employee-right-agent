"""The LangGraph workflow.

::

    classify → retrieve → assess ─┬─ sufficient ──────────→ generate → score → END
                  ↑               ├─ thin, attempts left ─→ refine ──┘
                  └───────────────┘
                                  └─ thin, none left ─────→ refuse → END

Node boundaries are span boundaries, so the trace shape falls out of the design
instead of being hand-instrumented.

**Most of these nodes make no model call, on purpose.** Intent classification is
keyword matching; the sufficiency gate is arithmetic; refinement borrows
vocabulary from the corpus itself.  Putting a model in every box is the standard
way to make an agent expensive and non-deterministic for no measurable gain.
The two places a model earns its cost are generation and (optionally, on a
sample) judging.

State holds **plain types only**.  The checkpointer serialises it, and a
checkpoint you cannot read from another process is a checkpoint you cannot
debug.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any, Iterator, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from rights_agent.config import Settings, settings as load_settings
from rights_agent.judges import HeuristicJudge, Judge
from rights_agent.llm import LLMClient, generate
from rights_agent.log import get_logger
from rights_agent.retrieval import Doc, Retriever, format_context, refine_query, sufficiency
from rights_agent.telemetry import CHAIN, SEMCONV, span

log = get_logger("graph")

#: Keyword → intent.  First match in this order wins; the order matters because
#: "dismissal for refusing a variation" is a dismissal question, not a pay one.
INTENT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dismissal", ("dismiss", "unfair dismissal", "redundan", "reinstate", "fire and rehire")),
    ("leave", ("leave", "bereave", "patern", "matern", "parental", "sick pay", "sickness")),
    ("hours", ("hours", "shift", "zero hours", "flexible working", "rota", "working time")),
    ("pay", ("pay", "wage", "wages", "deduction", "tips", "gratuit", "minimum wage", "holiday")),
    ("harassment", ("harass", "equality", "discriminat", "disclosure", "whistle")),
    ("unions", ("union", "ballot", "industrial action", "blacklist", "collective bargaining")),
    ("enforcement", ("enforce", "penalt", "notice", "agency", "tribunal", "offence", "inspect")),
)
DEFAULT_INTENT = "general"


def classify_intent(question: str) -> str:
    """Keyword-based intent label.

    Deliberately not a model call: an LLM here would cost a round trip and a
    token bill to produce a label that a lookup table gets right, and would make
    the label non-deterministic between runs of the same eval.
    """
    lowered = question.lower()
    for intent, keywords in INTENT_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return intent
    return DEFAULT_INTENT


def _merge_ms(left: dict[str, float] | None, right: dict[str, float] | None) -> dict[str, float]:
    """Reducer that *sums* stage durations.

    ``retrieve`` runs again after a refinement, and the honest total for "time
    spent retrieving" is the sum of both passes, not whichever happened last.
    """
    merged = dict(left or {})
    for key, value in (right or {}).items():
        merged[key] = round(merged.get(key, 0.0) + float(value), 3)
    return merged


class AgentState(TypedDict, total=False):
    """Plain types only -- this gets serialised by the checkpointer."""

    question: str
    session_id: str
    user_id: str
    rewritten_query: str
    #: The question the sufficiency gate scores.  Equal to ``question`` unless a
    #: conversational follow-up was resolved against the user's own earlier
    #: turn -- see the note on :func:`_node_retrieve`.
    scored_question: str
    intent: str
    route: str
    docs: list[dict[str, Any]]
    context: str
    answer: str
    citations: list[str]
    attempts: int
    sufficiency: float
    refused: bool
    scores: dict[str, float]
    stage_ms: Annotated[dict[str, float], _merge_ms]
    llm_stats: dict[str, float]
    index_version: str
    #: The model that actually served this request, as the client reported it --
    #: empty when no model was called (a refusal never reaches ``generate``).
    #:
    #: Declared here because LangGraph **silently drops updates for keys the
    #: state schema does not declare**. Undeclared, this field looked like it was
    #: being written and was not, and every downstream consumer fell back to the
    #: *configured* model -- which is the one question the field exists to answer.
    model: str
    #: What was asked for. Differs from ``model`` after a fallback, and that
    #: difference is the whole point of recording both.
    requested_model: str
    error: str


def initial_state(
    question: str,
    *,
    session_id: str = "",
    user_id: str = "",
    scored_question: str | None = None,
) -> AgentState:
    """Every field initialised explicitly (§12.5).

    Leaving fields out and relying on ``total=False`` means a node reads
    ``state.get("attempts")`` as ``None`` on the first pass and as an int on the
    second -- a class of bug that only shows up under refinement.
    """
    return {
        "question": question,
        "session_id": session_id,
        "user_id": user_id,
        "rewritten_query": question,
        "scored_question": scored_question or question,
        "intent": "",
        "route": "",
        "docs": [],
        "context": "",
        "answer": "",
        "citations": [],
        "attempts": 0,
        "sufficiency": 0.0,
        "refused": False,
        "scores": {},
        "stage_ms": {},
        "llm_stats": {},
        "index_version": "",
        "model": "",
        "requested_model": "",
        "error": "",
    }


@dataclass
class AgentDeps:
    """Injected collaborators.

    Passed in rather than imported at call time so a test can substitute a
    retriever or a client without monkeypatching module globals -- and so the
    dashboard and the eval suite share one construction path.
    """

    settings: Settings = field(default_factory=load_settings)
    retriever: Retriever | None = None
    judge: Judge | None = None
    client: LLMClient | None = None
    degraded: bool | None = None
    score_inline: bool = True

    def __post_init__(self) -> None:
        if self.retriever is None:
            self.retriever = Retriever(self.settings)
        if self.judge is None:
            self.judge = HeuristicJudge()

    @property
    def index_version(self) -> str:
        assert self.retriever is not None
        return self.retriever.index_version


@contextmanager
def _timed(state_key: str, updates: dict[str, Any]) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        updates["stage_ms"] = {state_key: round((time.perf_counter() - started) * 1_000, 3)}


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def _node_classify(deps: AgentDeps):
    def classify(state: AgentState) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        with _timed("classify", updates), span("rag.classify", CHAIN) as current:
            intent = classify_intent(state["question"])
            current.set_input(state["question"])
            current.set_output(intent)
            updates["intent"] = intent
            updates["index_version"] = deps.index_version
            # On every path, so a refusal still records what was configured even
            # though no model was called.
            updates["requested_model"] = deps.settings.model
        return updates

    return classify


def _node_retrieve(deps: AgentDeps):
    def retrieve(state: AgentState) -> dict[str, Any]:
        assert deps.retriever is not None
        updates: dict[str, Any] = {}
        with _timed("retrieve", updates):
            query = state.get("rewritten_query") or state["question"]
            docs = deps.retriever.search(query, k=deps.settings.top_k)
            context = format_context(docs)
            # Sufficiency is never scored against a rewrite this system invented:
            # a refined query retrieves beautifully for itself and still fails
            # the user, so refinement must not be able to talk its way past the
            # gate.
            #
            # ``scored_question`` exists for the one case that is not that: a
            # conversational follow-up ("how long is it?") resolved against the
            # user's *own* previous turn. Those words came from the user, one
            # turn earlier, so they are part of the question as asked -- and
            # scoring the fragment alone would refuse every follow-up. The
            # distinction is who supplied the words, and ``refine`` never
            # touches this field.
            score = sufficiency(docs, state.get("scored_question") or state["question"])
            updates.update(
                docs=[doc.to_dict() for doc in docs],
                context=context,
                sufficiency=score,
            )
        return updates

    return retrieve


def _node_assess(deps: AgentDeps):
    def assess(state: AgentState) -> dict[str, Any]:
        """Record the gate decision as a span.  No model call: it is arithmetic."""
        updates: dict[str, Any] = {}
        with _timed("assess", updates), span("rag.assess", CHAIN) as current:
            score = float(state.get("sufficiency", 0.0))
            threshold = deps.settings.sufficiency_threshold
            attempts = int(state.get("attempts", 0))
            route = _route(score, threshold, attempts, deps.settings.max_attempts)
            current.set_attributes(
                {
                    SEMCONV.INPUT_VALUE: state["question"],
                    SEMCONV.OUTPUT_VALUE: route,
                    "gate.sufficiency": score,
                    "gate.threshold": threshold,
                    "gate.attempts": attempts,
                    "gate.max_attempts": deps.settings.max_attempts,
                    "gate.documents": len(state.get("docs") or []),
                    "gate.scored_question": state.get("scored_question") or state["question"],
                }
            )
            updates["route"] = route
        return updates

    return assess


def _route(score: float, threshold: float, attempts: int, max_attempts: int) -> str:
    if score >= threshold:
        return "generate"
    if attempts < max_attempts:
        return "refine"
    return "refuse"


def route_after_assess(state: AgentState) -> str:
    """Conditional edge: read the decision ``assess`` already recorded."""
    return state.get("route") or "refuse"


def _node_refine(deps: AgentDeps):
    def refine(state: AgentState) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        with _timed("refine", updates), span("rag.refine", CHAIN) as current:
            docs = [Doc.from_dict(payload) for payload in state.get("docs") or []]
            rewritten = refine_query(
                state["question"], docs, previous=state.get("rewritten_query", "")
            )
            current.set_input(state.get("rewritten_query", ""))
            current.set_output(rewritten)
            updates["rewritten_query"] = rewritten
            updates["attempts"] = int(state.get("attempts", 0)) + 1
        return updates

    return refine


def _node_generate(deps: AgentDeps):
    def generate_node(state: AgentState) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        with _timed("generate", updates):
            result = generate(
                state["question"],
                state.get("context", ""),
                deps.settings,
                client=deps.client,
                degraded=deps.degraded,
            )
            total_cost, breakdown = result.cost
            updates.update(
                answer=result.text,
                citations=list(result.citations),
                error=result.error,
                llm_stats={
                    "ttft_ms": result.ttft_ms,
                    "itl_ms_mean": result.itl_ms_mean,
                    "itl_ms_p95": result.itl_ms_p95,
                    "generation_ms": result.generation_ms,
                    "prompt_tokens": float(result.prompt_tokens),
                    "completion_tokens": float(result.completion_tokens),
                    "cached_tokens": float(result.cached_tokens),
                    "cost_usd": total_cost,
                    **{f"cost_{key}": value for key, value in breakdown.items()},
                },
            )
            # Both, always: a fallback is only visible if you record what was
            # asked for as well as what answered.
            updates["model"] = result.model
            updates["requested_model"] = deps.settings.model
        return updates

    return generate_node


def _node_refuse(deps: AgentDeps):
    def refuse(state: AgentState) -> dict[str, Any]:
        """Decline, stating the score and the threshold.

        A refusal that does not say *why* is indistinguishable from a bug, both
        to the user and to whoever reads the eval failure.
        """
        updates: dict[str, Any] = {}
        with _timed("refuse", updates), span("rag.refuse", CHAIN) as current:
            score = float(state.get("sufficiency", 0.0))
            threshold = deps.settings.sufficiency_threshold
            answer = (
                "I cannot answer this from the indexed document. Retrieval sufficiency "
                f"scored {score:.2f} against a threshold of {threshold:.2f}, after "
                f"{int(state.get('attempts', 0))} refinement attempt(s), so any answer "
                "would not be supported by the source."
            )
            current.set_attributes(
                {
                    SEMCONV.INPUT_VALUE: state["question"],
                    SEMCONV.OUTPUT_VALUE: answer,
                    "gate.sufficiency": score,
                    "gate.threshold": threshold,
                }
            )
            updates.update(answer=answer, refused=True, citations=[])
        return updates

    return refuse


def _node_score(deps: AgentDeps):
    def score_node(state: AgentState) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        with _timed("score", updates):
            if not deps.score_inline or deps.judge is None:
                updates["scores"] = {}
                return updates
            scores = deps.judge.score(
                state["question"],
                state.get("context", ""),
                state.get("answer", ""),
                state.get("citations", []),
            )
            updates["scores"] = scores.to_dict()
        return updates

    return score_node


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build_graph(deps: AgentDeps | None = None, *, checkpointer: Any | None = None):
    """Compile the workflow.

    ``MemorySaver`` by default: the checkpoint is what makes a trace replayable
    within a process, and nothing here needs it to outlive one.
    """
    deps = deps or AgentDeps()
    builder = StateGraph(AgentState)
    builder.add_node("classify", _node_classify(deps))
    builder.add_node("retrieve", _node_retrieve(deps))
    builder.add_node("assess", _node_assess(deps))
    builder.add_node("refine", _node_refine(deps))
    builder.add_node("generate", _node_generate(deps))
    builder.add_node("refuse", _node_refuse(deps))
    builder.add_node("score", _node_score(deps))

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "retrieve")
    builder.add_edge("retrieve", "assess")
    builder.add_conditional_edges(
        "assess",
        route_after_assess,
        {"generate": "generate", "refine": "refine", "refuse": "refuse"},
    )
    builder.add_edge("refine", "retrieve")
    builder.add_edge("generate", "score")
    builder.add_edge("score", END)
    builder.add_edge("refuse", END)
    return builder.compile(checkpointer=checkpointer or MemorySaver())
