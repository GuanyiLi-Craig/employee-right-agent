"""Session transcripts, and resolving a follow-up against them.

A chat interface invites follow-ups -- *"and what about section 19?"*, *"how long
is that?"* -- which retrieve for nothing on their own, because the words that
make them findable are in the previous turn.

The resolution here is deliberately **not** a model call, for the same reason
intent classification is not: a rewrite that changes when you re-run it makes
every downstream measurement unreproducible, and a round trip per turn is a real
bill. Instead a follow-up borrows the distinctive terms from the previous
question in the same session -- cheap, deterministic, and inspectable in the
trace.

Be honest about what this is and is not. It resolves *topic*, not reference: it
carries "bereavement leave" forward so the retriever has something to match. It
does not understand that "that" meant the notice period. The agent answers each
question from the retrieved provisions, and the transcript is context for
retrieval rather than a memory the model reasons over.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from rights_agent.log import get_logger
from rights_agent.retrieval import GENERIC_TERMS

log = get_logger("conversation")

#: Turns kept per session. Enough for a demo conversation and bounded, because
#: an unbounded in-memory transcript store is a slow memory leak with a
#: data-protection surface attached.
MAX_TURNS = 40

#: Sessions kept in memory, oldest evicted first.
MAX_SESSIONS = 200

#: A question this short, with no topic word of its own, cannot stand alone.
#: Kept deliberately tight: at seven words "What does the document say about
#: tips?" would qualify, and quietly appending the previous turn's topic to a
#: perfectly self-contained question is worse than not resolving at all.
SHORT_QUESTION_WORDS = 5

#: Shortest word treated as a topic. Four, not five: this corpus's own topic
#: words include "tips", "wage", "paid" and "sick", and a threshold that cannot
#: see them classifies a question about tips as having no subject.
MIN_TERM_LENGTH = 4

#: How far back to search for the question that established the topic.
#:
#: Not simply "the previous question": in a chain of follow-ups ("how long is
#: it?", then "and for agency workers?") the previous question is itself a
#: follow-up carrying no topic, so borrowing from it borrows nothing. The search
#: walks back to the most recent question that *has* distinctive terms, and stops
#: after this many so an ancient topic cannot resurface.
MAX_LOOKBACK = 6

_WORD_RE = re.compile(r"[a-z0-9£][a-z0-9£'\-]*")

#: Openings and pronouns that mark a question as a continuation rather than a
#: new topic. Matched on a lowercased question.
_FOLLOW_UP_MARKERS: tuple[str, ...] = (
    "what about",
    "how about",
    "and what",
    "and how",
    "and the",
    "what if",
    "does it",
    "does that",
    "is it",
    "is that",
    "can it",
    "can they",
    "who does",
    "how long",
    "how much",
    "how many",
    "why is",
    "why does",
    "the same",
    "instead",
    "as well",
    "too?",
    "also",
    "that one",
    "those",
    "it apply",
    "they apply",
)

#: Standalone pronouns that only make sense against a previous turn.
_ANAPHORA = frozenset({"it", "its", "that", "this", "they", "them", "those", "these", "there"})

#: A question opening with a conjunction is a continuation whatever follows it.
#: "And for agency workers?" carries its own distinctive terms and is still a
#: follow-up -- the topic it narrows lives in the previous turn.
_LEADING_CONJUNCTIONS = frozenset({"and", "or", "but", "also", "plus", "so"})


@dataclass(frozen=True, slots=True)
class Turn:
    """One message in a transcript.

    ``reconstructed`` marks a turn rebuilt from the audit record rather than
    replayed from memory.  The audit log deliberately does not store answer
    prose, so a reconstructed agent turn carries its citations and its numbers
    but not the words -- and the UI has to say so rather than render an empty
    bubble and let the reader assume the model said nothing.
    """

    role: str  # "user" | "agent"
    content: str
    ts: float = field(default_factory=time.time)
    request_id: str = ""
    citations: tuple[str, ...] = ()
    refused: bool = False
    reconstructed: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["citations"] = list(self.citations)
        return payload


class ConversationStore:
    """Bounded, thread-safe, in-memory transcripts keyed by session id.

    In memory on purpose: a transcript store is a personal-data store, and this
    project keeps exactly one durable record of a request -- the audit log, with
    redaction applied at capture. Adding a second durable copy of user questions
    for the convenience of a chat UI would be the wrong trade.
    """

    def __init__(self, max_turns: int = MAX_TURNS, max_sessions: int = MAX_SESSIONS) -> None:
        self.max_turns = max_turns
        self.max_sessions = max_sessions
        self._sessions: dict[str, deque[Turn]] = {}
        self._order: deque[str] = deque()
        self._lock = threading.Lock()

    def add(self, session_id: str, turn: Turn) -> None:
        with self._lock:
            transcript = self._sessions.get(session_id)
            if transcript is None:
                transcript = deque(maxlen=self.max_turns)
                self._sessions[session_id] = transcript
                self._order.append(session_id)
                while len(self._order) > self.max_sessions:
                    evicted = self._order.popleft()
                    self._sessions.pop(evicted, None)
            transcript.append(turn)

    def history(self, session_id: str) -> list[Turn]:
        with self._lock:
            return list(self._sessions.get(session_id, ()))

    def questions(self, session_id: str) -> list[str]:
        return [turn.content for turn in self.history(session_id) if turn.role == "user"]

    def transcript(self, session_id: str) -> list[dict[str, Any]]:
        return [turn.to_dict() for turn in self.history(session_id)]

    def sessions(self) -> list[str]:
        with self._lock:
            return list(self._order)

    def live(self) -> dict[str, list[Turn]]:
        """Every in-memory transcript, newest session last."""
        with self._lock:
            return {session: list(turns) for session, turns in self._sessions.items()}

    def clear(self, session_id: str | None = None) -> None:
        with self._lock:
            if session_id is None:
                self._sessions.clear()
                self._order.clear()
            else:
                self._sessions.pop(session_id, None)
                if session_id in self._order:
                    self._order.remove(session_id)


# --------------------------------------------------------------------------- #
# History across both tiers
# --------------------------------------------------------------------------- #
#: Sessions offered in the history list.  A list nobody can scan is a list
#: nobody uses.
MAX_LISTED_SESSIONS = 25

#: Characters of the opening question used as a conversation's title.
TITLE_CHARS = 64


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """One row of the history list.

    ``source`` is ``live`` when the full transcript is still in memory and
    ``audit`` when only the durable record survives.  The distinction is the
    honest one: transcripts are held in memory on purpose, so a restart leaves
    the questions and the numbers but not the answers.
    """

    session_id: str
    title: str
    source: str
    turns: int
    requests: int
    refused: int
    cost_usd: float
    first_ts: float
    last_ts: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _title_of(question: str) -> str:
    text = " ".join(question.split())
    return text if len(text) <= TITLE_CHARS else text[: TITLE_CHARS - 1] + "…"


def _epoch(value: Any) -> float:
    """Parse an ISO-8601 stamp from an audit row into an epoch float."""
    from datetime import datetime

    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (TypeError, ValueError):
        return 0.0


def reconstruct_turns(audit_rows: Sequence[Mapping[str, Any]]) -> list[Turn]:
    """Rebuild a transcript from audit records for one session.

    Answers are not in the audit record, by design -- storing the model's prose
    durably would mean a second copy of everything the trace already holds,
    under weaker retention rules.  So each agent turn is reconstructed from what
    *was* recorded: whether it answered or refused, and what it cited.
    """
    turns: list[Turn] = []
    for row in sorted(audit_rows, key=lambda r: int(r.get("sequence", 0))):
        stamp = _epoch(row.get("ts"))
        request_id = str(row.get("request_id") or "")
        turns.append(
            Turn(
                role="user",
                content=str(row.get("question") or ""),
                ts=stamp,
                request_id=request_id,
                reconstructed=True,
            )
        )
        refused = bool(row.get("refused"))
        citations = tuple(str(c) for c in (row.get("citations") or []))
        turns.append(
            Turn(
                role="agent",
                content=(
                    "Refused. The answer text is not retained: the audit record keeps the "
                    "decision, its score and its sources, not the model's prose."
                    if refused
                    else "Answered. The answer text is not retained — the audit record keeps "
                    "the citations and the numbers, not the model's prose."
                ),
                ts=stamp,
                request_id=request_id,
                citations=citations,
                refused=refused,
                reconstructed=True,
            )
        )
    return turns


def summarise_sessions(
    live: Mapping[str, Sequence[Turn]],
    audit_rows: Sequence[Mapping[str, Any]],
    metrics_rows: Sequence[Mapping[str, Any]] = (),
    limit: int = MAX_LISTED_SESSIONS,
    actors: Sequence[str] | None = None,
) -> list[SessionSummary]:
    """The history list, **newest first regardless of tier**.

    Recency is the only ranking: which tier a conversation came from is a badge,
    not a rank.  Sorting live transcripts above audit-only ones would bury the
    conversation you had five minutes before a restart underneath one you have
    barely started.

    ``actors`` filters the audit tier to requests made by particular callers.
    The chat's history is the chat's own history: the same audit log also holds
    every synthetic request from the traffic controls and every row of the
    golden set, and a history list swamped by ``eval-g029`` is a list nobody
    reads.  The unfiltered log stays available through the audit panel, which is
    where "everything this process ever decided" belongs.

    Cost comes from the metrics rows, which already group by session -- the same
    grouping the cost panel's per-conversation figure uses, so the two cannot
    disagree.
    """
    allowed = set(actors) if actors else None
    by_session: dict[str, list[Mapping[str, Any]]] = {}
    for row in audit_rows:
        if allowed is not None and str(row.get("actor") or "") not in allowed:
            continue
        by_session.setdefault(str(row.get("session_id") or ""), []).append(row)

    cost: dict[str, float] = {}
    for row in metrics_rows:
        key = str(row.get("session_id") or "")
        cost[key] = round(cost.get(key, 0.0) + float(row.get("cost_total_usd") or row.get("cost_usd") or 0.0), 8)

    summaries: list[SessionSummary] = []
    for session_id in set(live) | set(by_session) - {""}:
        if not session_id:
            continue
        turns = list(live.get(session_id, ()))
        rows = by_session.get(session_id, [])
        questions = [t.content for t in turns if t.role == "user"] or [
            str(r.get("question") or "") for r in sorted(rows, key=lambda r: int(r.get("sequence", 0)))
        ]
        if not questions:
            continue
        stamps = [t.ts for t in turns] or [_epoch(r.get("ts")) for r in rows]
        summaries.append(
            SessionSummary(
                session_id=session_id,
                title=_title_of(questions[0]),
                source="live" if turns else "audit",
                turns=len(turns) or len(rows) * 2,
                requests=len(rows) or len(questions),
                refused=sum(1 for r in rows if r.get("refused")),
                cost_usd=cost.get(session_id, 0.0),
                first_ts=min(stamps) if stamps else 0.0,
                last_ts=max(stamps) if stamps else 0.0,
            )
        )
    summaries.sort(key=lambda s: s.last_ts, reverse=True)
    return summaries[:limit]


# --------------------------------------------------------------------------- #
# Follow-up resolution
# --------------------------------------------------------------------------- #
def _terms(text: str) -> list[str]:
    return [
        word
        for word in _WORD_RE.findall(text.lower())
        if len(word) >= MIN_TERM_LENGTH and word not in GENERIC_TERMS
    ]


def looks_like_follow_up(question: str) -> bool:
    """Whether ``question`` probably needs the previous turn to make sense.

    Three signals, any of which is enough: it is very short, it opens with a
    continuation phrase, or it leans on a pronoun with no noun of its own.
    Cheap and imperfect -- and the cost of a false positive is only that a few
    extra terms join the retrieval query.
    """
    lowered = question.lower().strip()
    if not lowered:
        return False
    words = _WORD_RE.findall(lowered)
    if words and words[0] in _LEADING_CONJUNCTIONS:
        return True
    if any(marker in lowered for marker in _FOLLOW_UP_MARKERS):
        return True
    if len(words) <= SHORT_QUESTION_WORDS and not _terms(question):
        return True
    return bool(_ANAPHORA & set(words) and len(_terms(question)) <= 1)


@dataclass(frozen=True, slots=True)
class Contextualisation:
    """What was sent to the retriever, and why."""

    query: str
    used_history: bool
    borrowed: tuple[str, ...] = ()
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["borrowed"] = list(self.borrowed)
        return payload


def contextualise(
    question: str, previous_questions: Iterable[str], max_terms: int = 6
) -> Contextualisation:
    """Resolve a follow-up by borrowing terms from the question that set the topic.

    Returns the question unchanged whenever it stands on its own, so a normal
    question is never quietly rewritten -- and for anything but a follow-up the
    *original* question is what the sufficiency gate scores.
    """
    question = question.strip()
    previous = [q for q in previous_questions if q.strip()][-MAX_LOOKBACK:]
    if not question or not previous:
        return Contextualisation(query=question, used_history=False)
    if not looks_like_follow_up(question):
        return Contextualisation(
            query=question, used_history=False, reason="question stands on its own"
        )

    already = set(_WORD_RE.findall(question.lower()))
    borrowed: list[str] = []
    # Borrow from the most recent question that is not itself a follow-up. In a
    # chain -- "how long is it?", then "and for agency workers?" -- the question
    # immediately before carries no topic, and borrowing "long" from it is worse
    # than borrowing nothing.
    for earlier in reversed(previous):
        if looks_like_follow_up(earlier):
            continue
        for term in _terms(earlier):
            if term not in already and term not in borrowed:
                borrowed.append(term)
            if len(borrowed) >= max_terms:
                break
        if borrowed:
            break

    if not borrowed:
        return Contextualisation(
            query=question, used_history=False, reason="nothing distinctive to carry forward"
        )
    log.debug("follow-up resolved with %s", borrowed)
    return Contextualisation(
        query=f"{question} {' '.join(borrowed)}",
        used_history=True,
        borrowed=tuple(borrowed),
        reason="follow-up: borrowed the distinctive terms of the last question with a topic",
    )
