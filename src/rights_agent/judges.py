"""The RAG triad, a citation check, and the calibration that makes them mean something.

============================  ==========================================  ==============
Metric                        Question it answers                         Blames
============================  ==========================================  ==============
``context_relevance``         Did we retrieve material capable of          retrieval
                              answering?
``groundedness``              Is every claim supported by that material?  generation
``answer_relevance``          Does the answer address what was asked?      alignment
``citation_coverage``         Is each claim attributable?                  attribution
============================  ==========================================  ==============

Two judges behind one interface:

* :class:`HeuristicJudge` -- lexical overlap.  Deterministic, offline, cheap,
  weak.  Runs in CI on every commit and never flakes.
* :class:`LLMJudge` -- a model reading a rubric.  Stronger, slower, biased,
  non-deterministic.  Runs on a sample.

And :func:`cohens_kappa`, which is the part most projects skip.  Kappa corrects
for chance: if 90% of answers are good, a judge that says "good" every time
scores 90% raw agreement and is worthless.  **Gate the instrument before you
gate with it.**
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Protocol

from rights_agent.config import Settings
from rights_agent.config import settings as load_settings
from rights_agent.document.nodes import citation_resolves
from rights_agent.llm import (
    CITATION_RE as _CITATION_RE,
)
from rights_agent.llm import (
    extract_citations,
    make_client,
    parse_context,
)
from rights_agent.log import get_logger
from rights_agent.retrieval import GENERIC_TERMS as QUESTION_SCAFFOLDING
from rights_agent.telemetry import EVALUATOR, SEMCONV, span

log = get_logger("judges")

_WORD_RE = re.compile(r"[a-z0-9£][a-z0-9£'\-]*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


#: Words that carry no evidential weight when checking whether a sentence is
#: supported by the context.
_FILLER = frozenset(
    ["a", "about", "above", "after", "all", "also", "an", "and", "any", "are", "as", "at", "be", "been", "being", "between", "but", "by", "can", "cannot", "could", "did", "do", "does", "doing", "done", "during", "each", "either", "else", "for", "from", "further", "had", "has", "have", "having", "he", "her", "him", "his", "how", "however", "if", "in", "into", "is", "it", "its", "itself", "may", "means", "might", "must", "no", "nor", "not", "of", "on", "only", "or", "other", "our", "out", "over", "own", "provides", "same", "shall", "she", "should", "so", "some", "such", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "through", "to", "under", "until", "up", "upon", "was", "we", "were", "what", "when", "where", "whether", "which", "while", "who", "whom", "whose", "will", "with", "within", "without", "would", "you", "your"]
)

#: A sentence counts as grounded when this fraction of its content words appears
#: in the context.  Set from observation, not taste: legal answers that quote
#: their source clear 0.8 comfortably, and paraphrases sit far below it -- which
#: is the known weakness this judge is calibrated against.
GROUNDED_SENTENCE_THRESHOLD = 0.6

#: Minimum content words for a sentence to be scored at all.  Shorter fragments
#: ("It does.") are noise in either direction.
MIN_SENTENCE_WORDS = 3


def content_words(text: str) -> set[str]:
    """Words that could carry evidential weight, for context and answer text."""
    return {word for word in _WORD_RE.findall(text.lower()) if word not in _FILLER and len(word) > 2}


def question_terms(question: str) -> set[str]:
    """Content words of a *question*, minus its scaffolding.

    "What does the document say about X" carries four words that no provision
    can ever match.  Counting them against relevance would depress every score
    by a constant that has nothing to do with retrieval or generation -- so the
    same stoplist the sufficiency gate uses is applied here, from one place.
    """
    return content_words(question) - QUESTION_SCAFFOLDING


#: A sentinel that cannot occur in an answer, so masking is reversible.
_MASK = "\x00{}\x00"
_MASK_RE = re.compile(r"\x00(\d+)\x00")


def sentences(text: str) -> list[str]:
    """Split into sentences without splitting *inside* a citation.

    Citations abbreviate, and abbreviations end in a full stop followed by a
    space: ``[Sch. 12 para. 4(2)]``.  Splitting on ``". "`` tore that mark into
    ``[Sch.`` / ``12 para.`` / ``4(2)] provides: ...``, and the fragment that
    carried the claim no longer contained a ``[...]`` mark at all -- so
    citation coverage read 0.00 for an answer in which every sentence was
    correctly cited.  Mask the marks, split, then restore them.
    """
    marks: list[str] = []

    def _hide(match: re.Match[str]) -> str:
        marks.append(match.group(0))
        return _MASK.format(len(marks) - 1)

    masked = _CITATION_RE.sub(_hide, text.strip())
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(masked) if part.strip()]
    restored = [_MASK_RE.sub(lambda m: marks[int(m.group(1))], part) for part in parts]
    return [part for part in restored if part.strip()]


# --------------------------------------------------------------------------- #
# Score container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class JudgeScores:
    """All four metrics, plus which judge produced them."""

    context_relevance: float = 0.0
    groundedness: float = 0.0
    answer_relevance: float = 0.0
    citation_coverage: float = 0.0
    judge: str = ""
    error: str = ""
    detail: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        """The flat form that goes on a metrics row's ``scores``."""
        return {
            "context_relevance": round(self.context_relevance, 4),
            "groundedness": round(self.groundedness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "citation_coverage": round(self.citation_coverage, 4),
        }

    def as_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class Judge(Protocol):
    name: str

    def score(
        self, question: str, context: str, answer: str, citations: Sequence[str] = ()
    ) -> JudgeScores: ...


# --------------------------------------------------------------------------- #
# Heuristic judge
# --------------------------------------------------------------------------- #
class HeuristicJudge:
    """Lexical-overlap judge: deterministic, offline, and honestly weak.

    Its structural blind spot is paraphrase -- a correct answer that reuses none
    of the context's vocabulary scores zero.  That is not a bug to be patched
    out; it is the reason :func:`calibrate` exists, and the reason the
    calibration set deliberately contains paraphrases.
    """

    name = "heuristic-lexical"

    def score(
        self, question: str, context: str, answer: str, citations: Sequence[str] = ()
    ) -> JudgeScores:
        with span("rag.judge", EVALUATOR, **{"evaluator.name": self.name}) as current:
            asked = question_terms(question)
            context_terms = content_words(context)
            answer_terms = content_words(answer)

            context_relevance = len(asked & context_terms) / len(asked) if asked else 0.0
            answer_relevance = len(asked & answer_terms) / len(asked) if asked else 0.0
            groundedness, grounded_detail = self._groundedness(answer, context_terms)
            citation_coverage, citation_detail = self._citation_coverage(answer, context, citations)

            scores = JudgeScores(
                context_relevance=round(min(1.0, context_relevance), 6),
                groundedness=round(groundedness, 6),
                answer_relevance=round(min(1.0, answer_relevance), 6),
                citation_coverage=round(citation_coverage, 6),
                judge=self.name,
                detail={**grounded_detail, **citation_detail},
            )
            current.set_attributes({SEMCONV.OUTPUT_VALUE: scores.as_json(), **scores.to_dict()})
            return scores

    @staticmethod
    def _groundedness(answer: str, context_terms: set[str]) -> tuple[float, dict[str, float]]:
        scored = 0
        supported = 0
        for sentence in sentences(answer):
            terms = content_words(sentence)
            if len(terms) < MIN_SENTENCE_WORDS:
                continue
            scored += 1
            overlap = len(terms & context_terms) / len(terms)
            if overlap >= GROUNDED_SENTENCE_THRESHOLD:
                supported += 1
        if scored == 0:
            return 0.0, {"sentences_scored": 0.0, "sentences_supported": 0.0}
        return supported / scored, {
            "sentences_scored": float(scored),
            "sentences_supported": float(supported),
        }

    @staticmethod
    def _citation_coverage(
        answer: str, context: str, citations: Sequence[str]
    ) -> tuple[float, dict[str, float]]:
        """Fraction of substantive sentences that carry a resolvable citation.

        A citation that does not appear in the context is worse than no
        citation: it looks like attribution and is not.  Those are counted as
        uncited *and* reported separately.

        Resolution is at provision level (see
        :func:`~rights_agent.document.nodes.canonical_citation`).  A model that
        cites ``s.19(4)`` where the block was headed ``s.19`` has cited *more*
        precisely, not wrongly, and scoring that zero would measure the
        formatting of the citation rather than whether the claim is attributable.
        """
        available = [block.citation for block in parse_context(context)]
        found = list(citations) or extract_citations(answer)
        unresolvable = [
            citation
            for citation in found
            if available and not citation_resolves(citation, available)
        ]

        scored = 0
        cited = 0
        for sentence in sentences(answer):
            if len(content_words(sentence)) < MIN_SENTENCE_WORDS:
                continue
            scored += 1
            marks = [match.group(1).strip() for match in _CITATION_RE.finditer(sentence)]
            if marks and (
                not available or any(citation_resolves(mark, available) for mark in marks)
            ):
                cited += 1
        detail = {
            "citations_found": float(len(found)),
            "citations_unresolvable": float(len(unresolvable)),
        }
        if scored == 0:
            return 0.0, detail
        return cited / scored, detail


# --------------------------------------------------------------------------- #
# LLM judge
# --------------------------------------------------------------------------- #
JUDGE_RUBRIC = """You are grading one answer produced by a retrieval system.

Score each dimension from 0.0 to 1.0:

- context_relevance: does the CONTEXT contain material capable of answering the QUESTION?
- groundedness: is every factual claim in the ANSWER supported by the CONTEXT?
- answer_relevance: does the ANSWER address the QUESTION that was asked?
- citation_coverage: does each claim carry a citation that appears in the CONTEXT?

Rules:
- Ignore length. A short answer is not worse for being short.
- Ignore formatting, tone, and whether the answer reads well.
- Judge only support and relevance. Do not reward fluency.
- A claim absent from the CONTEXT is ungrounded even if it is true in the world.

Reply with JSON only, exactly:
{"context_relevance": 0.0, "groundedness": 0.0, "answer_relevance": 0.0, "citation_coverage": 0.0}
"""


class LLMJudge:
    """A model reading :data:`JUDGE_RUBRIC`.

    Stronger than the heuristic and worse behaved: slower, non-deterministic,
    and biased toward answers that look like what it would have written.  It
    runs on a sample, never as the CI gate, and its agreement with human labels
    is measured before any of its numbers are believed.
    """

    name = "llm-rubric"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self._client = make_client(self.settings)
        self.name = f"llm-rubric:{self._client.model}"

    def score(
        self, question: str, context: str, answer: str, citations: Sequence[str] = ()
    ) -> JudgeScores:
        prompt = (
            f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}\n\n"
            "Reply with JSON only."
        )
        with span("rag.judge", EVALUATOR, **{"evaluator.name": self.name}) as current:
            try:
                raw = "".join(self._client.stream(JUDGE_RUBRIC, prompt)).strip()
                payload = _extract_json(raw)
                scores = JudgeScores(
                    context_relevance=_clamp(payload.get("context_relevance")),
                    groundedness=_clamp(payload.get("groundedness")),
                    answer_relevance=_clamp(payload.get("answer_relevance")),
                    citation_coverage=_clamp(payload.get("citation_coverage")),
                    judge=self.name,
                )
            except Exception as exc:  # noqa: BLE001 - an unparseable judge is a result
                log.warning("LLM judge failed: %s", exc)
                scores = JudgeScores(judge=self.name, error=f"{type(exc).__name__}: {exc}")
                current.record_exception(exc)
            current.set_attributes({SEMCONV.OUTPUT_VALUE: scores.as_json(), **scores.to_dict()})
            return scores


def _extract_json(text: str) -> Mapping[str, object]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in judge reply: {text[:120]!r}")
    return json.loads(text[start : end + 1])


def _clamp(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def cohens_kappa(human: Sequence[int], machine: Sequence[int]) -> float:
    """Chance-corrected agreement between two binary labellers.

    Returns 0.0 when agreement is entirely explained by chance -- including the
    degenerate case where one labeller never varies, which is exactly the case
    raw agreement flatters.
    """
    if len(human) != len(machine):
        raise ValueError("label sequences must be the same length")
    total = len(human)
    if total == 0:
        return 0.0
    observed = sum(1 for h, m in zip(human, machine, strict=True) if h == m) / total
    expected = 0.0
    for label in (0, 1):
        p_human = sum(1 for h in human if h == label) / total
        p_machine = sum(1 for m in machine if m == label) / total
        expected += p_human * p_machine
    if expected >= 1.0:
        # Both labellers were constant and identical: agreement is total and
        # entirely uninformative.  Reporting 1.0 here would be a lie.
        return 0.0
    return round((observed - expected) / (1.0 - expected), 6)


@dataclass(frozen=True, slots=True)
class Calibration:
    """How well a judge agrees with the humans who labelled the set."""

    judge: str
    n: int
    threshold: float
    kappa: float
    agreement: float
    confusion: dict[str, int]
    disagreements: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def calibrate(
    judge: Judge,
    rows: Sequence[Mapping[str, object]],
    scorer: Callable[[JudgeScores], float] | None = None,
    threshold: float = 0.7,
) -> Calibration:
    """Compare a judge's binarised score against human labels.

    Each row needs ``question``, ``context``, ``answer`` and ``human_label``
    (1 = acceptable, 0 = not).  Expect the kappa to drop materially once hard
    cases are added to the set: the judge did not get worse, the test got
    honest.
    """
    scorer = scorer or (lambda scores: scores.groundedness)
    human: list[int] = []
    machine: list[int] = []
    confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    disagreements: list[dict[str, object]] = []

    for row in rows:
        question = str(row.get("question", ""))
        context = str(row.get("context", ""))
        answer = str(row.get("answer", ""))
        label = int(row.get("human_label", 0))
        scores = judge.score(question, context, answer)
        value = scorer(scores)
        predicted = 1 if value >= threshold else 0
        human.append(label)
        machine.append(predicted)
        key = {(1, 1): "tp", (0, 1): "fp", (0, 0): "tn", (1, 0): "fn"}[(label, predicted)]
        confusion[key] += 1
        if predicted != label:
            disagreements.append(
                {
                    "id": row.get("id", ""),
                    "human_label": label,
                    "machine_label": predicted,
                    "score": round(value, 4),
                    "note": row.get("note", ""),
                }
            )

    agreement = (
        sum(1 for h, m in zip(human, machine, strict=True) if h == m) / len(human) if human else 0.0
    )
    return Calibration(
        judge=getattr(judge, "name", "unknown"),
        n=len(human),
        threshold=threshold,
        kappa=cohens_kappa(human, machine),
        agreement=round(agreement, 6),
        confusion=confusion,
        disagreements=disagreements,
    )


def score_summary(scores: Sequence[JudgeScores]) -> dict[str, float]:
    """Mean of each dimension over a batch (the aggregate the gate reads)."""
    if not scores:
        return {}
    return {
        name: round(mean(getattr(score, name) for score in scores), 6)
        for name in ("context_relevance", "groundedness", "answer_relevance", "citation_coverage")
    }
