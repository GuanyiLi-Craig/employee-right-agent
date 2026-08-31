"""Search, small-to-big expansion, context assembly and the sufficiency gate.

Three failure modes are designed against here, because each one is silent:

* **Unbounded expansion.** In an amending document a single "section" can
  contain an entire inserted chapter -- tens of thousands of characters.
  Expansion is an optimisation, not an obligation, so it is capped.
* **A context budget that skips.** A naive assembler ``break``s on the first
  block that does not fit; if the top hit is a 36,000-character provision you
  get an *empty context* and a model answering from nothing, with no error
  anywhere.  Blocks are truncated, never skipped.
* **Scoring the wrong question.** Sufficiency is always computed against the
  *original* question.  A refined query can retrieve beautifully for itself and
  still fail the user.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from rights_agent.config import LEAF_COLLECTION, PARENT_COLLECTION, Settings
from rights_agent.config import settings as load_settings
from rights_agent.embedding import assert_embedder_matches
from rights_agent.log import get_logger
from rights_agent.store import (
    IndexManifest,
    chroma_client,
    open_collection,
    pinned_embedder,
    require_manifest,
)
from rights_agent.telemetry import RETRIEVER, SEMCONV, set_retrieval_documents, span

log = get_logger("retrieval")

#: Weighting of the two sufficiency signals (§11.4).
SIMILARITY_WEIGHT = 0.35
COVERAGE_WEIGHT = 0.65

#: Words too generic to count as evidence that a question was covered.
#:
#: Two groups: ordinary English scaffolding, and the phrasing this project asks
#: golden questions to use ("what does the *document* say about...", "which
#: *provision* *covers*...").  Leaving the second group in would mean every
#: question carried terms that no provision can ever match, which drags
#: coverage down uniformly and makes the threshold meaningless.
GENERIC_TERMS = frozenset(
    ["about", "above", "according", "act", "acts", "after", "against", "already", "although", "another", "anything", "around", "because", "before", "being", "below", "between", "cannot", "could", "cover", "covered", "covers", "different", "does", "doing", "document", "documents", "during", "either", "enough", "entitled", "every", "explain", "from", "further", "generally", "happen", "having", "however", "itself", "least", "legal", "legally", "maybe", "mention", "mentioned", "mentions", "means", "might", "much", "must", "never", "nothing", "often", "other", "others", "outside", "please", "provide", "provided", "provides", "provision", "provisions", "purposes", "rather", "really", "regarding", "relation", "require", "required", "requirement", "requires", "rights", "said", "say", "says", "section", "sections", "should", "simply", "since", "something", "specific", "state", "stated", "states", "statute", "still", "such", "summarise", "summarize", "their", "there", "these", "thing", "things", "those", "through", "under", "unless", "until", "using", "various", "very", "what", "whatever", "when", "where", "whether", "which", "while", "whose", "within", "without", "would"]
)

_WORD_RE = re.compile(r"[a-z0-9£][a-z0-9£'\-]*")

#: Number of top hits considered for widening, and for the similarity signal.
EXPAND_TOP = 3
SUFFICIENCY_TOP = 3

#: Candidate leaves fetched per requested block.
#:
#: Widening is *subtractive*: several of the top leaves routinely sit in the same
#: provision, and the parent block quotes all of them, so the duplicates are
#: dropped. Fetching exactly ``k`` leaves therefore returned fewer than ``k``
#: blocks -- three, for a question whose top six leaves came from two
#: provisions. The generator got half the context budget it was configured to
#: get, and the shortfall was invisible: no error, just a thinner answer. Fetch a
#: pool, dedupe, then trim to ``k``.
OVERFETCH = 3


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Doc:
    """One retrieval result.

    ``score`` is cosine similarity in 0..1 (``1 - distance``).  ``expanded``
    records whether this block was widened from a leaf to its parent provision,
    which is the difference between quoting a subsection and quoting the
    provision that gives it meaning.
    """

    id: str
    citation: str
    breadcrumb: str
    text: str
    score: float
    parent_id: str = ""
    expanded: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form for LangGraph state (which must be serialisable)."""
        return {
            "id": self.id,
            "citation": self.citation,
            "breadcrumb": self.breadcrumb,
            "text": self.text,
            "score": self.score,
            "parent_id": self.parent_id,
            "expanded": self.expanded,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Doc:
        return cls(
            id=str(payload.get("id", "")),
            citation=str(payload.get("citation", "")),
            breadcrumb=str(payload.get("breadcrumb", "")),
            text=str(payload.get("text", "")),
            score=float(payload.get("score", 0.0)),
            parent_id=str(payload.get("parent_id", "")),
            expanded=bool(payload.get("expanded", False)),
            metadata=dict(payload.get("metadata", {})),
        )


# --------------------------------------------------------------------------- #
# Lexical helpers
# --------------------------------------------------------------------------- #
def words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def distinctive_words(question: str, min_length: int = 5) -> set[str]:
    """Words in the question that would be evidence of coverage if retrieved.

    Short words and a generic stoplist are removed: "what does the document say
    about..." is scaffolding, and counting it as coverage would make every
    question look answerable.
    """
    return {
        word
        for word in _WORD_RE.findall(question.lower())
        if len(word) >= min_length and word not in GENERIC_TERMS
    }


def coverage(question: str, docs: Sequence[Doc]) -> float:
    """Fraction of the question's distinctive words present in the evidence."""
    terms = distinctive_words(question)
    if not terms:
        return 0.0
    haystack: set[str] = set()
    for doc in docs:
        haystack |= words(doc.text)
        haystack |= words(doc.breadcrumb)
    return len(terms & haystack) / len(terms)


def sufficiency(docs: Sequence[Doc], question: str, top: int = SUFFICIENCY_TOP) -> float:
    """Hybrid gate: similarity and lexical coverage, weighted toward coverage.

    Two signals because neither is trustworthy alone.  Similarity is poorly
    calibrated *across* queries -- 0.31 on one question means something
    different from 0.31 on another -- so it can never be the whole gate.
    Coverage is crude and lexical and *decisive*: "cryptocurrency" is simply not
    in an employment statute, and no embedding score should be allowed to hide
    that.
    """
    if not docs:
        return 0.0
    scores = sorted((doc.score for doc in docs), reverse=True)[:top]
    similarity = mean(scores) if scores else 0.0
    return round(
        SIMILARITY_WEIGHT * max(0.0, min(1.0, similarity)) + COVERAGE_WEIGHT * coverage(question, docs),
        6,
    )


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #
class Retriever:
    """Reads the two hierarchical collections.

    Construction enforces the pinning rule: the embedder comes from the
    manifest, never from this process's preference.  Querying a hashed index
    with MiniLM vectors does not raise -- it returns confident nonsense -- so
    the only safe behaviour is to refuse to start.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.manifest: IndexManifest = require_manifest(self.settings)
        embedder, name = pinned_embedder(self.settings, self.manifest)
        assert_embedder_matches(self.manifest.embedding_model, name)
        self.embedder = embedder
        self.embedder_name = name
        client = chroma_client(self.settings)
        self.leaves = open_collection(client, LEAF_COLLECTION, embedder)
        self.parents = open_collection(client, PARENT_COLLECTION, embedder)
        #: Read once: it bounds the candidate pool, and Chroma raises rather than
        #: clamping when ``n_results`` exceeds the collection size.
        self.leaf_count = self.leaves.count()
        log.info(
            "retriever ready: index_version=%s embedder=%s leaves=%d",
            self.manifest.index_version,
            name,
            self.leaf_count,
        )

    @property
    def index_version(self) -> str:
        return self.manifest.index_version

    # ---- search -----------------------------------------------------------
    def search(
        self,
        query: str,
        k: int | None = None,
        where: dict[str, Any] | None = None,
        expand: bool = True,
    ) -> list[Doc]:
        """Retrieve ``k`` leaves, optionally widened to their provisions."""
        k = k or self.settings.top_k
        query = query.strip()
        if not query:
            return []

        with span(
            "rag.retrieve",
            RETRIEVER,
            **{
                SEMCONV.INPUT_VALUE: query,
                "retrieval.k": k,
                "retrieval.expand": expand,
                "metadata.index_version": self.index_version,
                "metadata.embedding_model": self.embedder_name,
            },
        ) as current:
            pool = min(k * OVERFETCH, self.leaf_count) if expand else k
            raw = self.leaves.query(
                query_texts=[query],
                n_results=pool,
                where=where or None,
                include=["documents", "metadatas", "distances"],
            )
            docs = self._to_docs(raw)
            if expand:
                docs = self._expand(docs)[:k]
            current.set_attribute("retrieval.k", k)
            current.set_attribute("retrieval.pool", pool)
            current.set_attribute("retrieval.returned", len(docs))
            current.set_attribute("retrieval.expanded", sum(1 for d in docs if d.expanded))
            set_retrieval_documents(
                current,
                [
                    {
                        "id": doc.id,
                        "score": doc.score,
                        "content": doc.text,
                        "metadata": {
                            "citation": doc.citation,
                            "breadcrumb": doc.breadcrumb,
                            "expanded": doc.expanded,
                        },
                    }
                    for doc in docs
                ],
            )
            current.set_output(", ".join(doc.citation for doc in docs))
        return docs

    def _to_docs(self, raw: Mapping[str, Any]) -> list[Doc]:
        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        docs: list[Doc] = []
        for index, identifier in enumerate(ids):
            metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            # Cosine space: Chroma returns 1 - cosine similarity.
            score = max(0.0, min(1.0, 1.0 - distance))
            raw_text = str(metadata.get("raw_text") or "")
            if not raw_text and index < len(documents):
                # Fall back to the embedded document, minus the breadcrumb line.
                embedded = str(documents[index] or "")
                raw_text = embedded.split("\n", 1)[-1] if "\n" in embedded else embedded
            docs.append(
                Doc(
                    id=str(identifier),
                    citation=str(metadata.get("citation") or identifier),
                    breadcrumb=str(metadata.get("breadcrumb") or ""),
                    text=raw_text,
                    score=round(score, 6),
                    parent_id=str(metadata.get("parent_id") or ""),
                    metadata=metadata,
                )
            )
        return docs

    # ---- small-to-big -----------------------------------------------------
    def _expand(self, docs: list[Doc]) -> list[Doc]:
        """Widen the top hits to their parent provision, subject to a cap."""
        cap = self.settings.max_parent_chars
        wanted = {doc.id for doc in docs[:EXPAND_TOP] if doc.parent_id}
        if not wanted:
            return docs
        parents = self._fetch_parents(
            [doc.parent_id for doc in docs[:EXPAND_TOP] if doc.id in wanted]
        )
        out: list[Doc] = []
        seen_citations: set[str] = set()
        covered_parents: set[str] = set()
        for doc in docs:
            parent = parents.get(doc.parent_id) if doc.id in wanted else None
            if parent is None:
                if doc.citation in seen_citations or doc.parent_id in covered_parents:
                    # A leaf whose provision was already widened in: the parent
                    # block quotes this text verbatim, so including the leaf as
                    # well spends context budget on a duplicate.
                    continue
                seen_citations.add(doc.citation)
                out.append(doc)
                continue
            text = str(parent.get("text") or "")
            if not text:
                out.append(doc)
                continue
            if len(text) > cap:
                # Keep the leaf.  Expanding here would silently consume the
                # whole context budget for one provision.
                if doc.citation in seen_citations:
                    continue
                seen_citations.add(doc.citation)
                out.append(
                    Doc(
                        **{
                            **doc.to_dict(),
                            "metadata": {
                                **doc.metadata,
                                "expand_skipped": True,
                                "parent_chars": len(text),
                            },
                        }
                    )
                )
                continue
            metadata = dict(parent.get("metadata") or {})
            citation = str(metadata.get("citation") or doc.citation)
            if citation in seen_citations:
                # Two leaves in the same provision widen to the same parent;
                # the parent's text already covers both, so keeping the second
                # copy would spend a retrieval slot on nothing.
                log.debug("dropping duplicate expansion of %s", citation)
                continue
            seen_citations.add(citation)
            covered_parents.add(doc.parent_id)
            out.append(
                Doc(
                    id=doc.parent_id,
                    citation=str(metadata.get("citation") or doc.citation),
                    breadcrumb=str(metadata.get("breadcrumb") or doc.breadcrumb),
                    text=text,
                    score=doc.score,
                    parent_id="",
                    expanded=True,
                    metadata={**metadata, "expanded_from": doc.id, "leaf_citation": doc.citation},
                )
            )
        return out

    def _fetch_parents(self, ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        unique = sorted({identifier for identifier in ids if identifier})
        if not unique:
            return {}
        try:
            raw = self.parents.get(ids=list(unique), include=["documents", "metadatas"])
        except Exception as exc:  # noqa: BLE001 - a missing parent is not fatal
            log.warning("parent lookup failed, keeping leaves: %s", exc)
            return {}
        out: dict[str, dict[str, Any]] = {}
        for index, identifier in enumerate(raw.get("ids") or []):
            metadata = dict((raw.get("metadatas") or [{}])[index] or {})
            text = str(metadata.get("raw_text") or "")
            if not text:
                embedded = str(((raw.get("documents") or [""])[index]) or "")
                text = embedded.split("\n", 1)[-1] if "\n" in embedded else embedded
            out[str(identifier)] = {"text": text, "metadata": metadata}
        return out


# --------------------------------------------------------------------------- #
# Context assembly
# --------------------------------------------------------------------------- #
def format_context(
    docs: Sequence[Doc],
    budget_chars: int | None = None,
    min_block_chars: int | None = None,
) -> str:
    """Assemble a prompt context, de-duplicated by citation.

    Truncates an oversized block rather than skipping it: an empty context is
    far worse than a shortened one, because the model answers anyway and
    nothing in the logs says why the answer was groundless.
    """
    settings = load_settings()
    budget = budget_chars if budget_chars is not None else settings.context_budget_chars
    minimum = min_block_chars if min_block_chars is not None else settings.min_block_chars
    if not docs:
        return ""

    blocks: list[str] = []
    seen: set[str] = set()
    remaining = budget
    for doc in docs:
        if doc.citation in seen:
            continue
        seen.add(doc.citation)
        header = f"[{doc.citation}] {doc.breadcrumb}".strip()
        body = doc.text.strip()
        block = f"{header}\n{body}" if body else header
        if len(block) <= remaining:
            blocks.append(block)
            remaining -= len(block) + 2
        else:
            allowance = max(minimum, remaining) - len(header) - 2
            if allowance <= 0:
                if blocks:
                    break
                # Nothing has fitted yet: emit the header alone rather than
                # returning an empty context.
                blocks.append(header[: max(0, budget)])
                break
            blocks.append(f"{header}\n{body[:allowance].rstrip()} […truncated]")
            break
        if remaining <= minimum:
            break
    return "\n\n".join(blocks).strip()


def refine_query(question: str, docs: Sequence[Doc], previous: str = "") -> str:
    """Expand the query using the corpus's own vocabulary.

    No model call: the terms that make a statute searchable are already in the
    breadcrumbs of whatever came back, and borrowing them is both cheaper and
    more predictable than asking a model to guess synonyms.
    """
    base = (previous or question).strip()
    borrowed: list[str] = []
    seen = words(base)
    for doc in docs[:EXPAND_TOP]:
        for token in _WORD_RE.findall(doc.breadcrumb.lower()):
            if len(token) >= 5 and token not in seen and token not in GENERIC_TERMS:
                seen.add(token)
                borrowed.append(token)
            if len(borrowed) >= 6:
                break
        if len(borrowed) >= 6:
            break
    if not borrowed:
        return base
    return f"{base} {' '.join(borrowed)}"
