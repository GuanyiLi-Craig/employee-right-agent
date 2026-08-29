"""Sufficiency, context assembly and refinement -- without touching an index."""

from __future__ import annotations

import pytest

from rights_agent.retrieval import (
    COVERAGE_WEIGHT,
    SIMILARITY_WEIGHT,
    Doc,
    coverage,
    distinctive_words,
    format_context,
    refine_query,
    sufficiency,
)


def make_doc(
    citation: str = "s.1(1)",
    text: str = "An employer must make a guaranteed hours offer.",
    score: float = 0.5,
    breadcrumb: str = "Act > Part 1 Employment rights > s.1 Right to guaranteed hours",
    **kwargs: object,
) -> Doc:
    return Doc(
        id=f"l-{citation}",
        citation=citation,
        breadcrumb=breadcrumb,
        text=text,
        score=score,
        **kwargs,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# Doc
# --------------------------------------------------------------------------- #
def test_doc_round_trips_through_plain_dicts() -> None:
    """State must hold plain types: the checkpointer serialises it."""
    doc = make_doc(expanded=True, parent_id="p1", metadata={"page": 3})
    payload = doc.to_dict()
    assert isinstance(payload, dict) and all(isinstance(key, str) for key in payload)
    assert Doc.from_dict(payload) == doc


def test_doc_from_dict_tolerates_missing_keys() -> None:
    doc = Doc.from_dict({"id": "x"})
    assert doc.id == "x" and doc.score == 0.0 and doc.metadata == {}


# --------------------------------------------------------------------------- #
# Sufficiency
# --------------------------------------------------------------------------- #
def test_distinctive_words_drops_scaffolding() -> None:
    terms = distinctive_words("What does the document say about bereavement leave?")
    assert "bereavement" in terms
    assert "document" not in terms and "about" not in terms and "say" not in terms


def test_coverage_is_the_fraction_of_distinctive_terms_found() -> None:
    docs = [make_doc(text="bereavement leave is available to a bereaved person")]
    assert coverage("What about bereavement leave?", docs) == 1.0
    assert coverage("What about cryptocurrency mining?", docs) == 0.0


def test_coverage_counts_breadcrumbs_as_evidence() -> None:
    """The breadcrumb is full of exact terms; that is why it is embedded."""
    docs = [make_doc(text="(1) An employer must comply.", breadcrumb="Act > s.19 Bereavement leave")]
    assert coverage("bereavement leave", docs) == 1.0


def test_sufficiency_is_zero_with_no_documents() -> None:
    assert sufficiency([], "anything") == 0.0


def test_sufficiency_weights_coverage_above_similarity() -> None:
    """Similarity is poorly calibrated across queries; coverage is decisive."""
    assert COVERAGE_WEIGHT > SIMILARITY_WEIGHT
    high_similarity_no_coverage = [make_doc(text="unrelated words entirely", score=1.0)]
    low_similarity_full_coverage = [
        make_doc(text="bereavement leave for a bereaved person", score=0.05)
    ]
    question = "What about bereavement leave?"
    assert sufficiency(low_similarity_full_coverage, question) > sufficiency(
        high_similarity_no_coverage, question
    )


def test_sufficiency_uses_only_the_top_scores_for_similarity() -> None:
    docs = [make_doc(citation=f"s.{i}", score=score) for i, score in enumerate((0.9, 0.9, 0.9, 0.0))]
    assert sufficiency(docs, "no matching terms here", top=3) == pytest.approx(
        SIMILARITY_WEIGHT * 0.9, abs=1e-6
    )


def test_sufficiency_is_bounded() -> None:
    docs = [make_doc(text="bereavement leave", score=5.0)]
    assert 0.0 <= sufficiency(docs, "bereavement leave") <= 1.0


# --------------------------------------------------------------------------- #
# Context assembly
# --------------------------------------------------------------------------- #
def test_empty_docs_produce_an_empty_context() -> None:
    assert format_context([]) == ""


def test_context_is_never_empty_when_docs_exist() -> None:
    """Pitfall 3: a naive assembler skips the first block that does not fit."""
    giant = make_doc(text="x" * 36_000)
    context = format_context([giant], budget_chars=2_000, min_block_chars=400)
    assert context
    assert "s.1(1)" in context
    assert "truncated" in context


def test_context_truncates_rather_than_skipping_the_top_hit() -> None:
    docs = [make_doc(citation="s.1(1)", text="x" * 5_000), make_doc(citation="s.2(1)", text="short")]
    context = format_context(docs, budget_chars=1_200, min_block_chars=300)
    assert "s.1(1)" in context, "the top hit was dropped"


def test_context_deduplicates_by_citation() -> None:
    docs = [make_doc(citation="s.1"), make_doc(citation="s.1"), make_doc(citation="s.2")]
    context = format_context(docs, budget_chars=6_000)
    assert context.count("[s.1]") == 1
    assert "[s.2]" in context


def test_each_block_is_headed_by_its_citation_and_breadcrumb() -> None:
    context = format_context([make_doc()], budget_chars=6_000)
    first_line = context.splitlines()[0]
    assert first_line.startswith("[s.1(1)] Act > Part 1")


# --------------------------------------------------------------------------- #
# Refinement
# --------------------------------------------------------------------------- #
def test_refine_borrows_vocabulary_from_the_breadcrumbs() -> None:
    """No model call: the searchable terms are already in what came back."""
    docs = [make_doc(breadcrumb="Act > Part 1 Employment rights > s.19 Bereavement leave")]
    refined = refine_query("time off after a death", docs)
    assert refined.startswith("time off after a death")
    assert "bereavement" in refined.lower()


def test_refine_is_a_no_op_when_there_is_nothing_to_borrow() -> None:
    docs = [make_doc(breadcrumb="Act")]
    assert refine_query("a question", docs) == "a question"


def test_refine_builds_on_the_previous_rewrite() -> None:
    docs = [make_doc(breadcrumb="Act > s.19 Bereavement leave")]
    refined = refine_query("original", docs, previous="original expanded")
    assert refined.startswith("original expanded")


# --------------------------------------------------------------------------- #
# The candidate pool
# --------------------------------------------------------------------------- #
class _FakeCollection:
    """Records what was asked for, and answers with leaves that share parents."""

    def __init__(self, leaves: int) -> None:
        self.leaves = leaves
        self.requested: list[int] = []

    def count(self) -> int:
        return self.leaves

    def query(self, *, query_texts, n_results, where=None, include=None):
        self.requested.append(n_results)
        # Every three consecutive leaves belong to one provision, so widening
        # collapses them into a single block.
        ids, metadatas, distances = [], [], []
        for index in range(min(n_results, self.leaves)):
            provision = index // 3
            ids.append(f"l{index}")
            metadatas.append(
                {
                    "citation": f"s.{provision}({index % 3 + 1})",
                    "breadcrumb": f"Act > s.{provision}",
                    "raw_text": f"leaf {index}",
                    "parent_id": f"p{provision}",
                }
            )
            distances.append(0.1 + index * 0.01)
        return {
            "ids": [ids],
            "documents": [[m["raw_text"] for m in metadatas]],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def get(self, *, ids, include=None):
        return {
            "ids": list(ids),
            "documents": [f"provision {i}" for i in ids],
            "metadatas": [
                {"citation": f"s.{i[1:]}", "breadcrumb": f"Act > s.{i[1:]}", "raw_text": f"all of {i}"}
                for i in ids
            ],
        }


def _retriever_over(collection: _FakeCollection, settings) -> object:
    from rights_agent.retrieval import Retriever

    retriever = Retriever.__new__(Retriever)
    retriever.settings = settings
    retriever.leaves = collection
    retriever.parents = collection
    retriever.leaf_count = collection.count()
    retriever.embedder_name = "fake"

    class _Manifest:
        index_version = "fake+fake+fake"
        embedding_model = "fake"

    retriever.manifest = _Manifest()
    return retriever


def test_widening_does_not_shrink_the_result_below_k(isolated_settings) -> None:
    """Widening is subtractive: leaves in one provision collapse to one block.

    Fetching exactly ``k`` leaves therefore returned fewer than ``k`` blocks --
    two, where the top six leaves came from two provisions. The generator got a
    third of the context it was configured for and nothing said so.
    """
    from rights_agent.retrieval import OVERFETCH

    collection = _FakeCollection(leaves=600)
    retriever = _retriever_over(collection, isolated_settings)

    docs = retriever.search("a question", k=6)

    assert collection.requested == [6 * OVERFETCH], "the pool was not over-fetched"
    assert len(docs) == 6, f"asked for 6 blocks, got {len(docs)}"
    assert len({doc.citation for doc in docs}) == 6, "the blocks are not distinct"


def test_the_pool_never_exceeds_the_collection(isolated_settings) -> None:
    """Chroma raises rather than clamping when ``n_results`` is out of range,
    so a small corpus must not be asked for more leaves than it holds."""
    collection = _FakeCollection(leaves=4)
    retriever = _retriever_over(collection, isolated_settings)

    docs = retriever.search("a question", k=6)

    assert collection.requested == [4]
    assert len(docs) <= 4
