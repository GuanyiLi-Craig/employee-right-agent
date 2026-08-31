"""The two pipelines, and what separates them."""

from __future__ import annotations

import pytest

from rights_agent.config import CHUNK_CHARS, OVERLAP_CHARS
from rights_agent.document.nodes import KIND_INSERTED
from rights_agent.document.nodes import leaves as tree_leaves
from rights_agent.document.parser import parse_text
from rights_agent.pipelines.hierarchical import build_rows
from rights_agent.pipelines.simple import fixed_window_chunks


# --------------------------------------------------------------------------- #
# Fixed windows
# --------------------------------------------------------------------------- #
def test_windows_cover_the_whole_text() -> None:
    text = "".join(f"{index:04d}." for index in range(600))  # 3000 chars
    chunks = fixed_window_chunks(text, size=1_000, overlap=150)
    assert chunks[0][0] == 0
    assert chunks[-1][0] + 1_000 >= len(text)
    assert all(chunk for _, chunk in chunks)


def test_window_count_matches_the_arithmetic() -> None:
    """§8.2: within 10% of ``len(text) / (size - overlap)``."""
    text = "x" * 100_000
    chunks = fixed_window_chunks(text, size=CHUNK_CHARS, overlap=OVERLAP_CHARS)
    expected = len(text) / (CHUNK_CHARS - OVERLAP_CHARS)
    assert abs(len(chunks) - expected) <= 0.10 * expected


def test_overlap_reduces_but_does_not_remove_boundary_loss() -> None:
    text = "A" * 500 + "THE DEFINITION SPANS HERE" + "B" * 500
    without = fixed_window_chunks(text, size=510, overlap=0)
    with_overlap = fixed_window_chunks(text, size=510, overlap=200)
    assert not any("THE DEFINITION SPANS HERE" in chunk for _, chunk in without)
    assert any("THE DEFINITION SPANS HERE" in chunk for _, chunk in with_overlap)


def test_blank_windows_are_skipped() -> None:
    assert fixed_window_chunks(" " * 5_000, size=1_000, overlap=100) == []


def test_short_text_yields_one_window() -> None:
    assert len(fixed_window_chunks("a short document", size=1_000, overlap=150)) == 1


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, 150), (100, -1)])
def test_invalid_window_geometry_is_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        fixed_window_chunks("text", size=size, overlap=overlap)


# --------------------------------------------------------------------------- #
# Hierarchical rows
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rows(corpus_text: str):
    return build_rows(parse_text(corpus_text), "test-version")


def test_every_leaf_document_starts_with_its_breadcrumb(rows) -> None:
    """The breadcrumb IS the embedded text, not metadata stored beside it."""
    leaves, _ = rows
    for document, metadata in zip(leaves["documents"], leaves["metadatas"], strict=True):
        assert document.startswith(metadata["breadcrumb"])
        assert document != metadata["breadcrumb"], "the leaf text is missing"


def test_leaf_metadata_carries_what_retrieval_and_audit_need(rows) -> None:
    leaves, _ = rows
    required = {
        "citation",
        "breadcrumb",
        "parent_id",
        "part",
        "section_number",
        "section_title",
        "kind",
        "page",
        "chars",
        "raw_text",
        "index_version",
    }
    assert required <= set(leaves["metadatas"][0])


def test_raw_text_excludes_the_breadcrumb(rows) -> None:
    """Prompt assembly puts the breadcrumb in the citation line, once."""
    leaves, _ = rows
    for metadata in leaves["metadatas"][:50]:
        assert metadata["breadcrumb"] not in metadata["raw_text"]


def test_ids_carry_a_document_order_ordinal_and_are_unique(rows) -> None:
    leaves, parents = rows
    for collection, prefix in ((leaves, "l"), (parents, "p")):
        ids = collection["ids"]
        assert len(ids) == len(set(ids))
        assert all(identifier.startswith(prefix) for identifier in ids)
        ordinals = [int(identifier[1:].split("::")[0]) for identifier in ids]
        assert ordinals == sorted(ordinals), "ids must follow document order"


def test_every_leaf_points_at_a_parent_row(rows) -> None:
    leaves, parents = rows
    parent_ids = set(parents["ids"])
    for metadata in leaves["metadatas"]:
        assert metadata["parent_id"] in parent_ids


def test_parent_rows_contain_their_childrens_text(rows) -> None:
    leaves, parents = rows
    by_id = dict(zip(parents["ids"], parents["metadatas"], strict=True))
    sample = leaves["metadatas"][100]
    parent = by_id[sample["parent_id"]]
    assert sample["raw_text"].strip("() 0123456789")[:40] in parent["raw_text"]


def test_inserted_provisions_appear_as_their_own_parent_rows(rows) -> None:
    _, parents = rows
    inserted = [m for m in parents["metadatas"] if m["kind"] == KIND_INSERTED]
    assert inserted, "the corpus is expected to contain inserted provisions"
    for metadata in inserted:
        assert "as inserted by" in metadata["citation"]
        assert metadata["host_document"]


def test_row_counts_match_the_tree(corpus_text: str, rows) -> None:
    tree = parse_text(corpus_text).tree
    leaves, _ = rows
    assert len(leaves["ids"]) == len(tree_leaves(tree))


def test_building_twice_produces_identical_rows(corpus_text: str) -> None:
    """Two consecutive ingests must produce identical chunk ids (§9.5)."""
    first = build_rows(parse_text(corpus_text), "v1")
    second = build_rows(parse_text(corpus_text), "v1")
    assert first[0]["ids"] == second[0]["ids"]
    assert first[0]["documents"] == second[0]["documents"]
    assert first[1]["ids"] == second[1]["ids"]


def test_index_version_is_stamped_on_every_row(rows) -> None:
    """Without it you cannot answer "which index produced this" six months on."""
    leaves, parents = rows
    for collection in (leaves, parents):
        assert all(m["index_version"] == "test-version" for m in collection["metadatas"])
