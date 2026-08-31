"""The document tree: breadcrumbs, citations, leaves."""

from __future__ import annotations

import pytest

from rights_agent.document.nodes import (
    KIND_DOCUMENT,
    KIND_HEADING,
    KIND_INSERTED,
    KIND_PART,
    KIND_SCHEDULE,
    KIND_SECTION,
    KIND_SUBSECTION,
    Node,
    leaves,
    provisions,
    stats,
)


@pytest.fixture
def act() -> Node:
    document = Node(KIND_DOCUMENT, title="Demonstration Act 2026")
    part = document.add(Node(KIND_PART, number="1", title="Employment rights"))
    heading = part.add(Node(KIND_HEADING, title="Zero hours workers, etc"))
    section = heading.add(Node(KIND_SECTION, number="1", title="Right to guaranteed hours"))
    section.add(Node(KIND_SUBSECTION, number="1", text="An employer must make an offer."))
    amending = section.add(Node(KIND_SUBSECTION, number="2", text="After section 27B insert—"))
    inserted = amending.add(
        Node(
            KIND_INSERTED,
            number="27BA",
            title="Guaranteed hours",
            host_document="Employment Rights Act 1996",
            inserted_by="1",
        )
    )
    inserted.add(Node(KIND_SUBSECTION, number="1", text="The offer must be in writing."))
    schedule = document.add(Node(KIND_SCHEDULE, number="1", title="Procedure"))
    paragraph = schedule.add(Node(KIND_SECTION, number="3", title="Applications"))
    paragraph.add(Node(KIND_SUBSECTION, number="2", text="An application must be in writing."))
    return document


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown node kind"):
        Node("paragraph")


def test_breadcrumb_runs_from_the_document_to_the_node(act: Node) -> None:
    leaf = act.children[0].children[0].children[0].children[0]
    breadcrumb = leaf.breadcrumb()
    assert breadcrumb == (
        "Demonstration Act 2026 > Part 1 Employment rights > Zero hours workers, etc > "
        "s.1 Right to guaranteed hours > (1)"
    )


def test_citation_of_a_plain_subsection(act: Node) -> None:
    leaf = act.children[0].children[0].children[0].children[0]
    assert leaf.citation() == "s.1(1)"


def test_inserted_provisions_cite_both_documents(act: Node) -> None:
    """Attributing an inserted provision to its host section points every
    citation at the wrong document."""
    inserted = next(node for node in act.walk() if node.kind == KIND_INSERTED)
    assert inserted.citation() == "Employment Rights Act 1996 s.27BA (as inserted by s.1)"
    child = inserted.children[0]
    assert child.citation() == "Employment Rights Act 1996 s.27BA(1) (as inserted by s.1)"


def test_inserted_citation_excludes_the_host_subsection_number(act: Node) -> None:
    """The host's ``(2)`` is not part of the inserted provision's number."""
    inserted = next(node for node in act.walk() if node.kind == KIND_INSERTED)
    assert "(2)" not in inserted.children[0].citation()


def test_schedule_paragraphs_are_not_cited_as_sections(act: Node) -> None:
    subsection = act.children[1].children[0].children[0]
    assert subsection.citation() == "Sch. 1 para. 3(2)"


def test_leaves_are_the_most_precise_citable_units(act: Node) -> None:
    citations = {node.citation() for node in leaves(act)}
    assert citations == {
        "s.1(1)",
        "Employment Rights Act 1996 s.27BA(1) (as inserted by s.1)",
        "Sch. 1 para. 3(2)",
    }
    # The subsection that merely introduces the inserted provision is not a leaf.
    assert "s.1(2)" not in citations


def test_provisions_are_sections_and_inserted_provisions(act: Node) -> None:
    kinds = {node.kind for node in provisions(act)}
    assert kinds == {KIND_SECTION, KIND_INSERTED}


def test_full_text_includes_descendants_but_own_text_does_not(act: Node) -> None:
    section = act.children[0].children[0].children[0]
    assert "An employer must make an offer." in section.full_text()
    assert "The offer must be in writing." in section.full_text()
    assert section.own_text() == ""


def test_walk_order_is_deterministic(act: Node) -> None:
    first = [node.label() for node in act.walk()]
    second = [node.label() for node in act.walk()]
    assert first == second


def test_stats_counts_every_kind(act: Node) -> None:
    counts = stats(act)
    assert counts[KIND_SECTION] == 2
    assert counts[KIND_SUBSECTION] == 4
    assert counts[KIND_INSERTED] == 1


def test_enclosing_provision_returns_self_for_a_provision(act: Node) -> None:
    section = act.children[0].children[0].children[0]
    assert section.enclosing_provision() is section
    assert section.children[0].enclosing_provision() is section


def test_ancestor_skips_self(act: Node) -> None:
    section = act.children[0].children[0].children[0]
    assert section.ancestor(KIND_SECTION) is None
    assert section.ancestor(KIND_PART) is act.children[0]
