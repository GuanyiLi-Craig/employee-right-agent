"""The parser, and the four traps it exists to survive."""

from __future__ import annotations

import textwrap

import pytest

from rights_agent.document.nodes import (
    KIND_HEADING,
    KIND_INSERTED,
    KIND_SCHEDULE,
    KIND_PART,
    KIND_SECTION,
    KIND_SUBSECTION,
    leaves,
    stats,
)
from rights_agent.document.parser import (
    ParserError,
    TreeExpectations,
    parse_text,
    validate_tree,
)

SMALL_EXPECTATIONS = TreeExpectations(min_sections=1, min_subsections=1, min_parts=1)


def _document(body: str, front_matter: str = "") -> str:
    """Wrap ``body`` in the minimum front matter the parser needs."""
    header = front_matter or textwrap.dedent(
        """\
                            Demonstration Act 2026

                                 2026 CHAPTER 14

        BE IT ENACTED by the King's most Excellent Majesty, as follows:—
        """
    )
    return f"{header}\n{body}"


# --------------------------------------------------------------------------- #
# Trap 1: the table of contents looks exactly like the body
# --------------------------------------------------------------------------- #
def test_table_of_contents_is_not_parsed_as_body() -> None:
    text = _document(
        body=textwrap.dedent(
            """\
                                             PART 1

                                       EMPLOYMENT RIGHTS

            1     Right to guaranteed hours

                (1)  An employer must make an offer.
            """
        ),
        front_matter=textwrap.dedent(
            """\
                            Demonstration Act 2026

                                    CONTENTS

                                     PART 1

                               EMPLOYMENT RIGHTS

            1     Right to guaranteed hours
            2     Guaranteed hours: number of hours
            3     Exceptions for temporary need

            BE IT ENACTED by the King's most Excellent Majesty, as follows:—
            """
        ),
    )
    tree = parse_text(text).tree
    counts = stats(tree)
    assert counts[KIND_SECTION] == 1, "the contents list was parsed as body sections"
    assert counts[KIND_PART] == 1


def test_front_matter_without_an_enacting_formula_falls_back_to_structure() -> None:
    """A contents list has section headings but never their subsections."""
    text = textwrap.dedent(
        """\
                            Demonstration Act 2026

                                    CONTENTS

                                     PART 1

                               EMPLOYMENT RIGHTS

        1     Right to guaranteed hours
        2     Guaranteed hours: number of hours
        \f
                                     PART 1

                               EMPLOYMENT RIGHTS

        1     Right to guaranteed hours

            (1)  An employer must make an offer.
        """
    )
    tree = parse_text(text).tree
    assert stats(tree)[KIND_SECTION] == 1


def test_unparseable_source_raises_rather_than_returning_an_empty_tree() -> None:
    with pytest.raises(ParserError, match="could not locate the start of the body"):
        parse_text("just some prose with no structure at all\n")


# --------------------------------------------------------------------------- #
# Trap 2: running headers, filtered case-sensitively
# --------------------------------------------------------------------------- #
def test_running_headers_are_filtered_without_eating_part_markers() -> None:
    """``Part 1 — …`` is furniture; ``PART 1`` is structure.  Only case tells them apart."""
    text = _document(
        textwrap.dedent(
            """\
                                             PART 1

                                       EMPLOYMENT RIGHTS

            1     Right to guaranteed hours

                (1)  An employer must make an offer.
            \f
            7      Demonstration Act 2026 (c. 14)
                   Part 1 — Employment rights

                (2)  The offer must be in writing.
            """
        )
    )
    tree = parse_text(text).tree
    counts = stats(tree)
    assert counts[KIND_PART] == 1, "the real PART marker was filtered as a header"
    assert counts[KIND_SUBSECTION] == 2
    breadcrumbs = {node.breadcrumb() for node in leaves(tree)}
    assert not any("Part 1 — Employment rights" in crumb for crumb in breadcrumbs)


def test_page_numbers_and_schedule_authority_lines_are_dropped() -> None:
    text = _document(
        textwrap.dedent(
            """\
                                             PART 1

                                       EMPLOYMENT RIGHTS

            1     Right to guaranteed hours

                (1)  An employer must make an offer.
            \f
                                            12

                                       Section 1

                (2)  The offer must be in writing.
            """
        )
    )
    tree = parse_text(text).tree
    text_of_leaves = " ".join(node.text for node in leaves(tree))
    assert "12" not in text_of_leaves.split()
    assert "Section 1" not in text_of_leaves


# --------------------------------------------------------------------------- #
# Trap 3: quoted material that mimics a cross-heading
# --------------------------------------------------------------------------- #
def test_real_cross_heading_is_recognised() -> None:
    text = _document(
        textwrap.dedent(
            """\
                                             PART 1

                                       EMPLOYMENT RIGHTS


                                    Zero hours workers, etc

            1     Right to guaranteed hours

                (1)  An employer must make an offer.
            """
        )
    )
    tree = parse_text(text).tree
    assert stats(tree)[KIND_HEADING] == 1
    leaf = leaves(tree)[0]
    assert "Zero hours workers, etc" in leaf.breadcrumb()


def test_quoted_pseudo_heading_is_not_a_cross_heading() -> None:
    """Position, not content: a quoted heading is followed by indented text."""
    text = _document(
        textwrap.dedent(
            """\
                                             PART 1

                                       EMPLOYMENT RIGHTS

            1     Right to guaranteed hours

                (1)  In the Employment Rights Act 1996, after section 27B insert—

                          Guaranteed hours: further provision

                          “27BA  Right to guaranteed hours

                          (1)  An employer must secure that every worker is offered hours.
            """
        )
    )
    tree = parse_text(text).tree
    assert stats(tree).get(KIND_HEADING, 0) == 0, "quoted material was promoted to a heading"
    assert stats(tree)[KIND_INSERTED] == 1


def test_wrapped_clause_ending_in_a_smart_quote_is_not_a_heading() -> None:
    """The exact shape that bites: ``…of the employer.”`` before a section line."""
    text = _document(
        textwrap.dedent(
            """\
                                             PART 1

                                       EMPLOYMENT RIGHTS

            1     Right to guaranteed hours

                (1)  In the Employment Rights Act 1996, after section 27B insert—

                          “27BA  Right to guaranteed hours

                          (1)  In section 27BA, references to guaranteed hours include
                              references to an offer made by an agent of the employer.”


            2     Guaranteed hours: number of hours

                (1)  An employer must state the number of hours.
            """
        )
    )
    tree = parse_text(text).tree
    assert stats(tree).get(KIND_HEADING, 0) == 0
    assert stats(tree)[KIND_SECTION] == 2


# --------------------------------------------------------------------------- #
# Trap 4: inserted provisions
# --------------------------------------------------------------------------- #
def test_inserted_provision_is_nested_and_names_its_host() -> None:
    text = _document(
        textwrap.dedent(
            """\
                                             PART 1

                                       EMPLOYMENT RIGHTS

            1     Right to guaranteed hours

                (1)  An employer must make an offer.

                (2)  In the Employment Rights Act 1996, after section 27B insert—

                          “27BA  Right to guaranteed hours

                          (1)  An employer must secure that hours are offered.

                          (2)  Regulations may make provision about the manner of an offer.

                (3)  This section applies to agency workers.
            """
        )
    )
    tree = parse_text(text).tree
    inserted = next(node for node in tree.walk() if node.kind == KIND_INSERTED)
    assert inserted.host_document == "Employment Rights Act 1996"
    assert inserted.inserted_by == "1"
    assert [child.number for child in inserted.children] == ["1", "2"]
    citations = {node.citation() for node in leaves(tree)}
    assert "Employment Rights Act 1996 s.27BA(1) (as inserted by s.1)" in citations
    # (3) belongs to the host section again, not to the inserted provision.
    assert "s.1(3)" in citations


def test_consecutive_inserted_provisions_inherit_the_host_document() -> None:
    text = _document(
        textwrap.dedent(
            """\
                                             PART 1

                                       EMPLOYMENT RIGHTS

            1     Amendments

                (1)  In the Employment Rights Act 1996, after section 27B insert—

                          “27BA  Guaranteed hours

                          (1)  An employer must secure that hours are offered.

                          “27BB  Guaranteed hours: supplementary

                          (1)  In section 27BA, references include an agent's offer.
            """
        )
    )
    tree = parse_text(text).tree
    hosts = {
        node.host_document for node in tree.walk() if node.kind == KIND_INSERTED
    }
    assert hosts == {"Employment Rights Act 1996"}


# --------------------------------------------------------------------------- #
# Validation gate
# --------------------------------------------------------------------------- #
def test_validation_reports_every_failure_at_once() -> None:
    text = _document(
        textwrap.dedent(
            """\
                                             PART 1

                                       EMPLOYMENT RIGHTS

            1     Right to guaranteed hours

                (1)  An employer must make an offer.
            """
        )
    )
    tree = parse_text(text).tree
    with pytest.raises(ParserError) as excinfo:
        validate_tree(tree)
    message = str(excinfo.value)
    assert "section: found 1" in message
    assert "subsection: found 1" in message
    assert "part: found 1" in message


def test_validation_passes_on_a_small_well_formed_tree() -> None:
    text = _document(
        textwrap.dedent(
            """\
                                             PART 1

                                       EMPLOYMENT RIGHTS

            1     Right to guaranteed hours

                (1)  An employer must make an offer.
            """
        )
    )
    tree = parse_text(text).tree
    counts = validate_tree(tree, SMALL_EXPECTATIONS)
    assert counts[KIND_SECTION] == 1


# --------------------------------------------------------------------------- #
# The committed corpus
# --------------------------------------------------------------------------- #
def test_generated_corpus_clears_the_gate(corpus_text: str) -> None:
    result = parse_text(corpus_text)
    counts = validate_tree(result.tree)
    assert counts[KIND_SECTION] >= 150
    assert counts[KIND_SUBSECTION] >= 900
    assert counts[KIND_PART] >= 5
    assert counts[KIND_INSERTED] >= 6


def test_generated_corpus_leaf_count_tracks_subsections(corpus_text: str) -> None:
    tree = parse_text(corpus_text).tree
    counts = stats(tree)
    assert abs(len(leaves(tree)) - counts[KIND_SUBSECTION]) <= 0.10 * counts[KIND_SUBSECTION]


def test_corpus_generation_is_deterministic() -> None:
    from rights_agent.tools.corpus import render

    assert render() == render()


# --------------------------------------------------------------------------- #
# Schedule paragraphs, and the provisions they insert
# --------------------------------------------------------------------------- #
def test_an_indented_schedule_paragraph_is_a_paragraph_not_prose() -> None:
    """Body sections start at column 0; schedule paragraphs are set in a space
    or two.  Requiring column 0 dropped all 180 of the real Act's schedule
    paragraphs into the preceding block's text."""
    text = _document(
        textwrap.dedent(
            """\
            1  Opening section

                (1)  This section introduces Schedule 4.

                                            SCHEDULE 4                     Section 1

                                    PAY AND CONDITIONS OF SUPPORT STAFF

             2          In the Education Act 2002, after Part 8 insert—

                          “148A  The Negotiating Body

                            (1)  There is to be a body known as the Negotiating Body.
            """
        )
    )
    tree = parse_text(text).tree
    schedule = next(node for node in tree.walk() if node.kind == KIND_SCHEDULE)
    paragraph = next(
        node for node in schedule.walk() if node.kind == KIND_SECTION and node.number == "2"
    )
    assert paragraph.title.startswith("In the Education Act 2002")


def test_a_schedule_paragraph_attributes_the_provision_it_inserts() -> None:
    """The host Act is named on the paragraph's own line, so it lands in the
    paragraph's title rather than its text -- and an inserted provision with no
    host is cited as "the host Act", naming nothing a reader could look up."""
    text = _document(
        textwrap.dedent(
            """\
            1  Opening section

                (1)  This section introduces Schedule 4.

                                            SCHEDULE 4                     Section 1

                                    PAY AND CONDITIONS OF SUPPORT STAFF

             2          In the Education Act 2002, after Part 8 insert—

                          “148A  The Negotiating Body

                            (1)  There is to be a body known as the Negotiating Body.
            """
        )
    )
    tree = parse_text(text).tree
    inserted = next(node for node in tree.walk() if node.kind == KIND_INSERTED)
    assert inserted.host_document == "Education Act 2002"
    # Cited as the schedule paragraph that did the inserting, not as s.2 --
    # which exists, and is a different provision entirely.
    assert "Education Act 2002 s.148A(1) (as inserted by Sch. 4 para. 2)" in {
        node.citation() for node in leaves(tree)
    }


def test_a_statute_name_with_a_qualifier_in_brackets_is_read_whole() -> None:
    """``Trade Union and Labour Relations (Consolidation) Act 1992``: a token
    class without brackets stopped at the qualifier and matched nothing."""
    text = _document(
        textwrap.dedent(
            """\
            1  Amendments

                (1)  The Trade Union and Labour Relations (Consolidation) Act 1992 is amended
                     in accordance with subsections (2) to (6).

                (2)  In Part 3, before section 137 insert—

                          “136A  Right to a statement

                            (1)  A worker is entitled to a statement.
            """
        )
    )
    tree = parse_text(text).tree
    inserted = next(node for node in tree.walk() if node.kind == KIND_INSERTED)
    assert inserted.host_document == "Trade Union and Labour Relations (Consolidation) Act 1992"


def test_a_schedule_host_is_resolved_through_its_authorising_section() -> None:
    """A schedule that amends another Act names it once, in the body section
    that introduces the schedule, and never again inside the schedule."""
    text = _document(
        textwrap.dedent(
            """\
            1  Seafarers' wages

                (1)  Schedule 5 amends the Seafarers' Wages Act 2023.

                                            SCHEDULE 5                     Section 1

                                    SEAFARERS' WAGES AND WORKING CONDITIONS

             9          After section 4 insert—

                          “4A  Remuneration regulations

                            (1)  Regulations may specify requirements.
            """
        )
    )
    tree = parse_text(text).tree
    inserted = next(node for node in tree.walk() if node.kind == KIND_INSERTED)
    assert inserted.host_document == "Seafarers' Wages Act 2023"
    schedule = next(node for node in tree.walk() if node.kind == KIND_SCHEDULE)
    assert schedule.authorising_section == "1"


def test_a_wrapped_cross_reference_is_not_a_new_subsection() -> None:
    """Statutory cross-references wrap, and they wrap across the number.

    Read as an opener, ``(2) of that section.`` became a subsection whose entire
    text was ``of that section.`` -- a chunk that says nothing, is retrievable,
    and duly turned up as cited evidence in an answer. 260 of the real Act's
    2,141 leaves were fragments of this kind.
    """
    text = _document(
        textwrap.dedent(
            """\
            1  Right to guaranteed hours

                (1)  An employer must make an offer that complies with subsection
                     (2) of that section.

                (2)  Regulations may make provision about the manner of an offer.
            """
        )
    )
    tree = parse_text(text).tree
    section = next(node for node in tree.walk() if node.kind == KIND_SECTION)
    subsections = [child for child in section.children if child.kind == KIND_SUBSECTION]

    assert [child.number for child in subsections] == ["1", "2"]
    assert "of that section" in (subsections[0].text or ""), (
        "the wrapped remainder belongs to the subsection it continues"
    )
    assert subsections[1].text.startswith("Regulations may make provision")


def test_a_genuine_subsection_after_a_finished_sentence_still_opens() -> None:
    """The guard needs both signals. A real opener follows a completed
    sentence, however deeply the layout happens to indent it."""
    text = _document(
        textwrap.dedent(
            """\
            1  Right to guaranteed hours

                (1)  An employer must make a guaranteed hours offer.

                      (2)  Regulations may make provision about the manner of an offer.
            """
        )
    )
    tree = parse_text(text).tree
    section = next(node for node in tree.walk() if node.kind == KIND_SECTION)
    assert [c.number for c in section.children if c.kind == KIND_SUBSECTION] == ["1", "2"]
