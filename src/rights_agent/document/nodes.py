"""The document tree.

``breadcrumb()`` is the single most important method in the project: it is what
turns an orphaned clause back into a findable one.  Everything the hierarchical
pipeline does downstream is a consequence of prepending it to the embedded text.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

#: Structural kinds, ordered from the root down.  ``inserted`` is a provision
#: that this document inserts into *another* document -- it looks like a section
#: but must never be cited as one of ours (pitfall 7).
KIND_DOCUMENT = "document"
KIND_PART = "part"
KIND_SCHEDULE = "schedule"
KIND_HEADING = "heading"
KIND_SECTION = "section"
KIND_INSERTED = "inserted"
KIND_SUBSECTION = "subsection"

KINDS = (
    KIND_DOCUMENT,
    KIND_PART,
    KIND_SCHEDULE,
    KIND_HEADING,
    KIND_SECTION,
    KIND_INSERTED,
    KIND_SUBSECTION,
)

#: Kinds that own a citable provision -- the targets of small-to-big expansion.
PROVISION_KINDS = frozenset({KIND_SECTION, KIND_INSERTED})


@dataclass(eq=False)
class Node:
    """One structural element of the corpus.

    ``text`` is the node's *own* text only, excluding children; ``full_text()``
    is the recursive form.  Keeping them separate is what lets a leaf be
    embedded precisely and then widened to its parent provision on demand.
    """

    kind: str
    number: str = ""
    title: str = ""
    text: str = ""
    page: int = 0
    parent: Node | None = field(default=None, repr=False, compare=False)
    children: list[Node] = field(default_factory=list, repr=False)
    #: For ``inserted`` provisions: the document they are inserted into and the
    #: section of *this* document doing the inserting.
    host_document: str = ""
    inserted_by: str = ""
    #: When the inserting provision is a schedule paragraph rather than a body
    #: section, the schedule's number. ``inserted_by`` alone would render
    #: "as inserted by s.1" for Schedule 4 paragraph 1 -- naming a real but
    #: unrelated section of the enacting Act.
    inserted_by_schedule: str = ""
    #: On a schedule: the body section that introduces it, read from the
    #: ``SCHEDULE 5    Section 56`` marker. That section is where the host Act a
    #: schedule amends is named, so it is the only route to attributing the
    #: provisions the schedule inserts.
    authorising_section: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown node kind {self.kind!r}; expected one of {KINDS}")

    # ---- tree construction -------------------------------------------------
    def add(self, child: Node) -> Node:
        """Attach ``child`` and return it."""
        child.parent = self
        self.children.append(child)
        return child

    # ---- navigation --------------------------------------------------------
    def path(self) -> list[Node]:
        """Root → self."""
        chain: list[Node] = []
        node: Node | None = self
        while node is not None:
            chain.append(node)
            node = node.parent
        chain.reverse()
        return chain

    def depth(self) -> int:
        return len(self.path()) - 1

    def walk(self) -> Iterator[Node]:
        """Depth-first, self first.  Order is deterministic, which is what makes
        document-order ordinals stable between builds."""
        yield self
        for child in self.children:
            yield from child.walk()

    def ancestor(self, *kinds: str) -> Node | None:
        """Nearest ancestor (excluding self) of any of ``kinds``."""
        node = self.parent
        while node is not None:
            if node.kind in kinds:
                return node
            node = node.parent
        return None

    def enclosing_provision(self) -> Node | None:
        """The section or inserted provision this node belongs to.

        Returns ``self`` when the node *is* the provision.
        """
        if self.kind in PROVISION_KINDS:
            return self
        return self.ancestor(*PROVISION_KINDS)

    def is_leaf(self) -> bool:
        """True when nothing below this node would be more precise to retrieve."""
        return not self.children

    # ---- rendering ---------------------------------------------------------
    def label(self) -> str:
        """Short human label, e.g. ``Part 1 Employment rights``, ``s.25 …``, ``(3)``."""
        if self.kind == KIND_DOCUMENT:
            return self.title or "Document"
        if self.kind == KIND_PART:
            return _join(f"Part {self.number}", self.title)
        if self.kind == KIND_SCHEDULE:
            return _join(f"Schedule {self.number}", self.title)
        if self.kind == KIND_HEADING:
            return self.title
        if self.kind == KIND_SECTION:
            # Schedule paragraphs carry no heading of their own, so the label is
            # just the number -- ``_join`` drops the empty title.
            return _join(f"s.{self.number}", self.title)
        if self.kind == KIND_INSERTED:
            host = f" [{self.host_document}]" if self.host_document else ""
            return _join(f"s.{self.number}{host}", self.title)
        return f"({self.number})"

    def breadcrumb(self) -> str:
        """``" > ".join`` of every label from the root to self.

        This string is prepended to the text that gets embedded.  Not stored
        beside it -- embedded *with* it.  The topic, the jurisdiction and the
        provision number all live in the ancestors, so a leaf without its
        breadcrumb retrieves for nothing.
        """
        return " > ".join(node.label() for node in self.path())

    def citation(self) -> str:
        """Short quotable form, e.g. ``s.25(1)``.

        For provisions this document *inserts* into another one, the citation
        names the host document and the inserting section, because attributing
        them to the host section of this Act would point every citation at the
        wrong document.
        """
        provision = self.enclosing_provision()
        if provision is None:
            # Above the provision level: cite the structural container.
            if self.kind == KIND_PART:
                return f"Part {self.number}"
            if self.kind == KIND_SCHEDULE:
                return f"Sch. {self.number}"
            if self.kind == KIND_HEADING:
                part = self.ancestor(KIND_PART, KIND_SCHEDULE)
                return f"{part.citation()} — {self.title}" if part else self.title
            return self.title or self.kind

        suffix = "".join(f"({n.number})" for n in self._subsection_chain())
        if provision.kind == KIND_INSERTED:
            host = provision.host_document or "the host Act"
            by = ""
            if provision.inserted_by and provision.inserted_by_schedule:
                by = (
                    f" (as inserted by Sch. {provision.inserted_by_schedule}"
                    f" para. {provision.inserted_by})"
                )
            elif provision.inserted_by:
                by = f" (as inserted by s.{provision.inserted_by})"
            return f"{host} s.{provision.number}{suffix}{by}"
        schedule = provision.ancestor(KIND_SCHEDULE)
        if schedule is not None:
            # Inside a Schedule a column-0 numbered provision is a paragraph,
            # not a section, and must not be cited as one.
            return f"Sch. {schedule.number} para. {provision.number}{suffix}"
        return f"s.{provision.number}{suffix}"

    def _subsection_chain(self) -> list[Node]:
        """Subsection nodes strictly *below* the enclosing provision.

        Sliced after the provision rather than filtered over the whole path: an
        inserted provision can sit inside a subsection of its host section, and
        that host subsection's number is not part of the inserted provision's
        citation.
        """
        chain = self.path()
        for index in range(len(chain) - 1, -1, -1):
            if chain[index].kind in PROVISION_KINDS:
                chain = chain[index + 1 :]
                break
        return [n for n in chain if n.kind == KIND_SUBSECTION]

    def full_text(self) -> str:
        """Self plus every descendant, in document order."""
        parts: list[str] = []
        for node in self.walk():
            own = node.text.strip()
            if node is not self and node.kind == KIND_SUBSECTION:
                own = f"({node.number}) {own}".strip() if own else ""
            if own:
                parts.append(own)
            elif node is not self and node.title and node.kind != KIND_SUBSECTION:
                parts.append(node.label())
        return "\n".join(parts).strip()

    def own_text(self) -> str:
        """This node's text with its own number prefixed, for prompt assembly."""
        text = self.text.strip()
        if self.kind == KIND_SUBSECTION and text:
            return f"({self.number}) {text}"
        return text

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Node(kind={self.kind!r}, number={self.number!r}, title={self.title[:40]!r}, children={len(self.children)})"


def _join(*parts: str) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())


# --------------------------------------------------------------------------- #
# Comparing citations
# --------------------------------------------------------------------------- #
#: Stripped first, so the *inserted* provision is the subject rather than the
#: section doing the inserting: "ERA 1996 s.80EA (as inserted by s.23)" is about
#: s.80EA, and matching on s.23 would attribute it to the wrong document.
_AS_INSERTED = re.compile(r"\s*\(\s*as inserted by[^)]*\)\s*", re.IGNORECASE)
_SCHEDULE_REF = re.compile(
    r"\bsch(?:edule)?\.?\s*(\w+)\s*,?\s*para(?:graph)?s?\.?\s*(\d+[A-Z]*)", re.IGNORECASE
)
_SECTION_REF = re.compile(r"\bs(?:ection)?s?\.?\s*(\d+[A-Z]*)", re.IGNORECASE)


def canonical_citation(citation: str) -> str:
    """Reduce a citation to the provision it identifies.

    Citations are compared, not just rendered, and the two ends rarely agree on
    depth. A retrieved block is headed with the provision (``s.19``); a model
    citing it will very reasonably name the subsection it used (``s.19(4)``), or
    write ``s.80EA(1)`` where the header said
    ``Employment Rights Act 1996 s.80EA (as inserted by s.23)``. Every one of
    those is *correct*, and exact string matching scores them all zero.

    So both sides are reduced to the provision -- ``s.19``, ``s.80EA``,
    ``Sch. 1 para. 3`` -- before comparison. The check keeps its teeth: a
    citation naming a provision that is not in the context still does not
    resolve, which is the failure that matters.
    """
    text = _AS_INSERTED.sub(" ", citation or "").strip()
    if not text:
        return ""
    schedule = _SCHEDULE_REF.search(text)
    if schedule:
        return f"sch.{schedule.group(1).lower()} para.{schedule.group(2).lower()}"
    section = _SECTION_REF.search(text)
    if section:
        return f"s.{section.group(1).lower()}"
    return re.sub(r"\s+", " ", text).strip().lower()


def citation_resolves(citation: str, available: Iterable[str]) -> bool:
    """Whether ``citation`` names a provision present in ``available``."""
    key = canonical_citation(citation)
    return bool(key) and key in {canonical_citation(item) for item in available}


def stats(tree: Node) -> Counter[str]:
    """Count nodes by kind.  Used by the parser's validation gate."""
    return Counter(node.kind for node in tree.walk())


def orphan_subsections(tree: Node) -> list[Node]:
    """Subsections with no provision above them.

    They exist because the parser did not recognise the provision that opens
    them -- a layout it has not been taught. The text is real, but nothing can
    say *which* provision it belongs to.
    """
    return [
        node
        for node in tree.walk()
        if node.kind == KIND_SUBSECTION and node.enclosing_provision() is None
    ]


def leaves(tree: Node) -> list[Node]:
    """Nodes to embed in the searched collection.

    A leaf is the most precise citable unit: a subsection with no children, or a
    provision that has no subsections at all.  Structural containers (parts,
    cross-headings) are never leaves -- their text lives in their descendants.

    **Orphan subsections are excluded.** A chunk that cannot name its provision
    cannot be cited, and an answer that cannot cite is an answer nobody can
    check -- which is the entire argument for the hierarchical pipeline. Keeping
    them would put uncitable text in the collection whose purpose is citable
    text. They are counted instead: :func:`orphan_subsections` and the
    ``orphan_subsections`` figure in the index manifest, so the parser's
    remaining blind spot is a number somebody can act on rather than silently
    bad data.
    """
    out: list[Node] = []
    for node in tree.walk():
        if node.kind == KIND_SUBSECTION and node.is_leaf():
            if node.enclosing_provision() is None:
                continue
            out.append(node)
        elif node.kind in PROVISION_KINDS and not any(
            c.kind == KIND_SUBSECTION for c in node.children
        ):
            out.append(node)
    return out


def provisions(tree: Node) -> list[Node]:
    """Sections and inserted provisions -- the parent collection's rows."""
    return [node for node in tree.walk() if node.kind in PROVISION_KINDS]
