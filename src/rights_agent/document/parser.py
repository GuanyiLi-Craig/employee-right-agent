"""Layout text → :class:`~rights_agent.document.nodes.Node` tree.

The parser reads ``pdftotext -layout`` output because indentation is a
structural signal: a subsection is recognised by being indented under a
column-0 section line, and losing that indentation loses the tree.

Markers are detected in this precedence order:

===============  =======================================  ==========================
Marker           Typical form                             Rule
===============  =======================================  ==========================
Part             ``PART 1`` alone on a line, ALL CAPS     next caps line is its title
Schedule         ``SCHEDULE 1``                           may contain its own Parts
Section          ``25   Right not to be unfairly …``      number in column 0
Inserted         ``  “27BA  Right to guaranteed hours``   letter suffix, indented
Subsection       ``    (1)  An employer must …``          indented, parenthesised
Cross-heading    ``       Dismissal``                     followed by a section line
===============  =======================================  ==========================

Four traps, each of which produces a plausible-looking but wrong tree if missed:

1. **The table of contents looks exactly like the body.** Front matter is
   skipped up to the enacting formula; without that you build a second, empty
   copy of every section.
2. **Running headers repeat on every page.** They are filtered
   *case-sensitively*: the header ``Part 1 — Employment rights`` is mixed case
   and the real marker ``PART 1`` is not, so a case-insensitive filter silently
   deletes every Part.
3. **Quoted material mimics structure.** A real cross-heading is immediately
   followed by a column-0 section line; a quoted one is followed by indented
   text. Disambiguation is by position, not content.
4. **Inserted provisions must not be attributed to their host section.** A
   number with a letter suffix at non-zero indent opens a provision inserted
   into *another* document; its subsections are nested under it and its
   citation names both documents.

One deliberate simplification: paragraph markers (``(a)``, ``(i)``) are folded
into the text of the subsection that contains them rather than becoming nodes.
A paragraph lifted out of its subsection is rarely meaningful on its own, and
folding keeps the leaf count aligned with the subsection count.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from rights_agent.document.nodes import (
    KIND_DOCUMENT,
    KIND_HEADING,
    KIND_INSERTED,
    KIND_PART,
    KIND_SCHEDULE,
    KIND_SECTION,
    KIND_SUBSECTION,
    Node,
    orphan_subsections,
    stats,
)
from rights_agent.log import get_logger

log = get_logger("document.parser")

#: Bumped whenever the tree shape changes; participates in ``index_version``.
PARSER_NAME = "layout-statute"

QUOTE_CHARS = "“”‘’\"'"

# --------------------------------------------------------------------------- #
# Line patterns.  Every pattern that could match a running header is anchored
# and case-sensitive on purpose (trap 2).
# --------------------------------------------------------------------------- #
RE_PART = re.compile(r"^\s*PART\s+([0-9]+|[IVXLCDM]+)\s*$")
#: ``SCHEDULE 11`` alone, or -- as legislation.gov.uk renders it -- followed on
#: the same line by the section that introduces it:
#: ``SCHEDULE 11                Section 149(2) and (3)``.
RE_SCHEDULE = re.compile(
    r"^\s*SCHEDULE\s+([0-9]+|[IVXLCDM]+)\s*(?:Sections?\s+([\d(),\s\band]+))?\s*$"
)
#: Note the parentheses in the token class. Statute short titles contain them --
#: ``Trade Union and Labour Relations (Consolidation) Act 1992`` -- and a class
#: of ``[\w'-]`` alone silently failed on every Act with a qualifier in its name.
_ACT_NAME = r"[A-Z][\w'’\-()]*(?:\s+[\w'’\-()]+){0,10}\s+Act\s+\d{4}"
#: The authorising section names the host Act that a Schedule amends:
#: ``Schedule 5 amends the Seafarers' Wages Act 2023.`` and ``Schedule 6 amends
#: Schedule A1 to the Trade Union and Labour Relations (Consolidation) Act
#: 1992``. Inside such a Schedule the host is never named again, so an inserted
#: provision there can only be attributed by following this cross-reference.
RE_SCHEDULE_AMENDS = re.compile(rf"\bamends\b[^.]*?\bthe\s+({_ACT_NAME})")

RE_SECTION = re.compile(r"^(\d+[A-Z]*)\s{2,}(\S.*)$")
#: An inserted provision: a number with a letter suffix, then its heading.
#:
#: One space is enough -- the real Act sets ``148B Matters within the SSSNB's
#: remit`` and ``27BA Right for qualifying workers to be offered guaranteed
#: hours`` with a single space, and requiring two lost every one of them into the
#: previous subsection's text. One space alone would also match a wrapped
#: fragment like ``2A of the Employment Rights Act 1996``, so
#: :func:`_is_inserted_provision` adds the conditions that tell a heading from a
#: continuation line.
RE_INSERTED = re.compile(r"^[\s“”\"']*(\d+[A-Z]+)\s+(\S.*)$")
RE_SUBSECTION = re.compile(r"^\((\d+)\)\s+(.*)$")
#: A provision that opens with its first subsection on the same line:
#: ``1    (1) The Secretary of State may make a scheme``.
#:
#: This is how legislation.gov.uk sets Schedule paragraphs, and missing it was
#: expensive: the opener became continuation text, every subsection under it had
#: no provision to belong to, and 39% of the real Act's leaves ended up with the
#: citation ``"subsection"`` -- unciteable chunks in the index whose whole
#: purpose is to be citable.
RE_PROVISION_WITH_SUBSECTION = re.compile(r"^(\d+[A-Z]*)\s+\((\d+)\)\s+(\S.*)$")

#: How far a provision opener may be indented. Body sections sit at column 0;
#: Schedule paragraphs are typically indented by one or two.
MAX_PROVISION_INDENT = 3
RE_PARAGRAPH = re.compile(r"^\(([a-z]{1,3}|[ivxlcdm]{1,6})\)\s+(.*)$")
#: The Act being amended. Two shapes, because real Acts use both:
#:
#: * ``In the Employment Rights Act 1996, after section 27B insert—`` -- named in
#:   the same subsection that does the inserting.
#: * ``The Equality Act 2010 is amended as follows.`` in subsection (1), then
#:   ``after section 40A insert—`` several subsections later. This is the common
#:   one, and matching only the first shape left 19 of 20 inserted provisions in
#:   the real Act citing "the host Act" -- which names no document at all.
RE_INSERT_DIRECTIVE = re.compile(
    rf"\b(?:[IiOo][nf]|of)\s+the\s+({_ACT_NAME})"
    rf"|\b(?:The|the)\s+({_ACT_NAME})\s+is\s+amended"
)
#: The enacting formula, allowing for a drop cap.
#:
#: Typeset Acts set the first letter as a large initial, and ``pdftotext``
#: renders that as the letter, a run of spaces, then the rest of the word:
#: ``B     E IT ENACTED by the King's most Excellent Majesty``. Matching
#: ``BE IT ENACTED`` literally misses every real Act and falls back to a
#: structural guess -- which works, but locates the body less precisely.
RE_ENACTING = re.compile(r"^\s*B\s*E\s+IT\s+ENACTED\b|^\s*Be\s+it\s+enacted\b")
RE_SCHEDULE_AUTHORITY = re.compile(r"^\s*Sections?\s+[\d,\sand]+$")
RE_PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$")

#: A section heading longer than this is almost certainly body text that
#: happened to start with a number.
MAX_HEADING_CHARS = 110


class ParserError(RuntimeError):
    """Raised when the source cannot be turned into a usable tree."""


@dataclass(frozen=True, slots=True)
class Line:
    """One physical line, with the indentation the layout preserved."""

    text: str
    indent: int
    page: int

    @property
    def blank(self) -> bool:
        return not self.text


@dataclass(frozen=True, slots=True)
class ParseResult:
    tree: Node
    parser_name: str
    counts: dict[str, int]
    pages: int
    source: Path | None


# --------------------------------------------------------------------------- #
# Source loading
# --------------------------------------------------------------------------- #
def load_corpus_text(path: Path) -> str:
    """Read a corpus as layout-preserved text.

    ``.txt``/``.text`` files are read directly; PDFs are converted with
    ``pdftotext -layout``, which must be on ``PATH``.
    """
    if not path.exists():
        raise ParserError(
            f"corpus not found at {path}. Generate the demonstration corpus with "
            "`uv run rights-corpus`, or point RIGHTS_CORPUS at your own PDF."
        )
    if path.suffix.lower() in {".txt", ".text"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() != ".pdf":
        raise ParserError(f"unsupported corpus type {path.suffix!r}; expected .pdf or .txt")
    if shutil.which("pdftotext") is None:
        raise ParserError(
            "pdftotext is not installed but is required to read a PDF corpus "
            "(brew install poppler / apt-get install -y poppler-utils). "
            "Alternatively point RIGHTS_CORPUS at a layout text file."
        )
    try:
        completed = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            check=True,
            capture_output=True,
            timeout=300,
        )
    except subprocess.CalledProcessError as exc:
        raise ParserError(f"pdftotext failed on {path}: {exc.stderr.decode(errors='replace')[:400]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ParserError(f"pdftotext timed out on {path}") from exc
    return completed.stdout.decode("utf-8", errors="replace")


def corpus_fingerprint(path: Path) -> str:
    """Short content hash of the corpus file, for ``index_version``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:8]


# --------------------------------------------------------------------------- #
# Front matter and running headers
# --------------------------------------------------------------------------- #
def _document_title(pages: list[str]) -> str:
    """First substantial line of the front matter."""
    for line in pages[0].splitlines():
        stripped = line.strip()
        if len(stripped) > 8 and not stripped.isdigit():
            return stripped
    return "Document"


def _header_patterns(title: str) -> list[re.Pattern[str]]:
    """Case-sensitive patterns for repeated page furniture.

    ``re.escape`` on the detected title means a document whose short title
    contains regex metacharacters still works.
    """
    escaped = re.escape(title)
    return [
        re.compile(rf"^\s*\d*\s*{escaped}(\s*\(c\.\s*\d+\))?\s*$"),
        re.compile(rf"^\s*\d+\s+{escaped}"),
        # ``Part 1 — Employment rights`` / ``Part 1—Employment rights``: mixed
        # case, so this cannot match the ALL CAPS ``PART 1`` marker.
        re.compile(r"^\s*Part\s+[0-9IVXLCDM]+\s*[—–-]"),
        re.compile(r"^\s*Schedules?\s*$"),
        re.compile(r"^\s*Schedule\s+\d+\s*[—–-]"),
    ]


def _skip_front_matter(pages: list[str]) -> tuple[int, int]:
    """Return ``(page_index, line_index)`` where the body begins.

    Primary signal is the enacting formula.  The fallback looks for the first
    ``PART``/``SCHEDULE`` marker that is followed by an actual subsection within
    a short window -- a table of contents lists section headings but never their
    subsections, which is what makes the two distinguishable.
    """
    for page_index, page in enumerate(pages):
        lines = page.splitlines()
        for line_index, line in enumerate(lines):
            if RE_ENACTING.match(line):
                cursor = line_index + 1
                while cursor < len(lines) and lines[cursor].strip():
                    cursor += 1
                log.debug("front matter ends at page %d line %d (enacting formula)", page_index + 1, cursor)
                return page_index, cursor

    for page_index, page in enumerate(pages):
        lines = page.splitlines()
        for line_index, line in enumerate(lines):
            if not (RE_PART.match(line) or RE_SCHEDULE.match(line)):
                continue
            window = lines[line_index : line_index + 60]
            if any(RE_SUBSECTION.match(candidate.strip()) for candidate in window):
                log.warning(
                    "no enacting formula found; body assumed to start at page %d line %d",
                    page_index + 1,
                    line_index,
                )
                return page_index, line_index
    raise ParserError(
        "could not locate the start of the body: no enacting formula and no Part "
        "marker followed by subsections. Is this a layout-preserved extract?"
    )


def _lines(text: str) -> tuple[list[Line], str, int]:
    """Tokenise into body lines, dropping front matter and page furniture."""
    pages = text.split("\f")
    title = _document_title(pages)
    headers = _header_patterns(title)
    start_page, start_line = _skip_front_matter(pages)

    out: list[Line] = []
    for page_index in range(start_page, len(pages)):
        raw_lines = pages[page_index].splitlines()
        first = start_line if page_index == start_page else 0
        for raw in raw_lines[first:]:
            if any(pattern.match(raw) for pattern in headers):
                continue
            if RE_PAGE_NUMBER.match(raw) or RE_SCHEDULE_AUTHORITY.match(raw):
                continue
            stripped = raw.rstrip()
            out.append(
                Line(
                    text=stripped.strip(),
                    indent=len(stripped) - len(stripped.lstrip()) if stripped.strip() else 0,
                    page=page_index + 1,
                )
            )
    return out, title, len(pages)


# --------------------------------------------------------------------------- #
# Classification helpers
# --------------------------------------------------------------------------- #
def _is_caps_title(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _is_section_line(line: Line) -> re.Match[str] | None:
    # Body sections start at column 0; schedule paragraphs are set one or two
    # spaces in. Requiring column 0 dropped every schedule paragraph, which
    # stranded the provisions they insert -- the paragraph carries the "In the
    # Education Act 2002 ... insert-" directive that names the host Act.
    if line.blank or line.indent > MAX_PROVISION_INDENT:
        return None
    match = RE_SECTION.match(line.text)
    if match is None:
        return None
    heading = match.group(2).strip()
    if len(heading) > MAX_HEADING_CHARS or heading.endswith("."):
        return None
    return match


def _is_inserted_provision(lines: list[Line], index: int) -> re.Match[str] | None:
    """An inserted-provision heading, told apart from a wrapped fragment.

    A heading stands on its own line after a blank, starts with a capital, and
    does not end in a full stop. A continuation line does none of those.
    """
    line = lines[index]
    if line.indent == 0:
        return None
    match = RE_INSERTED.match(line.text)
    if match is None:
        return None
    heading = match.group(2).strip()
    if len(heading) > MAX_HEADING_CHARS or heading.endswith("."):
        return None
    if not heading[:1].isupper():
        return None
    if index > 0 and not lines[index - 1].blank:
        return None
    return match


def _next_content(lines: list[Line], index: int) -> Line | None:
    for candidate in lines[index + 1 :]:
        if not candidate.blank:
            return candidate
    return None


#: Terminal punctuation on a line means it is prose, not a heading.  Checked
#: after quote characters are stripped: a wrapped clause ending ``employer.”``
#: is prose, and treating it as a cross-heading poisons every breadcrumb below
#: it (this is trap 3, and it bites in exactly this form).
_PROSE_ENDINGS = (".", ";", ":", ",", "—", "-", "and", " or")


def _is_cross_heading(lines: list[Line], index: int) -> bool:
    """Trap 3: disambiguate by position, not content.

    A cross-heading is short, indented, not ALL CAPS, starts with a capital,
    stands alone after a blank line, carries no terminal punctuation, and is
    *immediately followed by a column-0 section line*.  Quoted material that
    looks like a heading fails at least one of those -- most often the last.
    """
    line = lines[index]
    if line.blank or line.indent == 0 or len(line.text) > 80:
        return False
    text = line.text.strip(QUOTE_CHARS + " ")
    if not text or _is_caps_title(text) or text.endswith(_PROSE_ENDINGS):
        return False
    if not (text[0].isupper() or text[0].isdigit()):
        return False
    if RE_SUBSECTION.match(text) or RE_PARAGRAPH.match(text):
        return False
    if index > 0 and not lines[index - 1].blank:
        return False
    following = _next_content(lines, index)
    return following is not None and _is_section_line(following) is not None


_SMALL_WORDS = frozenset(
    "a an and as at by for from in into of on or the to with without".split()
)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip(QUOTE_CHARS + " ")).strip()


def _title_case(text: str) -> str:
    """Title-case an ALL CAPS structural title without shouting small words."""
    words = _clean(text).lower().split()
    return " ".join(
        word if index and word in _SMALL_WORDS else word[:1].upper() + word[1:]
        for index, word in enumerate(words)
    )


# --------------------------------------------------------------------------- #
# The parse
# --------------------------------------------------------------------------- #
class _Builder:
    """Stack-based tree assembly.

    Kept as a class because the state -- current part, heading, provision,
    subsection and the indent of any open inserted block -- is genuinely
    stateful, and threading six variables through a loop reads worse.
    """

    def __init__(self, title: str) -> None:
        self.root = Node(KIND_DOCUMENT, title=title)
        self.container: Node = self.root  # part or schedule
        self.schedule: Node | None = None
        self.heading: Node | None = None
        self.provision: Node | None = None
        self.subsection: Node | None = None
        self.inserted: Node | None = None
        self.inserted_indent: int | None = None
        self.text_target: Node | None = None

    # ---- containers -------------------------------------------------------
    def open_part(self, number: str, title: str, page: int) -> None:
        parent = self.schedule if self.schedule is not None else self.root
        self.container = parent.add(Node(KIND_PART, number=number, title=title, page=page))
        self.heading = None
        self._close_provision()

    def open_schedule(self, number: str, title: str, page: int, authorising: str = "") -> None:
        self.schedule = self.root.add(
            Node(
                KIND_SCHEDULE,
                number=number,
                title=title,
                page=page,
                authorising_section=authorising,
            )
        )
        self.container = self.schedule
        self.heading = None
        self._close_provision()

    def open_heading(self, title: str, page: int) -> None:
        self.heading = self.container.add(Node(KIND_HEADING, title=title, page=page))
        self._close_provision()

    def open_section(self, number: str, title: str, page: int) -> None:
        parent = self.heading if self.heading is not None else self.container
        self.provision = parent.add(Node(KIND_SECTION, number=number, title=title, page=page))
        self.subsection = None
        self.inserted = None
        self.inserted_indent = None
        self.text_target = self.provision

    def open_inserted(self, number: str, title: str, page: int, indent: int) -> None:
        """Trap 4: nest under whatever is doing the inserting, never beside it."""
        host_section = self.provision
        parent = self.subsection or host_section or self.container
        node = Node(
            KIND_INSERTED,
            number=number,
            title=title,
            page=page,
            host_document=self._host_document(),
            inserted_by=host_section.number if host_section is not None else "",
            inserted_by_schedule=self.schedule.number if self.schedule is not None else "",
        )
        self.inserted = parent.add(node)
        self.inserted_indent = indent
        self.subsection = None
        self.text_target = self.inserted

    def open_subsection(self, number: str, text: str, page: int, indent: int) -> None:
        if (
            self.inserted is not None
            and self.inserted_indent is not None
            and indent < self.inserted_indent - 2
        ):
            # Back out to the host provision: the inserted block has closed.
            self.inserted = None
            self.inserted_indent = None
        parent = self.inserted or self.provision
        if parent is None:
            # A subsection with no provision above it: keep the text rather
            # than dropping it, attached to the nearest container.
            parent = self.heading or self.container
        node = parent.add(Node(KIND_SUBSECTION, number=number, text=text, page=page))
        self.subsection = node
        self.text_target = node

    # ---- text -------------------------------------------------------------
    def append_text(self, text: str) -> None:
        target = self.text_target
        if target is None:
            return
        target.text = f"{target.text} {text}".strip() if target.text else text

    def _close_provision(self) -> None:
        self.provision = None
        self.subsection = None
        self.inserted = None
        self.inserted_indent = None
        self.text_target = None

    def _host_document(self) -> str:
        """The Act being amended, read from the inserting subsection's text.

        Falls back to the previous inserted provision in the same block: an
        inserting subsection often names the host Act once and then inserts
        several consecutive provisions into it.
        """
        # Nearest first: the inserting subsection, then every earlier subsection
        # of the same provision (where "The X Act is amended as follows" lives),
        # then the provision's own text.
        candidates: list[str] = []
        if self.subsection is not None:
            candidates.append(self.subsection.text)
        if self.provision is not None:
            candidates.append(self.provision.text)
            # A schedule paragraph shares its line with the directive --
            # ``1    In the Education Act 2002, after Part 8 insert-`` -- so the
            # host Act ends up in the paragraph's *title*, not its text.
            candidates.append(self.provision.title)
            for child in reversed(self.provision.children):
                if child.kind == KIND_SUBSECTION:
                    candidates.append(child.text)
        for text in candidates:
            match = RE_INSERT_DIRECTIVE.search(text or "")
            if match:
                return _clean(match.group(1) or match.group(2))
        if self.inserted is not None and self.inserted.host_document:
            return self.inserted.host_document
        return ""


#: Characters that end a clause in statutory drafting. A line ending in any of
#: these has finished saying something; a line ending in a bare word has not.
_TERMINATORS = ".;:\u2014\u2013!?\u201d\"'"


def _is_subsection_continuation(lines: list[Line], index: int) -> bool:
    """True when a ``(n)`` at line start continues the previous line.

    Statutory cross-references wrap, and they wrap across the number::

        (ii)   where section 27BC applies, that comply with subsection
               (2) of that section.

    Read as an opener, that produced a subsection ``s.1(2)`` whose entire text
    was ``of that section.`` -- a chunk that says nothing, is retrievable, and
    duly turned up as evidence in an answer. Two signals together, because
    either alone has false positives: the previous line stopped mid-sentence,
    *and* this line is indented past it, which is what a wrap looks like and
    what a genuine opener never does.
    """
    if index == 0:
        return False
    previous_index = None
    for offset in range(index - 1, -1, -1):
        if not lines[offset].blank:
            previous_index = offset
            break
    if previous_index is None:
        return False
    previous = lines[previous_index]
    text = (previous.text or "").rstrip()
    if not text or text[-1] in _TERMINATORS:
        return False
    # A heading is not an unfinished sentence. "1  Right to guaranteed hours"
    # ends in a bare word and is followed by an indented "(1)", which is exactly
    # the shape this guard looks for -- so without this the guard swallowed the
    # first subsection of every provision in the Act.
    if _is_structural(lines, previous_index):
        return False
    return lines[index].indent > previous.indent


def _is_structural(lines: list[Line], index: int) -> bool:
    """True when the line opens a container rather than continuing prose."""
    line = lines[index]
    return bool(
        RE_PART.match(line.text)
        or RE_SCHEDULE.match(line.text)
        or RE_PROVISION_WITH_SUBSECTION.match(line.text)
        or _is_section_line(line)
        or _is_inserted_provision(lines, index)
        or _is_caps_title(line.text)
    )


def _first_number(reference: str | None) -> str:
    """``"149(2) and (3)"`` -> ``"149"``; the section number without subsections."""
    if not reference:
        return ""
    match = re.match(r"\s*(\d+[A-Z]*)", reference)
    return match.group(1) if match else ""


def _resolve_schedule_hosts(tree: Node) -> int:
    """Attribute schedule-inserted provisions via their authorising section.

    A schedule that amends another Act names it once, in the body section that
    introduces the schedule -- ``Schedule 5 amends the Seafarers' Wages Act
    2023`` -- and never again inside the schedule itself. Without following that
    cross-reference every provision the schedule inserts is stranded as "the
    host Act", which is honest but useless: the citation names no statute a
    reader could look up. Returns the number of provisions attributed.
    """
    body_sections = {
        node.number: node
        for node in tree.walk()
        if node.kind == KIND_SECTION and node.ancestor(KIND_SCHEDULE) is None
    }
    resolved = 0
    for schedule in (n for n in tree.walk() if n.kind == KIND_SCHEDULE):
        section = body_sections.get(schedule.authorising_section)
        if section is None:
            continue
        text = " ".join([section.text or ""] + [c.text or "" for c in section.children])
        match = RE_SCHEDULE_AMENDS.search(text)
        if not match:
            continue
        host = _clean(match.group(1))
        for node in schedule.walk():
            if node.kind == KIND_INSERTED and not node.host_document:
                node.host_document = host
                resolved += 1
    return resolved


def parse_text(text: str, *, source: Path | None = None) -> ParseResult:
    """Parse layout text into a tree."""
    lines, title, pages = _lines(text)
    builder = _Builder(title)

    index = 0
    total = len(lines)
    while index < total:
        line = lines[index]
        if line.blank:
            index += 1
            continue

        part_match = RE_PART.match(line.text) if _is_caps_title(line.text) else None
        if part_match:
            following = _next_content(lines, index)
            part_title = ""
            if following is not None and _is_caps_title(following.text) and not RE_SCHEDULE.match(following.text):
                part_title = _title_case(following.text)
                index = lines.index(following, index + 1)
            builder.open_part(part_match.group(1), part_title, line.page)
            index += 1
            continue

        schedule_match = RE_SCHEDULE.match(line.text)
        if schedule_match:
            following = _next_content(lines, index)
            schedule_title = ""
            if following is not None and _is_caps_title(following.text) and not RE_PART.match(following.text):
                schedule_title = _title_case(following.text)
                index = lines.index(following, index + 1)
            builder.open_schedule(
                schedule_match.group(1),
                schedule_title,
                line.page,
                _first_number(schedule_match.group(2)),
            )
            index += 1
            continue

        section_match = _is_section_line(line)
        if section_match:
            builder.open_section(section_match.group(1), _clean(section_match.group(2)), line.page)
            index += 1
            continue

        if line.indent <= MAX_PROVISION_INDENT:
            combined = RE_PROVISION_WITH_SUBSECTION.match(line.text)
            if combined:
                number, first, body = combined.groups()
                builder.open_section(number, "", line.page)
                builder.open_subsection(first, _clean(body), line.page, line.indent + 1)
                index += 1
                continue

        if _is_cross_heading(lines, index):
            builder.open_heading(_clean(line.text), line.page)
            index += 1
            continue

        inserted_match = _is_inserted_provision(lines, index)
        if inserted_match:
            builder.open_inserted(
                inserted_match.group(1), _clean(inserted_match.group(2)), line.page, line.indent
            )
            index += 1
            continue

        subsection_match = RE_SUBSECTION.match(line.text)
        if subsection_match and line.indent > 0 and not _is_subsection_continuation(lines, index):
            builder.open_subsection(
                subsection_match.group(1), _clean(subsection_match.group(2)), line.page, line.indent
            )
            index += 1
            continue

        builder.append_text(_clean(line.text))
        index += 1

    tree = builder.root
    _prune(tree)
    attributed = _resolve_schedule_hosts(tree)
    counts = dict(stats(tree))
    log.info(
        "parsed %s: %s",
        source.name if source else "<text>",
        ", ".join(f"{kind}={count}" for kind, count in sorted(counts.items())),
    )
    if attributed:
        log.info("attributed %d schedule-inserted provisions via authorising sections", attributed)
    return ParseResult(tree=tree, parser_name=PARSER_NAME, counts=counts, pages=pages, source=source)


def parse_corpus(path: Path) -> ParseResult:
    """Load and parse a corpus file (PDF or layout text)."""
    return parse_text(load_corpus_text(path), source=path)


def _prune(tree: Node) -> None:
    """Drop cross-headings that ended up carrying nothing.

    A cross-heading is *inferred* -- from position, not from a marker -- so one
    with no section under it is probably a false positive, and it would
    otherwise appear in every breadcrumb beneath it.

    Parts and Schedules are **not** pruned, even when empty. They are declared
    by an unambiguous ALL-CAPS marker, so an empty one is real structure rather
    than a mistake: the real Act's Schedule 9 is a bare list of bodies with no
    numbered paragraphs, and dropping it lost a whole schedule and every
    citation that would have pointed into it.
    """
    for node in list(tree.walk()):
        if node.kind == KIND_HEADING and not node.children and not node.text.strip():
            if node.parent is not None:
                node.parent.children.remove(node)
                node.parent = None


# --------------------------------------------------------------------------- #
# Validation gate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class TreeExpectations:
    """Minimum structure a corpus must yield to be worth indexing.

    Defaults suit a mid-sized Act.  A smaller corpus should lower them
    explicitly rather than having the gate quietly pass on an empty tree.
    """

    min_sections: int = 150
    min_subsections: int = 900
    min_parts: int = 5
    #: Ceiling on subsections with no provision above them. Non-zero because
    #: real legislation contains layouts this parser has not been taught, and a
    #: hard zero would make any new corpus unusable; small enough that the 39%
    #: the real Act produced before the Schedule-paragraph fix would fail.
    max_orphan_share: float = 0.10


def validate_tree(tree: Node, expectations: TreeExpectations | None = None) -> dict[str, int]:
    """Assert the tree is deep enough and every leaf is locatable.

    Raises :class:`ParserError` listing every failure, not just the first --
    a parser change usually breaks several counts at once and seeing all of
    them is what makes the failure diagnosable.
    """
    expectations = expectations or TreeExpectations()
    counts = dict(stats(tree))
    problems: list[str] = []

    for kind, minimum in (
        (KIND_SECTION, expectations.min_sections),
        (KIND_SUBSECTION, expectations.min_subsections),
        (KIND_PART, expectations.min_parts),
    ):
        found = counts.get(kind, 0)
        if found < minimum:
            problems.append(f"{kind}: found {found}, expected at least {minimum}")

    orphans = [
        node
        for node in tree.walk()
        if node.kind == KIND_SUBSECTION and ">" not in node.breadcrumb()
    ]
    if orphans:
        problems.append(
            f"{len(orphans)} subsections have no ancestors in their breadcrumb "
            f"(first: {orphans[0].citation()!r})"
        )

    orphans = orphan_subsections(tree)
    subsections = counts.get(KIND_SUBSECTION, 0)
    share = len(orphans) / subsections if subsections else 0.0
    if share > expectations.max_orphan_share:
        problems.append(
            f"{len(orphans)} of {subsections} subsections ({share:.1%}) have no provision "
            f"above them, over the {expectations.max_orphan_share:.0%} ceiling. They are "
            "excluded from the index because they cannot be cited, so this is lost content: "
            "the parser has met a provision layout it does not recognise."
        )

    empty = [node for node in tree.walk() if node.kind == KIND_SUBSECTION and not node.text.strip()]
    if len(empty) > max(5, counts.get(KIND_SUBSECTION, 0) // 100):
        problems.append(f"{len(empty)} subsections have no text -- front matter parsed as body?")

    if problems:
        raise ParserError("tree validation failed:\n  - " + "\n  - ".join(problems))
    return counts
