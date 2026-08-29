"""Generate ``evals/golden.jsonl`` and ``evals/calibration.jsonl``.

    uv run python -m rights_agent goldens

Design rules the generated sets follow (§14.1):

* **Assert citations, not prose.**  Wording changes when models change; the
  source does not.  A suite that asserts exact text breaks on every improvement,
  and a suite that cries wolf gets deleted.
* **Stratify by intent** so a fix to one topic cannot mask a break in another.
* **Include out-of-scope cases** with ``should_refuse``.  Refusing correctly is
  a behaviour worth testing.
* **Phrase questions as "what does the document say"**, not "am I entitled to".
  The system reports what a source says; it does not advise, and keeping the
  golden set in that register stops the suite quietly asserting a claim about
  the world.
* **Keep a ``known_failure`` flag.**  The gate asserts the list does not *grow*,
  and fails if a known failure starts passing, so the marker gets removed.

Rows are generated against the *live* index and verified before being written:
a golden row whose expected citation is not retrievable is either rephrased or
marked as a known failure.  Committing an unverified golden set produces a suite
that fails for reasons that have nothing to do with the change under test.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from rights_agent.config import Settings, settings as load_settings
from rights_agent.document.nodes import KIND_PART, KIND_SECTION, Node
from rights_agent.document.parser import parse_corpus
from rights_agent.graph import classify_intent
from rights_agent.log import configure_logging, get_logger
from rights_agent.retrieval import Retriever, format_context, sufficiency
from rights_agent.entrypoints import operator_error_exit

log = get_logger("tools.goldens")

#: Questions per Part, giving a set stratified across the document's topics.
QUESTIONS_PER_PART = 5

#: Phrasings, applied in rotation so the set is not one template repeated.
TEMPLATES: tuple[str, ...] = (
    "What does section {number} say about {subject}?",
    "What does the document say about {subject}?",
    "Which provision covers {subject}?",
    "What does section {number} require in relation to {subject}?",
    "What does the document provide about {subject}?",
)

#: Out-of-scope questions.  A system that answers these is worse than one that
#: refuses them, and the eval has to be able to tell the difference.
OUT_OF_SCOPE: tuple[str, ...] = (
    "How do I mine cryptocurrency on company laptops?",
    "What is the capital gains tax rate on disposals of shares?",
    "How do I configure a Kubernetes ingress controller for TLS?",
    "What does the document say about interplanetary shipping tariffs?",
    "Which football team won the league in 1998?",
)

#: Paraphrases with none of the corpus's vocabulary.  A lexical embedder cannot
#: bridge these, and pretending otherwise would hide a real limitation.  They
#: are committed as known failures so the gap is tracked rather than forgotten.
KNOWN_FAILURE_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("Can my boss let me go without any warning at all?", "dismissal"),
    ("Am I allowed time off when a close relative passes away?", "leave"),
)


@dataclass
class GoldenRow:
    id: str
    question: str
    intent: str
    must_cite: list[str] = field(default_factory=list)
    should_refuse: bool = False
    known_failure: bool = False
    note: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


@dataclass
class CalibrationRow:
    id: str
    question: str
    context: str
    answer: str
    human_label: int
    note: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


# --------------------------------------------------------------------------- #
# Subjects
# --------------------------------------------------------------------------- #
_LEADING_NOISE = (
    "removal of the ",
    "removal of ",
    "power to make ",
    "duty to give ",
    "duty to ",
    "right to ",
    "right not to be ",
    "prohibition of ",
    "establishment of the ",
    "application for ",
    "appointment of ",
    "calculation of a ",
    "calculation of ",
)


def subject_of(section: Node) -> str:
    """A noun phrase from a section heading, usable inside a question.

    Headings are already written as topics ("Fair allocation of tips,
    gratuities and service charges"), so this only strips the verbal scaffolding
    that would make the question ungrammatical.
    """
    title = section.title.strip().rstrip(".")
    if ":" in title:
        head, _, tail = title.partition(":")
        title = f"{tail.strip()} ({head.strip().lower()})" if tail.strip() else head
    lowered = title[:1].lower() + title[1:]
    for prefix in _LEADING_NOISE:
        if lowered.startswith(prefix):
            stripped = lowered[len(prefix) :]
            # Only strip if what remains is still specific enough to retrieve on.
            # "Application for recognition" -> "recognition" is a worse question,
            # not a shorter one.
            if len(stripped.split()) >= 3 or len(stripped) >= 18:
                lowered = stripped
            break
    return lowered


def _sections_by_part(tree: Node) -> dict[str, list[Node]]:
    grouped: dict[str, list[Node]] = {}
    for node in tree.walk():
        if node.kind != KIND_SECTION:
            continue
        part = node.ancestor(KIND_PART)
        if part is None or part.parent is not tree:
            # Skip Schedule paragraphs: they are cited differently and would
            # skew the topical stratification.
            continue
        grouped.setdefault(part.label(), []).append(node)
    return grouped


def _spread(items: Sequence[Node], count: int) -> list[Node]:
    """Evenly spaced selection, so a Part is sampled across its whole range."""
    if count >= len(items):
        return list(items)
    step = len(items) / count
    return [items[int(index * step)] for index in range(count)]


# --------------------------------------------------------------------------- #
# Golden set
# --------------------------------------------------------------------------- #
def build_golden_rows(settings: Settings, retriever: Retriever) -> list[GoldenRow]:
    tree = parse_corpus(settings.corpus_path).tree
    grouped = _sections_by_part(tree)

    rows: list[GoldenRow] = []
    index = 0
    for part_label in sorted(grouped):
        for position, section in enumerate(_spread(grouped[part_label], QUESTIONS_PER_PART)):
            template = TEMPLATES[(index) % len(TEMPLATES)]
            question = template.format(number=section.number, subject=subject_of(section))
            citation = section.citation()
            index += 1
            rows.append(
                GoldenRow(
                    id=f"g{index:03d}",
                    question=question,
                    intent=classify_intent(question),
                    must_cite=[citation],
                    note=f"{part_label} · {section.title}",
                )
            )

    for offset, question in enumerate(OUT_OF_SCOPE, start=1):
        rows.append(
            GoldenRow(
                id=f"r{offset:03d}",
                question=question,
                intent=classify_intent(question),
                must_cite=[],
                should_refuse=True,
                note="out of scope",
            )
        )

    for offset, (question, topic) in enumerate(KNOWN_FAILURE_CANDIDATES, start=1):
        rows.append(
            GoldenRow(
                id=f"k{offset:03d}",
                question=question,
                intent=classify_intent(question),
                must_cite=[],
                known_failure=True,
                note=f"paraphrase with no corpus vocabulary ({topic})",
            )
        )

    return _verify(rows, retriever, settings)


def _verify(
    rows: list[GoldenRow], retriever: Retriever, settings: Settings
) -> list[GoldenRow]:
    """Check each row against the live index before committing it.

    Three outcomes: the expected citation is retrievable (keep), it is not but
    the question is in scope (mark ``known_failure`` and say why), or an
    out-of-scope question unexpectedly scores above the gate (report it -- that
    is a threshold problem, not a dataset problem).
    """
    verified: list[GoldenRow] = []
    for row in rows:
        docs = retriever.search(row.question)
        score = sufficiency(docs, row.question)
        retrieved = {doc.citation for doc in docs}
        # A widened parent covers its leaves, so a leaf-level expectation is met
        # by its provision and vice versa.
        prefixes = {citation.split("(")[0] for citation in retrieved}

        if row.should_refuse:
            if score >= settings.sufficiency_threshold:
                log.warning(
                    "out-of-scope row %s scored %.3f, above the %.2f threshold",
                    row.id,
                    score,
                    settings.sufficiency_threshold,
                )
                row.note = f"{row.note} (WARNING: scored {score:.3f})"
            verified.append(row)
            continue

        if row.known_failure:
            if row.must_cite and all(
                citation.split("(")[0] in prefixes for citation in row.must_cite
            ):
                log.warning("known failure %s now passes; drop the marker", row.id)
            verified.append(row)
            continue

        expected = {citation.split("(")[0] for citation in row.must_cite}
        if expected <= prefixes:
            verified.append(row)
            continue

        log.info(
            "row %s: expected %s not retrieved (got %s); marking known_failure",
            row.id,
            sorted(expected),
            sorted(prefixes)[:4],
        )
        row.known_failure = True
        row.note = f"{row.note} · expected citation not in top-{settings.top_k}"
        verified.append(row)
    return verified


# --------------------------------------------------------------------------- #
# Calibration set
# --------------------------------------------------------------------------- #
#: Hard cases, hand-written.  Clean examples alone yield a near-perfect kappa;
#: these drop it materially.  The judge did not get worse -- the test got honest.
HARD_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "h001",
        "kind": "paraphrase",
        "human_label": 1,
        "note": "correct paraphrase using none of the context's vocabulary; a lexical judge scores it 0 — a structural false negative",
    },
    {
        "id": "h002",
        "kind": "partial",
        "human_label": 0,
        "note": "one supported claim, one invented claim",
    },
    {
        "id": "h003",
        "kind": "uncited",
        "human_label": 0,
        "note": "accurate but uncited; ungateable as shipped, though groundedness alone scores it high",
    },
    {
        "id": "h004",
        "kind": "trailing",
        "human_label": 0,
        "note": "grounded first sentence, plausible-sounding unsupported second sentence",
    },
    {
        "id": "h005",
        "kind": "boilerplate",
        "human_label": 0,
        "note": "quotes real, correctly-cited boilerplate that does not answer the question; a support-only judge scores it perfect",
    },
)

#: Sentences that appear in almost every provision.  Quoting one is grounded,
#: cited, and useless -- which is why it is a hard case.
BOILERPLATE_MARKERS = (
    "may present a complaint",
    "Regulations under subsection",
    "must have regard to any relevant code of practice",
)

PARAPHRASE_ANSWER = (
    "Staff who lose someone close to them can take a short break from work, and the "
    "boss has to let them, without docking their wages for it."
)
#: Fabricated provisions for the negative rows.  Deliberately built from
#: vocabulary the corpus does not use: a lexical judge should score them low, and
#: if it does not, the judge -- not the row -- is what needs fixing.
FABRICATED_PROVISION = (
    "The Commissioner for Maritime Salvage shall publish a bilingual tonnage certificate "
    "in the harbour register before any vessel discharges ballast."
)

INVENTED_CLAIM = (
    "The employer must also publish the outcome in the Maritime Salvage Register within "
    "nine days of the harbour inspection."
)
TRAILING_CLAIM = (
    "In practice most harbour authorities extend this to a fortnight of ballast monitoring, "
    "which the salvage tribunals treat as the norm."
)


def build_calibration_rows(
    settings: Settings, retriever: Retriever, clean_questions: Sequence[str]
) -> list[CalibrationRow]:
    """Twelve clean rows plus the four hard cases."""
    rows: list[CalibrationRow] = []
    contexts: list[tuple[str, str]] = []
    for question in clean_questions:
        docs = retriever.search(question)
        contexts.append((question, format_context(docs)))

    for offset, (question, context) in enumerate(contexts, start=1):
        blocks = _split_blocks(context)
        if not blocks:
            continue
        citation, sentence = blocks[0]
        if offset % 3 == 0:
            # An ungrounded answer: right shape, wrong provenance.
            answer = f"[{citation}] provides: {FABRICATED_PROVISION}"
            label = 0
            note = "fabricated provision text"
        else:
            answer = f"[{citation}] provides: {sentence}"
            label = 1
            note = "verbatim quotation with a resolvable citation"
        rows.append(
            CalibrationRow(
                id=f"c{offset:03d}",
                question=question,
                context=context,
                answer=answer,
                human_label=label,
                note=note,
            )
        )

    # Hard cases reuse real contexts so the only thing that varies is the answer.
    for case in HARD_CASES:
        question, context = contexts[len(rows) % len(contexts)]
        blocks = _split_blocks(context)
        if not blocks:
            continue
        citation, sentence = blocks[0]
        kind = case["kind"]
        if kind == "paraphrase":
            question = "What does the document say about bereavement leave?"
            paraphrase_docs = retriever.search(question)
            context = format_context(paraphrase_docs)
            answer = PARAPHRASE_ANSWER
        elif kind == "partial":
            answer = f"[{citation}] provides: {sentence} {INVENTED_CLAIM}"
        elif kind == "uncited":
            answer = sentence
        elif kind == "boilerplate":
            question = "What does the document say about bereavement leave?"
            docs = retriever.search(question)
            context = format_context(docs)
            answer = _boilerplate_answer(context) or f"[{citation}] provides: {sentence}"
        else:
            answer = f"[{citation}] provides: {sentence} {TRAILING_CLAIM}"
        rows.append(
            CalibrationRow(
                id=str(case["id"]),
                question=question,
                context=context,
                answer=answer,
                human_label=int(case["human_label"]),
                note=str(case["note"]),
            )
        )
    return rows


#: Sentence boundary that also splits on the newline between subsections --
#: splitting on ``". "`` alone silently returns the whole provision as one
#: "sentence", which quietly turns a clean calibration row into a hard one.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.;])\s+")


def _boilerplate_answer(context: str) -> str:
    """A correctly-cited quotation of boilerplate from ``context``."""
    from rights_agent.llm import parse_context

    for block in parse_context(context):
        for part in _SENTENCE_BOUNDARY.split(block.text):
            part = part.strip()
            if any(marker in part for marker in BOILERPLATE_MARKERS):
                return f"[{block.citation}] provides: {part if part.endswith('.') else part + '.'}"
    return ""


def _split_blocks(context: str) -> list[tuple[str, str]]:
    """``(citation, first sentence)`` for each context block."""
    from rights_agent.llm import parse_context

    out: list[tuple[str, str]] = []
    for block in parse_context(context):
        sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(block.text) if part.strip()]
        if not sentences:
            continue
        first = sentences[0]
        out.append((block.citation, first if first.endswith(".") else f"{first}."))
    return out


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def write_jsonl(path: Path, rows: Iterable[object]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [row.to_json() for row in rows]  # type: ignore[attr-defined]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


#: Quality gates start here and are meant to be *ratcheted upward* as the system
#: improves -- never downward to fix a red build.  Written only when the baseline
#: file does not yet exist; an existing file's thresholds are left alone, because
#: regenerating the dataset must not quietly relax the gate.
#: Only used when no baseline exists yet. An existing file's thresholds are
#: always preserved -- a regeneration must not reset a gate.
DEFAULT_QUALITY_THRESHOLDS: dict[str, float] = {
    "groundedness_mean": 0.80,
    "groundedness_p10": 0.50,
    "citation_coverage_mean": 0.70,
    "citation_coverage_p10": 0.50,
    "context_relevance_mean": 0.45,
    "answer_relevance_mean": 0.40,
    "judge_kappa": 0.60,
}

#: The generator owns this key and rewrites it from the verified rows. Every
#: other key in the baseline is a human judgement -- a threshold, and the
#: measurement and reasoning behind it -- and is carried forward untouched.
#: Rebuilding the file from scratch quietly deleted the record of *why* a
#: threshold was where it was, which is the only thing that makes the next
#: change to it auditable.
GENERATED_BASELINE_KEYS = frozenset({"known_failures"})


def write_baseline(
    path: Path, golden: Sequence[GoldenRow], manifest: Any | None = None
) -> dict[str, Any]:
    """The committed baseline: which rows are known to fail, and the gates.

    Kept separate from ``golden.jsonl`` on purpose.  If the gate read its own
    baseline out of the dataset it is gating, regenerating the dataset would
    silently accept whatever the system currently does -- which is the one thing
    a regression gate must never do.
    """
    existing: dict[str, Any] = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        key: value for key, value in existing.items() if key not in GENERATED_BASELINE_KEYS
    }
    payload["known_failures"] = sorted(row.id for row in golden if row.known_failure)
    if manifest is not None:
        # Which corpus these rows were verified against. Expected citations name
        # provisions, and provisions only exist in one document -- so a dataset
        # generated for one corpus tells you nothing about another, and running
        # it anyway produces a wall of retrieval failures that look like a
        # regression in the retriever.
        payload["generated_for"] = {
            "corpus": Path(manifest.corpus_path).name,
            "corpus_sha8": manifest.corpus_sha[:8],
            "index_version": manifest.index_version,
        }
    payload.setdefault("quality_thresholds", DEFAULT_QUALITY_THRESHOLDS)
    payload.setdefault(
        "note",
        "known_failures must not grow; a row that starts passing must have its "
        "marker removed. Thresholds ratchet upward only.",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


@operator_error_exit
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the eval datasets from the live index.")
    parser.add_argument("--evals-dir", type=Path, default=Path("evals"))
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="also rewrite evals/baseline.json known_failures (a deliberate act)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    configure_logging("WARNING" if args.quiet else None)

    settings = load_settings()
    retriever = Retriever(settings)

    golden = build_golden_rows(settings, retriever)
    clean_questions = [
        row.question for row in golden if not row.should_refuse and not row.known_failure
    ][:12]
    calibration = build_calibration_rows(settings, retriever, clean_questions)

    # One directory per embedder: the datasets describe a retrieval config, not
    # just a corpus. See ``evals/conftest.py::dataset_dir``.
    out_dir = args.evals_dir / "datasets" / retriever.manifest.embedding_model
    out_dir.mkdir(parents=True, exist_ok=True)
    golden_path = out_dir / "golden.jsonl"
    calibration_path = out_dir / "calibration.jsonl"
    written_golden = write_jsonl(golden_path, golden)
    written_calibration = write_jsonl(calibration_path, calibration)

    known = sum(1 for row in golden if row.known_failure)
    refuse = sum(1 for row in golden if row.should_refuse)
    print(f"{golden_path}: {written_golden} rows "
          f"({written_golden - known - refuse} answerable, {refuse} refusal, {known} known failure)")
    print(f"{calibration_path}: {written_calibration} rows "
          f"({sum(1 for r in calibration if r.human_label == 1)} labelled acceptable)")
    if args.write_baseline:
        baseline_path = out_dir / "baseline.json"
        baseline = write_baseline(baseline_path, golden, retriever.manifest)
        print(f"{baseline_path}: {len(baseline['known_failures'])} known failures recorded")
    print(f"index_version: {retriever.index_version}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
