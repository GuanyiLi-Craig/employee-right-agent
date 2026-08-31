"""Side-by-side harness: fixed windows against the hierarchical index.

The contrast is the point of the exercise, and it is not about scores.  Both
indexes return *something* for every query.  The difference is that a fixed
window has no idea which provision it came from, so nothing it returns can be
cited -- and an answer that cannot be cited is an answer nobody can check.

    uv run rights-compare
    uv run rights-compare -q "What is the threshold for a penalty notice?"
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from rights_agent.config import SIMPLE_COLLECTION, Settings
from rights_agent.entrypoints import operator_error_exit
from rights_agent.log import get_logger
from rights_agent.pipelines.common import build_parser, settings_from_args
from rights_agent.retrieval import Retriever
from rights_agent.store import IndexNotBuiltError, chroma_client, open_collection

log = get_logger("pipelines.compare")

DEFAULT_QUESTIONS: tuple[str, ...] = (
    "What does the Act say about bereavement leave?",
    "What is the threshold for a penalty notice?",
    "How are tips allocated between workers?",
    "When must an employer give notice of a shift?",
    "What happens if an employer fails to consult before a collective redundancy?",
)

#: A citation looks like ``s.19(2)`` or ``Sch. 1 para. 3(1)``.  Used only to
#: demonstrate that fixed windows contain no such thing in a usable position.
CITATION_LIKE = re.compile(r"\bs\.\d+[A-Z]*(\(\d+\))?|\bSch\.\s*\d+\s*para\.\s*\d+")


@dataclass(frozen=True, slots=True)
class SimpleHit:
    id: str
    score: float
    offset: int
    preview: str
    citation: str = ""  # always empty: the pipeline never knew what a section was


@dataclass(frozen=True, slots=True)
class HierarchicalHit:
    id: str
    score: float
    citation: str
    breadcrumb: str
    expanded: bool
    preview: str


@dataclass(frozen=True, slots=True)
class Comparison:
    question: str
    simple: list[SimpleHit] = field(default_factory=list)
    hierarchical: list[HierarchicalHit] = field(default_factory=list)

    @property
    def simple_citable(self) -> int:
        return sum(1 for hit in self.simple if hit.citation)

    @property
    def hierarchical_citable(self) -> int:
        return sum(1 for hit in self.hierarchical if hit.citation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "simple": [asdict(hit) for hit in self.simple],
            "hierarchical": [asdict(hit) for hit in self.hierarchical],
            "simple_citable": self.simple_citable,
            "hierarchical_citable": self.hierarchical_citable,
        }


def _preview(text: str, width: int = 96) -> str:
    collapsed = " ".join(text.split())
    return collapsed[:width] + ("…" if len(collapsed) > width else "")


def compare(
    questions: Sequence[str], settings: Settings, k: int = 3
) -> list[Comparison]:
    """Query both indexes for each question."""
    retriever = Retriever(settings)
    client = chroma_client(settings)
    try:
        simple = open_collection(client, SIMPLE_COLLECTION, retriever.embedder)
    except IndexNotBuiltError as exc:
        raise IndexNotBuiltError(
            f"{exc} The baseline index is built separately: `uv run rights-ingest-simple` "
            "(or `docker compose run --rm ingest-simple`)."
        ) from exc

    results: list[Comparison] = []
    for question in questions:
        raw = simple.query(
            query_texts=[question],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        simple_hits: list[SimpleHit] = []
        for index, identifier in enumerate((raw.get("ids") or [[]])[0]):
            metadata = dict(((raw.get("metadatas") or [[{}]])[0])[index] or {})
            document = str(((raw.get("documents") or [[""]])[0])[index] or "")
            distance = float(((raw.get("distances") or [[1.0]])[0])[index])
            simple_hits.append(
                SimpleHit(
                    id=str(identifier),
                    score=round(max(0.0, 1.0 - distance), 4),
                    offset=int(metadata.get("offset", 0)),
                    preview=_preview(document),
                )
            )

        hierarchical_hits = [
            HierarchicalHit(
                id=doc.id,
                score=round(doc.score, 4),
                citation=doc.citation,
                breadcrumb=doc.breadcrumb,
                expanded=doc.expanded,
                preview=_preview(doc.text),
            )
            for doc in retriever.search(question, k=k)
        ]
        results.append(
            Comparison(question=question, simple=simple_hits, hierarchical=hierarchical_hits)
        )
    return results


def render(results: Sequence[Comparison]) -> str:
    """Human-readable report."""
    lines: list[str] = []
    for result in results:
        lines.append("=" * 100)
        lines.append(f"Q  {result.question}")
        lines.append("-" * 100)
        lines.append(f"  simple ({SIMPLE_COLLECTION}) — fixed windows, no structure")
        for hit in result.simple:
            lines.append(f"    {hit.score:.3f}  offset {hit.offset:>7}  citation: (none)")
            lines.append(f"           {hit.preview}")
        lines.append("  hierarchical (corpus_leaves + corpus_parents) — breadcrumb-embedded")
        for hit in result.hierarchical:
            marker = "widened" if hit.expanded else "leaf   "
            lines.append(f"    {hit.score:.3f}  {marker}  citation: {hit.citation}")
            lines.append(f"           {hit.preview}")
        lines.append(
            f"  citable results: simple {result.simple_citable}/{len(result.simple)}"
            f"   hierarchical {result.hierarchical_citable}/{len(result.hierarchical)}"
        )
    totals_simple = sum(r.simple_citable for r in results)
    totals_hier = sum(r.hierarchical_citable for r in results)
    total_simple_hits = sum(len(r.simple) for r in results)
    total_hier_hits = sum(len(r.hierarchical) for r in results)
    lines.append("=" * 100)
    lines.append(
        f"TOTAL citable: simple {totals_simple}/{total_simple_hits}, "
        f"hierarchical {totals_hier}/{total_hier_hits}"
    )
    lines.append(
        "The scores are comparable; the citations are not. That asymmetry is the whole result."
    )
    return "\n".join(lines)


@operator_error_exit
def main(argv: list[str] | None = None) -> int:
    parser = build_parser("Compare the fixed-window baseline against the hierarchical index.")
    parser.add_argument(
        "-q",
        "--question",
        action="append",
        dest="questions",
        help="question to compare (repeatable); defaults to a built-in set",
    )
    parser.add_argument("-k", type=int, default=3, help="results per index")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = parser.parse_args(argv)
    settings = settings_from_args(args)
    questions = tuple(args.questions or DEFAULT_QUESTIONS)
    results = compare(questions, settings, k=args.k)
    if args.json:
        print(json.dumps([result.to_dict() for result in results], indent=2, ensure_ascii=False))
    else:
        print(render(results))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
