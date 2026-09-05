"""Build a poisoned index, through the assistant's own unmodified pipeline.

    uv run python -m attacklab.attacks.corpus --with-poison   # hostile
    uv run python -m attacklab.attacks.corpus                 # clean twin

Four rules, and each one is load-bearing for demo 1.

**The poison is ingested through the unmodified pipeline.**  Same parser, same
embedder, same manifest writer.  If any of it is special-cased the demo is a
lie: the audience is being shown that an ordinary document, ingested
ordinarily, is enough.

**It lands in a different Chroma path with a different index version**, so
swapping back is a fast pointer change rather than a re-ingest.  Re-ingesting on
stage is dead air.

**The manifest records the payload id** for the test suite only.
:mod:`attacklab.supplychain.scan_index` must find the document by *content* and
is forbidden from reading that flag -- there is a test for it.

**The document has to parse.**  It is spliced into the corpus text as a further
subsection of the real section on bereavement leave, in the same layout the
parser already recognises, so it becomes an inserted provision with a real
citation.  If it rendered as an orphan the demo would become a story about a bad
parser instead of a story about injection.
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from attacklab.attacks.catalogue import Payload, indirect_payloads
from attacklab.attacks.catalogue import payload as get_payload
from rights_agent.config import Settings
from rights_agent.config import settings as load_settings
from rights_agent.entrypoints import operator_error_exit
from rights_agent.log import get_logger
from rights_agent.pipelines.hierarchical import ingest
from rights_agent.store import IndexManifest, load_manifest, reset_client_cache, write_manifest

log = get_logger("attacklab.corpus")

#: Where the two twins live.  Siblings of ``runs/`` rather than subdirectories
#: of it: nesting would mean the clean index's Chroma directory contained the
#: hostile one, and the swap is meant to be a pointer change.
#:
#: **Both** twins are built here, including the clean one, rather than reusing
#: the operator's ``runs/``. The operator's index was built with whatever
#: embedder was to hand; if the clean side of the comparison used a different
#: embedder from the poisoned side, the difference the audience sees would be
#: partly a difference in retrieval quality, and the demo would be measuring the
#: wrong thing.
POISON_RUNS_DIRNAME = "runs-poisoned"
CLEAN_RUNS_DIRNAME = "runs-clean"

#: Indentation the layout parser reads as a subsection of a section.
#:
#: Matched to the real corpus: four spaces, ``(n)``, two spaces, then text
#: wrapped at eight.  A payload written as prose is folded into this shape so it
#: parses as an ordinary subsection; a payload already written in corpus layout
#: (``p01``) is spliced verbatim.
SUBSECTION_INDENT = "    "
CONTINUATION_INDENT = "        "
WRAP_CHARS = 66


@dataclass(frozen=True, slots=True)
class PoisonedCorpus:
    """Where the poisoned artefacts went, and what is in them."""

    corpus_path: Path
    runs_dir: Path
    payload_ids: tuple[str, ...]
    manifest: IndexManifest | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "corpus_path": str(self.corpus_path),
            "runs_dir": str(self.runs_dir),
            "payload_ids": list(self.payload_ids),
            "index_version": self.manifest.index_version if self.manifest else "",
        }


def twin_runs_dir(with_poison: bool, settings: Settings | None = None) -> Path:
    """Where one twin's artefacts live."""
    settings = settings or load_settings()
    name = POISON_RUNS_DIRNAME if with_poison else CLEAN_RUNS_DIRNAME
    return settings.runs_dir.parent / name


def poison_runs_dir(settings: Settings | None = None) -> Path:
    return twin_runs_dir(True, settings)


def clean_runs_dir(settings: Settings | None = None) -> Path:
    return twin_runs_dir(False, settings)


def _as_subsection(text: str, number: int) -> str:
    """Fold prose into the layout the parser reads as a subsection.

    Wrapping rather than emitting one long line, because the parser decides what
    is a continuation from indentation and line shape.  A single 300-character
    line parses -- and looks nothing like the rest of the corpus, which matters:
    a payload the audience can spot by its formatting teaches the wrong lesson.
    """
    words = " ".join(text.split()).split(" ")
    prefix = f"{SUBSECTION_INDENT}({number})  "
    lines: list[str] = []
    current = prefix
    for word in words:
        candidate = f"{current}{word} "
        if len(candidate.rstrip()) > WRAP_CHARS + len(CONTINUATION_INDENT) and current.strip():
            lines.append(current.rstrip())
            current = f"{CONTINUATION_INDENT}{word} "
        else:
            current = candidate
    if current.strip():
        lines.append(current.rstrip())
    return "\n".join(lines) + "\n"


def _host_pattern(host_section: str) -> re.Pattern[str]:
    return re.compile(rf"^({re.escape(host_section)}\s*)$", re.MULTILINE)


def splice(corpus_text: str, payload: Payload) -> str:
    """Insert one payload into ``corpus_text`` where the parser will read it.

    ``p01`` is already written in corpus layout and is spliced verbatim.  The
    prose payloads are folded into a subsection of their host section, which is
    the transformation that makes them structurally indistinguishable from the
    real thing -- the parser, the embedder and the manifest writer are all
    untouched, and none of them is told which document is which.
    """
    if not payload.host_section:
        raise RuntimeError(
            f"payload {payload.id} has no host_section, so there is nowhere in the "
            "corpus to put it. Indirect payloads need one; direct ones are typed by "
            "the user and never reach the index."
        )
    pattern = _host_pattern(payload.host_section)
    # The *last* occurrence: the corpus opens with a contents list whose entries
    # are byte-identical to the section headings, and splicing into the table of
    # contents produces a provision the parser never sees as one.
    match = None
    for candidate in pattern.finditer(corpus_text):
        match = candidate
    if match is None:
        raise RuntimeError(
            f"could not find {payload.host_section!r} in the corpus. Update the "
            "payload's host_section rather than loosening the pattern: a payload "
            "spliced somewhere arbitrary will not be retrieved by its question, and "
            "the demo will quietly show nothing."
        )
    # Immediately before the next section heading, which is where a real
    # amending subsection sits.
    tail = corpus_text[match.end() :]
    next_section = re.search(r"\n(?=\s*\d+[A-Z]*\s{2,}\S)", tail)
    cut = match.end() + (next_section.start() if next_section else len(tail))
    if payload.filename.endswith(".md"):
        body = payload.text.rstrip() + "\n"
    else:
        body = _as_subsection(payload.text, _next_subsection_number(corpus_text[:cut]))
    return f"{corpus_text[:cut]}\n{body}{corpus_text[cut:]}"


def _next_subsection_number(before: str) -> int:
    """One past the last subsection number in the host section.

    Numbering the payload ``(9)`` when the host section stops at ``(8)`` is the
    difference between a provision that reads as part of the Act and one that
    reads as a mistake.
    """
    numbers = re.findall(r"^\s{2,6}\((\d+)\)\s", before, re.MULTILINE)
    return int(numbers[-1]) + 1 if numbers else 1


def splice_all(corpus_text: str, payloads: Sequence[Payload]) -> str:
    """Splice every payload, each at its own host section.

    One index holds all of them, which is what keeps the demos fast: no rebuild
    between demo 1 and demo 2.  It works only because no two payloads share a
    host -- if they did, demo 1's question would retrieve two poisons, the input
    scanner would catch whichever was crude, and demo 1 would report a
    containment nobody arranged.  Asserted here rather than trusted.
    """
    hosts = [payload.host_section for payload in payloads]
    duplicated = sorted({host for host in hosts if hosts.count(host) > 1})
    if duplicated:
        raise RuntimeError(
            f"these host sections are claimed by more than one payload: {duplicated}. "
            "Give each payload its own provision; see splice_all's docstring for why."
        )
    for payload in payloads:
        corpus_text = splice(corpus_text, payload)
    return corpus_text


def write_poisoned_corpus(
    settings: Settings, payloads: Sequence[Payload], destination: Path | None = None
) -> Path:
    """Write the spliced corpus next to the poisoned index."""
    runs = destination or poison_runs_dir(settings)
    runs.mkdir(parents=True, exist_ok=True)
    source = settings.corpus_path.read_text(encoding="utf-8")
    out = runs / "corpus.poisoned.txt"
    out.write_text(splice_all(source, payloads), encoding="utf-8")
    log.info(
        "wrote poisoned corpus to %s (%s)",
        out,
        ", ".join(payload.id for payload in payloads),
    )
    return out


def build(
    *,
    with_poison: bool,
    payload_ids: Sequence[str] | None = None,
    settings: Settings | None = None,
) -> PoisonedCorpus:
    """Ingest the poisoned (or clean) twin into its own runs directory."""
    settings = settings or load_settings()
    payloads = (
        [get_payload(identifier) for identifier in payload_ids]
        if payload_ids
        else list(indirect_payloads())
    )
    runs = twin_runs_dir(with_poison, settings)

    if runs.exists():
        # A fresh directory every time. Chroma will happily open a collection
        # built from a different corpus and answer confidently from it, and the
        # only symptom is a stale citation nobody notices until it is on screen.
        shutil.rmtree(runs)
    runs.mkdir(parents=True, exist_ok=True)

    if with_poison:
        corpus_path = write_poisoned_corpus(settings, payloads, runs)
    else:
        corpus_path = runs / settings.corpus_path.name
        shutil.copy2(settings.corpus_path, corpus_path)

    # The pipeline reads both from settings, so pointing them here is the whole
    # of the "different path, different version" requirement.
    poisoned_settings = settings.with_overrides(runs_dir=runs, corpus_path=corpus_path)
    reset_client_cache()
    manifest = ingest(poisoned_settings)

    # Recorded for the *test suite*, never for the scanner. scan_index.py must
    # find the document by content, and there is a test asserting it does not
    # read this file at all.
    if with_poison:
        annotated = IndexManifest(
            **{
                **{
                    field: getattr(manifest, field)
                    for field in IndexManifest.__dataclass_fields__
                    if field != "extra"
                },
                "extra": {
                    **manifest.extra,
                    "poisoned": True,
                    "payload_ids": [payload.id for payload in payloads],
                },
            }
        )
        write_manifest(poisoned_settings, annotated)
        manifest = annotated
    reset_client_cache()
    return PoisonedCorpus(
        corpus_path=corpus_path,
        runs_dir=runs,
        payload_ids=tuple(payload.id for payload in payloads) if with_poison else (),
        manifest=manifest,
    )


def twin_settings(with_poison: bool, settings: Settings | None = None) -> Settings:
    """Settings pointed at one twin, if it has been built.

    This is the "fast swap": no re-ingest, just a different runs directory.
    """
    settings = settings or load_settings()
    runs = twin_runs_dir(with_poison, settings)
    flag = " --with-poison" if with_poison else ""
    if not (runs / "index_manifest.json").exists():
        raise FileNotFoundError(
            f"no index at {runs}. Build it with:\n"
            f"    uv run python -m attacklab.attacks.corpus{flag}"
        )
    candidates = sorted(runs.glob("corpus*.txt"))
    if not candidates:
        raise FileNotFoundError(f"no corpus file under {runs}; rebuild that twin")
    return settings.with_overrides(runs_dir=runs, corpus_path=candidates[0])


def poisoned_settings(settings: Settings | None = None) -> Settings:
    return twin_settings(True, settings)


def clean_settings(settings: Settings | None = None) -> Settings:
    return twin_settings(False, settings)


def twin_built(with_poison: bool, settings: Settings | None = None) -> bool:
    """Whether a twin exists, without raising.  The console asks this on load."""
    return (twin_runs_dir(with_poison, settings) / "index_manifest.json").exists()


def is_poisoned(settings: Settings | None = None) -> bool:
    """Whether the *poisoned* runs directory currently holds the hostile twin.

    Reads the manifest flag, which is legitimate here: this is the lab telling
    the console what it built, not a scanner detecting anything.
    """
    try:
        found = load_manifest(poisoned_settings(settings))
    except (FileNotFoundError, OSError):
        return False
    return bool(found and found.extra.get("poisoned"))


@operator_error_exit
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the poisoned index (or its clean twin) in a separate path."
    )
    parser.add_argument(
        "--with-poison",
        action="store_true",
        help="include the hostile document; omit for the clean twin",
    )
    parser.add_argument(
        "--payload",
        action="append",
        dest="payloads",
        metavar="ID",
        help="payload id to splice in; repeatable. Default: every indirect payload.",
    )
    args = parser.parse_args(argv)

    built = build(with_poison=args.with_poison, payload_ids=args.payloads)
    manifest = built.manifest
    print(f"runs_dir           {built.runs_dir}")
    print(f"corpus             {built.corpus_path.name}")
    print(
        f"poisoned           {bool(built.payload_ids)}"
        + (f" ({', '.join(built.payload_ids)})" if built.payload_ids else "")
    )
    if manifest is not None:
        print(f"index_version      {manifest.index_version}")
        for name, count in sorted(manifest.collections.items()):
            print(f"{name:<19}{count} rows")
    print()
    print("Point the lab at it with RIGHTS_RUNS_DIR, or let the console do it:")
    print(f"    RIGHTS_RUNS_DIR={built.runs_dir} uv run rights-ask \"What does the document say about bereavement leave?\"")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
