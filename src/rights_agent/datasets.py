"""Where the eval datasets live, and why that depends on the embedder.

A golden row asserts that a question retrieves a particular citation. Whether it
does is a property of the *retrieval config*, not of the corpus alone: the same
30 questions on the real Act retrieve their expected citation 83% of the time on
the hashing bag-of-words, 90% on MiniLM and 100% on ``text-embedding-3-small``.
So ``known_failure`` -- and any honest quality floor -- differ per embedder.

One shared set would therefore be wrong for at least two of the three, and wrong
in the direction that looks like a regression in whichever one did not generate
it. Hence one directory per embedder, named by the embedder, with the full
``index_version`` stamped inside ``baseline.json`` so the corpus and parser are
checked too.
"""

from __future__ import annotations

from pathlib import Path

#: Files a complete dataset directory holds.
DATASET_FILES = ("golden.jsonl", "calibration.jsonl", "baseline.json")


class DatasetsMissingError(FileNotFoundError):
    """Raised when no dataset exists for the embedder the index was built with."""


def datasets_dir(evals_dir: Path, embedder: str) -> Path:
    """The dataset directory for ``embedder``, whether or not it exists."""
    return Path(evals_dir) / "datasets" / embedder


def available(evals_dir: Path) -> list[str]:
    root = Path(evals_dir) / "datasets"
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def require_datasets_dir(evals_dir: Path, embedder: str) -> Path:
    """The dataset directory for ``embedder``, or a message naming the fix."""
    path = datasets_dir(evals_dir, embedder)
    if path.is_dir():
        return path
    names = available(evals_dir)
    raise DatasetsMissingError(
        f"no eval datasets for embedder {embedder!r} (looked in {path}). "
        f"Available: {', '.join(names) if names else 'none'}. Either set "
        f"RIGHTS_EMBEDDER to one of those and rebuild the index, or generate a "
        f"set for this one with `python -m rights_agent goldens --write-baseline`."
    )
