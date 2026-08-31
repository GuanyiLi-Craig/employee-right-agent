"""The simple embedding pipeline: a fixed-window baseline.

Forty lines, and the correct default for flat prose.  It exists here for two
reasons: it is what everyone writes first, and it is the control that makes the
hierarchical result mean something.

What it cannot do is the point.  Nothing in this pipeline ever knew what a
section was, so no chunk it produces can carry a citation -- and an answer that
cannot cite is an answer nobody can check.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from rights_agent.config import CHUNK_CHARS, OVERLAP_CHARS, SIMPLE_COLLECTION, Settings
from rights_agent.document.parser import corpus_fingerprint, load_corpus_text
from rights_agent.entrypoints import operator_error_exit
from rights_agent.log import get_logger
from rights_agent.pipelines.common import build_parser, resolve_embedder, settings_from_args
from rights_agent.store import (
    add_in_batches,
    chroma_client,
    create_collection,
    make_index_version,
    now_iso,
)

log = get_logger("pipelines.simple")


def extract_text(path: Path) -> str:
    """All pages concatenated.  Structure is discarded, deliberately."""
    text = load_corpus_text(path)
    return "\n".join(page.strip() for page in text.split("\f"))


def fixed_window_chunks(
    text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS
) -> list[tuple[int, str]]:
    """Slide a fixed window over the text.

    Returns ``(offset, chunk)`` pairs.  Overlap reduces -- does not remove --
    boundary loss: a definition split across two windows is still split, just
    less often.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must be in [0, size); got {overlap} with size {size}")
    stride = size - overlap
    chunks: list[tuple[int, str]] = []
    for offset in range(0, max(len(text), 1), stride):
        window = text[offset : offset + size]
        if not window.strip():
            continue
        chunks.append((offset, window))
        if offset + size >= len(text):
            break
    return chunks


def ingest_simple(
    settings: Settings,
    *,
    size: int = CHUNK_CHARS,
    overlap: int = OVERLAP_CHARS,
    batch_size: int = 256,
) -> dict[str, object]:
    """Write fixed windows to ``corpus_simple``.  Returns a manifest dict."""
    started = time.perf_counter()
    corpus = settings.corpus_path
    text = extract_text(corpus)
    chunks = fixed_window_chunks(text, size=size, overlap=overlap)

    embedder, embedder_name = resolve_embedder(settings)
    index_version = make_index_version(embedder_name, corpus_fingerprint(corpus))

    ids = [f"s{index:05d}" for index, _ in enumerate(chunks)]
    documents = [chunk for _, chunk in chunks]
    metadatas = [
        # Offset, length and pipeline name are all this pipeline knows.  There
        # is no citation field to fill in, and inventing one would be a lie.
        {"offset": offset, "chars": len(chunk), "pipeline": "simple", "index_version": index_version}
        for offset, chunk in chunks
    ]

    client = chroma_client(settings)
    collection = create_collection(
        client,
        SIMPLE_COLLECTION,
        embedder,
        {"index_version": index_version, "embedding_model": embedder_name, "pipeline": "simple"},
    )
    add_in_batches(collection, ids, documents, metadatas, batch_size=batch_size)

    expected = max(1, len(text) // (size - overlap))
    manifest = {
        "index_version": index_version,
        "embedding_model": embedder_name,
        "collection": SIMPLE_COLLECTION,
        "chunks": len(chunks),
        "expected_chunks": expected,
        "chars": len(text),
        "chunk_chars": size,
        "overlap_chars": overlap,
        "corpus_path": str(corpus),
        "corpus_sha": corpus_fingerprint(corpus),
        "built_at": now_iso(),
        "build_seconds": round(time.perf_counter() - started, 3),
    }
    settings.ensure_runs_dir()
    settings.simple_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log.info("wrote %d chunks to %s", len(chunks), SIMPLE_COLLECTION)
    return manifest


@operator_error_exit
def main(argv: list[str] | None = None) -> int:
    parser = build_parser("Build the fixed-window baseline index.")
    parser.add_argument("--chunk-chars", type=int, default=CHUNK_CHARS)
    parser.add_argument("--overlap-chars", type=int, default=OVERLAP_CHARS)
    args = parser.parse_args(argv)
    settings = settings_from_args(args)
    manifest = ingest_simple(
        settings,
        size=args.chunk_chars,
        overlap=args.overlap_chars,
        batch_size=args.batch_size,
    )
    for key in (
        "index_version",
        "embedding_model",
        "chunks",
        "expected_chunks",
        "chars",
        "build_seconds",
    ):
        print(f"{key:<19}{manifest[key]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
