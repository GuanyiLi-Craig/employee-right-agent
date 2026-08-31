"""The hierarchical embedding pipeline: tree → two collections.

A subsection lifted out of a structured document is unfindable on its own.
*"The threshold is £500"* retrieves for nothing: the words that make it
findable -- the topic, the jurisdiction, the provision it belongs to -- live in
its **ancestors**.  So this pipeline rebuilds the tree the authors wrote and
**prepends each node's breadcrumb to the text it embeds**.  Not stored beside --
embedded *with*.

Two collections:

===================  ==================================  ========  ============
Collection           One row per                         Searched  Purpose
===================  ==================================  ========  ============
``corpus_leaves``    subsection (or provision if none)   yes       precision
``corpus_parents``   section / inserted provision        no        widening
===================  ==================================  ========  ============

Ids carry a document-order ordinal (``l00042::s.7(3)``).  Citation plus page is
*not* unique -- a section's ``(3)`` and an inserted provision's ``(3)`` can
share a page -- and Chroma 1.5 silently ignores a duplicate id rather than
raising, so a collision would show up as a quietly incomplete index.  Tree
order is deterministic, so the ordinals are stable between builds and ids stay
diffable.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from rights_agent.config import (
    LEAF_COLLECTION,
    PARENT_COLLECTION,
    PARSER_VERSION,
    Settings,
)
from rights_agent.document.nodes import (
    KIND_HEADING,
    KIND_PART,
    KIND_SCHEDULE,
    Node,
)
from rights_agent.document.nodes import (
    leaves as tree_leaves,
)
from rights_agent.document.nodes import (
    provisions as tree_provisions,
)
from rights_agent.document.parser import (
    ParseResult,
    corpus_fingerprint,
    parse_corpus,
    validate_tree,
)
from rights_agent.entrypoints import operator_error_exit
from rights_agent.log import get_logger
from rights_agent.pipelines.common import build_parser, resolve_embedder, settings_from_args
from rights_agent.store import (
    IndexManifest,
    add_in_batches,
    chroma_client,
    create_collection,
    make_index_version,
    now_iso,
    write_manifest,
)

log = get_logger("pipelines.hierarchical")


def _container_titles(node: Node) -> tuple[str, str]:
    """``(part label, cross-heading title)`` for metadata filtering."""
    part = node.ancestor(KIND_PART, KIND_SCHEDULE)
    heading = node.ancestor(KIND_HEADING)
    return (part.label() if part else "", heading.title if heading else "")


def _ordinals(tree: Node) -> dict[int, int]:
    """Document-order index for every node, by identity.

    Keyed on ``id()`` because :class:`Node` is deliberately unhashable-by-value
    (it compares by identity) and building a second index keyed on citation
    would reintroduce the collision this exists to avoid.
    """
    return {id(node): index for index, node in enumerate(tree.walk())}


def build_rows(
    result: ParseResult, index_version: str
) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    """Turn a tree into ``(leaf rows, parent rows)`` ready for Chroma."""
    tree = result.tree
    ordinals = _ordinals(tree)

    parent_ids: dict[int, str] = {}
    parents: dict[str, list[Any]] = {"ids": [], "documents": [], "metadatas": []}
    for provision in tree_provisions(tree):
        ordinal = ordinals[id(provision)]
        identifier = f"p{ordinal:05d}::{provision.citation()}"
        parent_ids[id(provision)] = identifier
        breadcrumb = provision.breadcrumb()
        full_text = provision.full_text()
        part, heading = _container_titles(provision)
        parents["ids"].append(identifier)
        # Parents are never searched, but they are embedded anyway: keeping one
        # code path means a future change of mind (searching both tiers, or
        # re-ranking) needs no re-ingest.
        parents["documents"].append(f"{breadcrumb}\n{full_text}")
        parents["metadatas"].append(
            {
                "citation": provision.citation(),
                "breadcrumb": breadcrumb,
                "kind": provision.kind,
                "section_number": provision.number,
                "section_title": provision.title,
                "part": part,
                "heading": heading,
                "page": provision.page,
                "chars": len(full_text),
                "child_count": len(provision.children),
                "raw_text": full_text,
                "host_document": provision.host_document,
                "inserted_by": provision.inserted_by,
                "index_version": index_version,
            }
        )

    leaves: dict[str, list[Any]] = {"ids": [], "documents": [], "metadatas": []}
    for node in tree_leaves(tree):
        ordinal = ordinals[id(node)]
        breadcrumb = node.breadcrumb()
        raw_text = node.own_text()
        provision = node.enclosing_provision()
        part, heading = _container_titles(node)
        leaves["ids"].append(f"l{ordinal:05d}::{node.citation()}")
        # The breadcrumb IS the embedded text, not metadata beside it.
        leaves["documents"].append(f"{breadcrumb}\n{raw_text}")
        leaves["metadatas"].append(
            {
                "citation": node.citation(),
                "breadcrumb": breadcrumb,
                "parent_id": parent_ids.get(id(provision), ""),
                "part": part,
                "heading": heading,
                "section_number": provision.number if provision else "",
                "section_title": provision.title if provision else "",
                "kind": node.kind,
                "page": node.page,
                "chars": len(raw_text),
                # Leaf text *without* the breadcrumb, for prompt assembly: the
                # breadcrumb belongs in the citation line, not repeated in the
                # quoted provision.
                "raw_text": raw_text,
                "index_version": index_version,
            }
        )
    return leaves, parents


def ingest(settings: Settings, *, batch_size: int = 256) -> IndexManifest:
    """Parse, validate and write both collections.  Returns the manifest."""
    started = time.perf_counter()
    corpus = settings.corpus_path
    result = parse_corpus(corpus)
    counts = validate_tree(result.tree)

    embedder, embedder_name = resolve_embedder(settings)
    index_version = make_index_version(embedder_name, corpus_fingerprint(corpus))

    leaves, parents = build_rows(result, index_version)
    if not leaves["ids"]:
        raise RuntimeError("the parser produced no leaves; refusing to write an empty index")

    client = chroma_client(settings)
    collection_metadata = {
        "index_version": index_version,
        "embedding_model": embedder_name,
        "parser_version": PARSER_VERSION,
        "corpus": corpus.name,
    }
    for name, rows in ((LEAF_COLLECTION, leaves), (PARENT_COLLECTION, parents)):
        collection = create_collection(
            client, name, embedder, {**collection_metadata, "tier": name}
        )
        add_in_batches(
            collection, rows["ids"], rows["documents"], rows["metadatas"], batch_size=batch_size
        )
        log.info("wrote %d rows to %s", len(rows["ids"]), name)

    manifest = IndexManifest(
        index_version=index_version,
        embedding_model=embedder_name,
        parser_version=PARSER_VERSION,
        corpus_path=str(corpus),
        corpus_sha=corpus_fingerprint(corpus),
        collections={
            LEAF_COLLECTION: len(leaves["ids"]),
            PARENT_COLLECTION: len(parents["ids"]),
        },
        tree_counts=counts,
        built_at=now_iso(),
        build_seconds=round(time.perf_counter() - started, 3),
        extra={"pages": result.pages, "parser_name": result.parser_name},
    )
    write_manifest(settings, manifest)
    return manifest


@operator_error_exit
def main(argv: list[str] | None = None) -> int:
    parser = build_parser("Build the hierarchical index (leaves + parents).")
    args = parser.parse_args(argv)
    settings = settings_from_args(args)
    manifest = ingest(settings, batch_size=args.batch_size)
    print(f"index_version      {manifest.index_version}")
    print(f"embedding_model    {manifest.embedding_model}")
    print(f"corpus             {Path(manifest.corpus_path).name} ({manifest.corpus_sha})")
    for name, count in sorted(manifest.collections.items()):
        print(f"{name:<19}{count} rows")
    print(
        "tree               "
        + ", ".join(f"{kind}={count}" for kind, count in sorted(manifest.tree_counts.items()))
    )
    print(f"built in           {manifest.build_seconds}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
