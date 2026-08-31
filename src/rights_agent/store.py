"""Chroma access and the index manifest.

Everything that touches the vector store goes through here, for one reason:
``get_collection`` in Chroma 1.5.x returns the *default* ONNX embedding
function when none is passed, **even when the collection's persisted
configuration names a different one**.  Querying a hashed index with MiniLM
vectors does not raise -- it returns confident nonsense.  So this module always
passes the embedder explicitly and then cross-checks it against what the
collection recorded.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Documents, EmbeddingFunction
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from rights_agent.config import PARSER_VERSION, Settings
from rights_agent.embedding import build_embedder, embedder_name
from rights_agent.log import get_logger

log = get_logger("store")

MANIFEST_SCHEMA = 1


class IndexNotBuiltError(RuntimeError):
    """Raised when a reader finds no usable index.

    The message names the command that fixes it, because the person hitting
    this is usually running the query service, not the ingest job.
    """


class StoreError(RuntimeError):
    """Raised on a store inconsistency that must not be worked around."""


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
_clients: dict[str, chromadb.ClientAPI] = {}


def chroma_client(settings: Settings) -> chromadb.ClientAPI:
    """A persistent client for ``settings.chroma_dir``.

    Cached per path: Chroma keeps one system per directory per process and
    raises if a second client asks for the same path with different settings.
    """
    settings.ensure_runs_dir()
    key = str(settings.chroma_dir)
    client = _clients.get(key)
    if client is None:
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        try:
            client = chromadb.PersistentClient(
                path=str(settings.chroma_dir),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
        except Exception as exc:
            raise StoreError(
                f"could not open the Chroma index at {settings.chroma_dir}: {exc}. "
                "Chroma stores its index in SQLite, which fails on some network and "
                "virtualised mounts; point RIGHTS_RUNS_DIR at local disk."
            ) from exc
        _clients[key] = client
    return client


def reset_client_cache() -> None:
    """Forget cached clients (used by tests that move ``RIGHTS_RUNS_DIR``)."""
    _clients.clear()


# --------------------------------------------------------------------------- #
# Collections
# --------------------------------------------------------------------------- #
def create_collection(
    client: chromadb.ClientAPI,
    name: str,
    embedder: EmbeddingFunction[Documents],
    metadata: dict[str, Any] | None = None,
) -> Collection:
    """Drop and recreate ``name`` in cosine space.

    The embedder is passed inside ``configuration`` so that its name and config
    are persisted with the collection; that record is what
    :func:`open_collection` later verifies against.
    """
    try:
        client.delete_collection(name)
        log.debug("dropped existing collection %s", name)
    except Exception:  # noqa: BLE001, S110 - "does not exist" is the normal case
        pass
    return client.create_collection(
        name=name,
        configuration={"hnsw": {"space": "cosine"}, "embedding_function": embedder},
        metadata=flatten_metadata(metadata or {}) or None,
    )


def open_collection(
    client: chromadb.ClientAPI,
    name: str,
    embedder: EmbeddingFunction[Documents],
) -> Collection:
    """Open ``name`` for reading, verifying the embedder it was built with.

    Chroma raises on an embedding-function *conflict* when one is passed
    explicitly -- which is precisely why it always is here.  Passing none makes
    it fall back to its own default and say nothing.
    """
    try:
        collection = client.get_collection(name, embedding_function=embedder)
    except NotFoundError as exc:
        raise IndexNotBuiltError(
            f"collection {name!r} does not exist. Build the index first: "
            "`uv run rights-ingest` (or `docker compose run --rm ingest`)."
        ) from exc
    except ValueError as exc:
        if "mbedding function" in str(exc):
            raise StoreError(
                f"collection {name!r} rejected embedder {embedder_name(embedder)!r}: {exc}. "
                "Cross-embedder queries return confident nonsense rather than an error, "
                "so this is fatal. Set RIGHTS_EMBEDDER to match the index, or rebuild it."
            ) from exc
        raise StoreError(f"could not open collection {name!r}: {exc}") from exc

    recorded = _recorded_embedder_name(collection)
    resolved = embedder_name(embedder)
    if recorded and recorded != resolved:
        raise StoreError(
            f"collection {name!r} was built with embedder {recorded!r} but this process "
            f"opened it with {resolved!r}. Cross-embedder queries return confident "
            f"nonsense rather than an error, so this is fatal. Set RIGHTS_EMBEDDER to "
            f"match the index, or rebuild it."
        )
    return collection


def _recorded_embedder_name(collection: Collection) -> str:
    """Embedder name from the collection's persisted configuration, if any."""
    model = getattr(collection, "_model", None)
    configuration = getattr(model, "configuration_json", None) or {}
    entry = configuration.get("embedding_function") or {}
    name = entry.get("name", "")
    # Chroma writes ``default`` when it could not resolve a registered function;
    # that tells us nothing about what actually produced the vectors.
    return "" if name == "default" else str(name)


def collection_count(client: chromadb.ClientAPI, name: str) -> int:
    try:
        return client.get_collection(name).count()
    except Exception:  # noqa: BLE001 - missing collection means zero rows
        return 0


# --------------------------------------------------------------------------- #
# Metadata hygiene
# --------------------------------------------------------------------------- #
def flatten_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Coerce a metadata dict to Chroma's primitive value types.

    Chroma 1.5 accepts a list without complaint and rejects ``None`` with a
    bare ``TypeError`` from its Rust bindings, so neither is left to chance:
    ``None`` keys are dropped and sequences are JSON-encoded.
    """
    out: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (bool, int, float, str)):
            out[key] = value
        elif isinstance(value, (list, tuple, set)):
            out[key] = json.dumps(sorted(str(item) for item in value), ensure_ascii=False)
        elif isinstance(value, dict):
            out[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            out[key] = str(value)
    return out


def assert_unique_ids(ids: Sequence[str]) -> None:
    """Fail loudly on duplicate ids.

    Chroma 1.5 silently ignores a re-added id rather than raising, so a
    collision would show up as a quietly incomplete index -- rows missing with
    no error anywhere.
    """
    seen: set[str] = set()
    duplicates: list[str] = []
    for identifier in ids:
        if identifier in seen:
            duplicates.append(identifier)
        seen.add(identifier)
    if duplicates:
        raise StoreError(
            f"{len(duplicates)} duplicate chunk ids, e.g. {duplicates[:3]}. "
            "Ids must include a document-order ordinal: citation and page are not unique."
        )


def add_in_batches(
    collection: Collection,
    ids: Sequence[str],
    documents: Sequence[str],
    metadatas: Sequence[dict[str, Any]],
    batch_size: int = 256,
) -> None:
    """Add rows in batches, embedding as we go."""
    if not (len(ids) == len(documents) == len(metadatas)):
        raise StoreError("ids, documents and metadatas must be the same length")
    assert_unique_ids(ids)
    for start in range(0, len(ids), batch_size):
        stop = start + batch_size
        collection.add(
            ids=list(ids[start:stop]),
            documents=list(documents[start:stop]),
            metadatas=[flatten_metadata(m) for m in metadatas[start:stop]],
        )


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class IndexManifest:
    """What was built, from what, by what -- written once per ingest.

    ``index_version`` appears in every row's metadata, in the collection
    metadata, on every metrics row and on every experiment, which is what makes
    "which index produced this answer" answerable six months later.
    """

    index_version: str
    embedding_model: str
    parser_version: str
    corpus_path: str
    corpus_sha: str
    collections: dict[str, int]
    tree_counts: dict[str, int]
    built_at: str
    build_seconds: float
    schema: int = MANIFEST_SCHEMA
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, ensure_ascii=False)


def make_index_version(embedder_name: str, corpus_sha: str) -> str:
    """``<parser>+<embedder>+<corpus hash>`` (§9.4)."""
    return f"{PARSER_VERSION}+{embedder_name}+{corpus_sha}"


def write_manifest(settings: Settings, manifest: IndexManifest) -> Path:
    """Write the manifest atomically, so a crashed ingest leaves no half file."""
    settings.ensure_runs_dir()
    path = settings.manifest_path
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(manifest.to_json() + "\n", encoding="utf-8")
    os.replace(temporary, path)
    log.info("wrote %s (index_version=%s)", path, manifest.index_version)
    return path


def load_manifest(settings: Settings) -> IndexManifest | None:
    """Read the manifest, or ``None`` when no index has been built."""
    path = settings.manifest_path
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StoreError(f"index manifest at {path} is unreadable: {exc}") from exc
    known = set(IndexManifest.__dataclass_fields__)
    extra = {k: v for k, v in payload.items() if k not in known}
    filtered = {k: v for k, v in payload.items() if k in known}
    filtered["extra"] = {**filtered.get("extra", {}), **extra}
    try:
        return IndexManifest(**filtered)
    except TypeError as exc:
        raise StoreError(
            f"index manifest at {path} is missing fields ({exc}). Rebuild the index."
        ) from exc


def require_manifest(settings: Settings) -> IndexManifest:
    """The manifest, or a message telling the operator how to create one."""
    manifest = load_manifest(settings)
    if manifest is None:
        raise IndexNotBuiltError(
            f"no index manifest at {settings.manifest_path}. The embedding pipeline "
            "runs separately from the query service: build the index with "
            "`uv run rights-ingest`, or `docker compose run --rm ingest`."
        )
    return manifest


def pinned_embedder(
    settings: Settings, manifest: IndexManifest | None = None
) -> tuple[EmbeddingFunction[Documents], str]:
    """The embedder recorded in the manifest -- the only one safe to query with.

    This is the pinning rule (§10.2) in its enforceable form: a reader never
    *chooses* an embedder, it reproduces the one the index was built with.
    """
    manifest = manifest or require_manifest(settings)
    recorded = manifest.embedding_model
    return build_embedder(recorded), recorded


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
