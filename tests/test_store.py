"""Manifest handling and the Chroma hygiene rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rights_agent.config import Settings
from rights_agent.embedding import HASHING_NAME, EmbedderError, HashingEmbedder
from rights_agent.store import (
    IndexManifest,
    IndexNotBuiltError,
    StoreError,
    add_in_batches,
    assert_unique_ids,
    flatten_metadata,
    load_manifest,
    make_index_version,
    pinned_embedder,
    require_manifest,
    write_manifest,
)


def manifest(**overrides: object) -> IndexManifest:
    base = dict(
        index_version="parser-3+hashing-bow-512+abcd1234",
        embedding_model=HASHING_NAME,
        parser_version="parser-3",
        corpus_path="/app/data/corpus.layout.txt",
        corpus_sha="abcd1234",
        collections={"corpus_leaves": 1087, "corpus_parents": 187},
        tree_counts={"section": 175},
        built_at="2026-08-28T00:00:00Z",
        build_seconds=0.8,
    )
    base.update(overrides)
    return IndexManifest(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# index_version
# --------------------------------------------------------------------------- #
def test_index_version_names_parser_embedder_and_corpus() -> None:
    version = make_index_version(HASHING_NAME, "abcd1234")
    assert version.count("+") == 2
    assert HASHING_NAME in version and "abcd1234" in version


# --------------------------------------------------------------------------- #
# Metadata hygiene
# --------------------------------------------------------------------------- #
def test_none_values_are_dropped_not_passed_to_chroma() -> None:
    """Chroma rejects ``None`` with a bare TypeError from its Rust bindings."""
    assert flatten_metadata({"a": None, "b": 1}) == {"b": 1}


def test_sequences_and_mappings_are_serialised() -> None:
    """Chroma accepts a list without complaint and then loses it."""
    out = flatten_metadata({"ids": ["b", "a"], "nested": {"x": 1}})
    assert json.loads(out["ids"]) == ["a", "b"]
    assert json.loads(out["nested"]) == {"x": 1}


def test_primitives_pass_through_unchanged() -> None:
    assert flatten_metadata({"s": "x", "i": 1, "f": 1.5, "b": True}) == {
        "s": "x",
        "i": 1,
        "f": 1.5,
        "b": True,
    }


def test_duplicate_ids_fail_loudly() -> None:
    """Chroma silently ignores a re-added id, so a collision is invisible."""
    assert_unique_ids(["a", "b"])
    with pytest.raises(StoreError, match="document-order ordinal"):
        assert_unique_ids(["a", "b", "a"])


def test_mismatched_row_lengths_are_rejected() -> None:
    with pytest.raises(StoreError, match="same length"):
        add_in_batches(object(), ["a", "b"], ["one"], [{}])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Manifest round-trip
# --------------------------------------------------------------------------- #
def test_manifest_round_trips(tmp_path: Path) -> None:
    settings = Settings.from_env().with_overrides(runs_dir=tmp_path)
    written = manifest()
    write_manifest(settings, written)
    assert load_manifest(settings) == written


def test_manifest_is_written_atomically(tmp_path: Path) -> None:
    """A crashed ingest must not leave a half-written manifest."""
    settings = Settings.from_env().with_overrides(runs_dir=tmp_path)
    write_manifest(settings, manifest())
    assert not list(tmp_path.glob("*.tmp"))


def test_missing_manifest_reads_as_none_but_requires_loudly(tmp_path: Path) -> None:
    settings = Settings.from_env().with_overrides(runs_dir=tmp_path)
    assert load_manifest(settings) is None
    with pytest.raises(IndexNotBuiltError, match="runs separately"):
        require_manifest(settings)


def test_unreadable_manifest_is_an_error_not_a_silent_none(tmp_path: Path) -> None:
    settings = Settings.from_env().with_overrides(runs_dir=tmp_path)
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    settings.manifest_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(StoreError, match="unreadable"):
        load_manifest(settings)


def test_a_manifest_missing_fields_says_to_rebuild(tmp_path: Path) -> None:
    settings = Settings.from_env().with_overrides(runs_dir=tmp_path)
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    settings.manifest_path.write_text(json.dumps({"index_version": "x"}), encoding="utf-8")
    with pytest.raises(StoreError, match="Rebuild the index"):
        load_manifest(settings)


def test_unknown_manifest_keys_are_preserved_not_rejected(tmp_path: Path) -> None:
    """A newer writer's extra fields must not break an older reader."""
    settings = Settings.from_env().with_overrides(runs_dir=tmp_path)
    write_manifest(settings, manifest())
    payload = json.loads(settings.manifest_path.read_text())
    payload["future_field"] = 42
    settings.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_manifest(settings)
    assert loaded is not None and loaded.extra["future_field"] == 42


# --------------------------------------------------------------------------- #
# Pinning
# --------------------------------------------------------------------------- #
def test_readers_reproduce_the_embedder_rather_than_choosing_one(tmp_path: Path) -> None:
    settings = Settings.from_env().with_overrides(runs_dir=tmp_path, embedder="onnx")
    write_manifest(settings, manifest())
    embedder, name = pinned_embedder(settings)
    assert name == HASHING_NAME, "the manifest wins over RIGHTS_EMBEDDER"
    assert isinstance(embedder, HashingEmbedder)


def test_a_manifest_naming_an_unbuildable_embedder_fails_fast(tmp_path: Path) -> None:
    settings = Settings.from_env().with_overrides(runs_dir=tmp_path)
    write_manifest(settings, manifest(embedding_model="word2vec-300"))
    with pytest.raises(EmbedderError, match="unknown embedder"):
        pinned_embedder(settings)
