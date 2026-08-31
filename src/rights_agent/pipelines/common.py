"""Shared argument handling for the ingest pipelines.

The embedding pipeline is operated separately from the query service -- it is
its own container and its own command -- so its CLI carries the switches that
decide *how the index is built*, and the query side never sees them.  It reads
what was built from the manifest instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rights_agent.config import Settings
from rights_agent.config import settings as load_settings
from rights_agent.embedding import HASHING_NAME, ONNX_NAME, get_embedder
from rights_agent.log import configure_logging, get_logger

log = get_logger("pipelines")


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="corpus file (.pdf or layout .txt); defaults to RIGHTS_CORPUS",
    )
    embedder = parser.add_mutually_exclusive_group()
    embedder.add_argument(
        "--embedder",
        choices=["auto", "onnx", "hashing"],
        default=None,
        help="which embedder to build with; defaults to RIGHTS_EMBEDDER",
    )
    embedder.add_argument(
        "--no-onnx",
        action="store_true",
        help="shorthand for --embedder hashing: fully offline, deterministic",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="where the index and manifests are written; defaults to RIGHTS_RUNS_DIR",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="rows embedded per Chroma call"
    )
    parser.add_argument("--quiet", action="store_true", help="log warnings and errors only")
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    """Apply CLI overrides on top of the environment."""
    configure_logging("WARNING" if getattr(args, "quiet", False) else None)
    resolved = load_settings()
    overrides: dict[str, object] = {}
    if getattr(args, "corpus", None):
        overrides["corpus_path"] = args.corpus.expanduser().resolve()
    if getattr(args, "runs_dir", None):
        overrides["runs_dir"] = args.runs_dir.expanduser().resolve()
    if getattr(args, "no_onnx", False):
        overrides["embedder"] = "hashing"
    elif getattr(args, "embedder", None):
        overrides["embedder"] = args.embedder
    return resolved.with_overrides(**overrides) if overrides else resolved


def resolve_embedder(settings: Settings):
    """Pick the embedder to *build* with, and say which one it is.

    Ingest chooses; every reader afterwards is pinned to this choice through the
    manifest.
    """
    embedder, name = get_embedder(prefer=settings.embedder)
    if name == HASHING_NAME and settings.embedder == "auto":
        log.warning(
            "building with the offline %s embedder: lexical, not semantic. "
            "Set RIGHTS_EMBEDDER=onnx once a model can be downloaded.",
            HASHING_NAME,
        )
    elif name == ONNX_NAME:
        log.info("building with %s", ONNX_NAME)
    return embedder, name
