"""Where the lab's on-disk artefacts live.

Anchored on the **working directory**, not on ``__file__``, and that is the
whole reason this module exists.

Every path in this package used to be written as
``Path(__file__).resolve().parents[3] / "fixtures"``, which is correct in a
source checkout -- ``src/attacklab/supplychain/x.py`` walks up to the repository
root -- and wrong everywhere the package is *installed*. In the container image
the package sits in ``site-packages``, so the same expression resolves to
``/opt/venv/lib/python3.13``, and ``docker compose run --rm scan`` died with

    PermissionError: [Errno 13] Permission denied:
    '/opt/venv/lib/python3.13/fixtures'

The image sets ``WORKDIR /app`` and copies ``data/`` and ``evals/`` there, and a
source checkout runs from the repository root, so the working directory is the
one anchor that is right in both. The ``__file__`` walk stays as a fallback for
the case nobody has thought of yet -- a test runner invoked from elsewhere --
and the marker check keeps it honest rather than silently returning ``/``.
"""

from __future__ import annotations

from pathlib import Path

#: What a project root looks like. ``pyproject.toml`` alone is not enough: a
#: parent directory of several checkouts could carry one.
MARKERS = ("pyproject.toml", "evals")


def _looks_like_root(path: Path) -> bool:
    return all((path / marker).exists() for marker in MARKERS)


def project_root() -> Path:
    """The directory holding ``data/``, ``evals/`` and ``fixtures/``.

    Tries the working directory and its parents first, then walks up from this
    file. Falls back to the working directory, because a wrong-but-writable
    path produces a legible error at the point of use, while raising here would
    take down an import.
    """
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if _looks_like_root(candidate):
            return candidate
    installed = Path(__file__).resolve()
    for candidate in installed.parents:
        if _looks_like_root(candidate):
            return candidate
    return here


def fixtures_dir() -> Path:
    """Generated model twins, skill manifests and the lockfile."""
    return project_root() / "fixtures"


def evals_dir() -> Path:
    """The datasets and thresholds the gate reads."""
    return project_root() / "evals"
