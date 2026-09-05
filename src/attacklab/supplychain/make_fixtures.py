"""Build the demo-3 fixtures: same weights, two containers.

    uv run python -m attacklab.supplychain.make_fixtures

**On the payload, and this is a hard limit.**  ``tiny.pt`` contains a reduce
that would write **one marker file** into the lab's scratch directory and
nothing else.  It never runs -- :mod:`attacklab.supplychain.scan_model` reads
the opcode stream with ``pickletools.genops`` and never calls ``pickle.load`` --
and the fixture exists so the scanner has something true to find.  **Do not
extend it.**  The detector is the deliverable; the payload is a fixture.

Generated rather than committed, for two reasons.  A binary with a reduce in it
is exactly the thing a repository scanner should complain about, and having it
appear in a checkout is how a lab loses its welcome on a corporate laptop.  And
generating it means the reader can see, in Python, precisely what the payload
is -- which is a stronger safety claim than a hash in a README.
"""

from __future__ import annotations

import argparse
import json

# Writing a fixture. Nothing in this repository calls pickle.load or
# pickle.loads -- the scanner reads opcodes with pickletools.genops instead,
# and tests/test_supplychain.py asserts that absence.
import pickle
import struct
from pathlib import Path

from attacklab.paths import fixtures_dir


def _models_dir() -> Path:
    """Resolved on use rather than at import.

    See :mod:`attacklab.paths` for why a ``__file__`` walk is wrong once the
    package is installed.
    """
    return fixtures_dir() / "models"

#: The scratch path the payload would touch, and the only one.
MARKER_NAME = "attacklab-pickle-marker.txt"

#: The "weights". Four floats, so both containers hold byte-identical numbers
#: and the demo line -- *same model, different container* -- is literally true.
WEIGHTS: dict[str, list[float]] = {
    "layer0.weight": [0.5, -0.25, 0.125, 0.0625],
    "layer0.bias": [0.0, 0.0],
}


class _Marker:
    """A benign reduce.

    ``__reduce__`` is what makes a pickle stream executable, and this one
    returns a call that appends one line to one file under the system temporary
    directory.  Written out longhand rather than using ``os.system`` so that a
    reader can confirm the blast radius by looking at it.
    """

    def __reduce__(self):  # type: ignore[no-untyped-def]
        import tempfile

        target = str(Path(tempfile.gettempdir()) / MARKER_NAME)
        # (function, args) -- the pair a pickle REDUCE opcode will call.
        return (_write_marker, (target,))


def _write_marker(target: str) -> str:
    """What the payload would do.  One line, one file, no network."""
    Path(target).write_text(
        "attacklab: this line proves a pickle payload executed on load.\n", encoding="utf-8"
    )
    return target


def write_pickle_model(path: Path) -> Path:
    """A ``.pt``-shaped artefact whose opcode stream carries a reduce."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": WEIGHTS, "_metadata": _Marker()}
    path.write_bytes(pickle.dumps(payload, protocol=4))
    return path


def write_safetensors_model(path: Path) -> Path:
    """The same weights in the safetensors container.

    Written by hand rather than with the ``safetensors`` package, because the
    format is a JSON header length, a JSON header and a tensor blob -- and
    writing it out is a clearer demonstration that there is **no code path in
    the container at all** than importing a library that hides it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = b""
    header: dict[str, object] = {}
    for name, values in WEIGHTS.items():
        data = struct.pack(f"<{len(values)}f", *values)
        header[name] = {
            "dtype": "F32",
            "shape": [len(values)],
            "data_offsets": [len(blob), len(blob) + len(data)],
        }
        blob += data
    header["__metadata__"] = {"format": "pt", "source": "attacklab fixture"}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # 8-byte little-endian header length, then the header, then the tensors.
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + blob)
    return path


#: The lockfile's three entries, and why each is shaped the way it is.
#:
#: One correct, one with a digest that has moved, and one pinned by a mutable
#: tag.  All three are **deliberately** what they are: demo 3's fourth beat is a
#: failing lockfile check, and a lockfile where everything matches demonstrates
#: nothing.  Each entry carries a ``_note`` saying so, because the alternative
#: is somebody finding a wrong digest in a repository and fixing it.
def write_lockfile(models_dir: Path, lock_dir: Path | None = None) -> Path:
    """Write ``fixtures/lock/models.lock.json`` with its two deliberate faults."""
    from attacklab.supplychain.scan_lock import digest_of

    lock_dir = lock_dir or (models_dir.parent / "lock")
    lock_dir.mkdir(parents=True, exist_ok=True)
    safetensors = models_dir / "tiny.safetensors"
    entries = [
        {
            "name": "tiny-model-pickle",
            "path": "models/tiny.pt",
            "pinned_by": "digest",
            # Wrong on purpose. This is demo 3's fourth beat.
            "sha256": "0" * 64,
            "_note": (
                "Digest deliberately wrong, so the lockfile check fails on stage. Do "
                "not 'correct' it: a lockfile where everything matches demonstrates "
                "nothing."
            ),
        },
        {
            "name": "tiny-model-safetensors",
            "path": "models/tiny.safetensors",
            "pinned_by": "digest",
            "sha256": digest_of(safetensors),
            "_note": "Correct, so a passing row is visible next to the failing ones.",
        },
        {
            "name": "sentence-transformer-minilm",
            "path": "models/tiny.safetensors",
            "pinned_by": "tag",
            "tag": "latest",
            "_note": (
                "Pinned by a mutable tag rather than a digest. A tag is a pointer "
                "somebody else can repoint, and this is the finding, not a mistake in "
                "the fixture."
            ),
        },
    ]
    path = lock_dir / "models.lock.json"
    path.write_text(
        json.dumps({"artefacts": entries}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def build(models_dir: Path | None = None) -> tuple[Path, Path, Path]:
    """Both model containers and the lockfile.  Returns their paths."""
    models_dir = models_dir or _models_dir()
    pickle_path = write_pickle_model(models_dir / "tiny.pt")
    safe_path = write_safetensors_model(models_dir / "tiny.safetensors")
    return pickle_path, safe_path, write_lockfile(models_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    pickle_path, safe_path, lock_path = build(args.models_dir)
    print(f"wrote {pickle_path}  ({pickle_path.stat().st_size} bytes, pickle, carries a reduce)")
    print(f"wrote {safe_path}  ({safe_path.stat().st_size} bytes, safetensors, no code path)")
    print(f"wrote {lock_path}  (one correct digest, one moved, one mutable tag)")
    print()
    print("Same weights. Different container. Nothing in this repository loads either.")
    print("The lockfile's two faults are deliberate: they are demo 3's fourth beat.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
