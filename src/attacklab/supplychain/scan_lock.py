"""Verify every model and dataset is pinned by digest, and that the digest matches.

**A tag is a mutable pointer.**  ``:latest`` and ``:v2.1`` are names someone
else can repoint, and the whole value of a lockfile is that it names bytes
rather than intentions.

Two checks, and the second is the one that catches a real attack:

1. every entry carries a ``sha256`` digest rather than only a tag;
2. the digest still matches the artefact on disk.

The lab's own dependency on the assistant is the same argument applied to
itself: the lab makes claims about a specific system's behaviour, and an
unpinned dependency means the claims are about whatever was on main that
morning.  Here they are the same commit, which is a stronger pin than a git ref.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from attacklab.supplychain.findings import FAIL, PASS, WARN, Finding


def digest_of(path: Path) -> str:
    """SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan(lockfile: Path, root: Path | None = None) -> list[Finding]:
    """Check every entry in ``lockfile``."""
    if not lockfile.is_file():
        return [
            Finding(
                scanner="scan_lock",
                severity=FAIL,
                subject=str(lockfile),
                summary="no lockfile: nothing is pinned",
                remedy="Write one. An unpinned model is a mutable dependency with no changelog.",
            )
        ]
    root = root or lockfile.parent.parent
    try:
        entries = json.loads(lockfile.read_text(encoding="utf-8")).get("artefacts") or []
    except (OSError, json.JSONDecodeError) as exc:
        return [
            Finding(
                scanner="scan_lock",
                severity=FAIL,
                subject=str(lockfile),
                summary=f"lockfile is unreadable: {exc}",
            )
        ]

    findings: list[Finding] = []
    for entry in entries:
        name = str(entry.get("name") or "?")
        recorded = str(entry.get("sha256") or "")
        relative = str(entry.get("path") or "")
        if not recorded:
            findings.append(
                Finding(
                    scanner="scan_lock",
                    severity=FAIL,
                    subject=name,
                    summary=(
                        f"pinned by tag {entry.get('tag') or '(none)'!r} and not by digest"
                    ),
                    remedy="Record a sha256. A tag is a mutable pointer.",
                )
            )
            continue
        path = (root / relative).resolve()
        if not path.is_file():
            findings.append(
                Finding(
                    scanner="scan_lock",
                    severity=WARN,
                    subject=name,
                    summary=f"pinned by digest but absent from disk at {relative}",
                    remedy="Build the fixtures with `make fixtures`.",
                )
            )
            continue
        actual = digest_of(path)
        if actual != recorded:
            findings.append(
                Finding(
                    scanner="scan_lock",
                    severity=FAIL,
                    subject=name,
                    summary=(
                        f"digest moved: lockfile says {recorded[:12]}…, the file is "
                        f"{actual[:12]}…"
                    ),
                    detail={"expected": recorded, "actual": actual, "path": relative},
                    remedy=(
                        "Do not deploy. Find out why the bytes changed before you "
                        "update the lockfile -- updating it first is how a compromise "
                        "becomes a commit."
                    ),
                )
            )
            continue
        findings.append(
            Finding(
                scanner="scan_lock",
                severity=PASS,
                subject=name,
                summary=f"pinned by digest {recorded[:12]}… and matching",
            )
        )
    if not findings:
        findings.append(
            Finding(
                scanner="scan_lock",
                severity=WARN,
                subject=str(lockfile),
                summary="lockfile contains no artefacts",
            )
        )
    return findings


def write(lockfile: Path, artefacts: dict[str, Path], root: Path) -> Path:
    """Generate a lockfile from what is on disk.  Used by ``make fixtures``."""
    entries = [
        {
            "name": name,
            "path": str(path.relative_to(root)),
            "sha256": digest_of(path),
            "pinned_by": "digest",
        }
        for name, path in sorted(artefacts.items())
    ]
    lockfile.parent.mkdir(parents=True, exist_ok=True)
    lockfile.write_text(
        json.dumps({"artefacts": entries}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return lockfile
