"""Generate ``evals/adversarial.jsonl`` from the catalogue.

    uv run python -m attacklab.attacks.dataset

**Derived, not authored.**  The rows are one per payload-by-control pair, and the
source of truth is :mod:`attacklab.attacks.catalogue` -- so the file cannot
drift from the expectations the gate actually asserts.  It is committed anyway,
for two reasons: a reviewer can read the whole containment matrix in one place
without running anything, and a diff on it makes a change in expectations
visible in a pull request rather than buried in a dataclass.

``make adversarial`` asserts against the catalogue, not against this file.  A
gate that read a generated artefact would be asserting that the generator ran.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attacklab.attacks.catalogue import RUNNABLE, Payload
from attacklab.paths import evals_dir
from attacklab.registry import RUNTIME_KEYS


def _dataset_path() -> Path:
    """Resolved on use, not at import: see :mod:`attacklab.paths`."""
    return evals_dir() / "adversarial.jsonl"


def rows(payloads: tuple[Payload, ...] = RUNNABLE) -> list[dict[str, object]]:
    """One row per payload and runtime control, plus one bare row per payload."""
    out: list[dict[str, object]] = []
    for payload in payloads:
        out.append(
            {
                "id": f"{payload.id}:bare",
                "payload": payload.id,
                "control": None,
                "vector": payload.vector.value,
                "goal": payload.goal.value,
                "owasp": payload.owasp,
                "asi": payload.asi,
                "expect": "contained" if payload.expect_bare_containment else "escapes",
                "because": payload.expect_bare_containment
                or "no control is on, so nothing is between the payload and the tool",
                "question": payload.question,
            }
        )
        for key in sorted(RUNTIME_KEYS):
            if key in payload.expect_blocked_by:
                expect, because = "contained", f"{key} stops this payload class"
            elif key in payload.expect_evades:
                expect, because = "escapes", f"{key} does not see this payload class"
            else:
                # Unclaimed either way: the catalogue makes no assertion, and the
                # dataset says so rather than guessing. Silence is a legitimate
                # position for a control nobody has measured against a payload.
                expect, because = "unclaimed", "the catalogue makes no claim here"
            out.append(
                {
                    "id": f"{payload.id}:{key}",
                    "payload": payload.id,
                    "control": key,
                    "vector": payload.vector.value,
                    "goal": payload.goal.value,
                    "owasp": payload.owasp,
                    "asi": payload.asi,
                    "expect": expect,
                    "because": because,
                    "question": payload.question,
                }
            )
    return out


def write(path: Path | None = None) -> Path:
    path = path or _dataset_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(row, sort_keys=True) for row in rows())
    path.write_text(body + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the committed file is out of date rather than rewriting it",
    )
    args = parser.parse_args(argv)
    generated = "\n".join(json.dumps(row, sort_keys=True) for row in rows()) + "\n"
    target = args.out or _dataset_path()
    if args.check:
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if current != generated:
            print(f"{target} is out of date; regenerate with `make dataset`")
            return 1
        print(f"{target} is up to date ({len(rows())} rows)")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(generated, encoding="utf-8")
    print(f"wrote {target} ({len(rows())} rows)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
