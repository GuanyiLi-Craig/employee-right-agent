"""Inspect a model artefact's pickle opcode stream **without executing it**.

``pickletools.genops`` reads the opcodes; ``pickle.load`` runs them.  This
module never calls the second one, and that is the whole design.

**Be precise about the current PyTorch position, because the naive version of
this argument is out of date and a knowledgeable audience will catch it.**
Since **PyTorch 2.6, ``torch.load`` defaults to ``weights_only=True``** -- a
restricted unpickler that permits only the globals needed to rebuild a state
dict, with anything else requiring an explicit ``add_safe_globals`` allowlist.
That is a real, meaningful improvement and the lab says so out loud.

Three things nonetheless remain true, and they are what the scanner is for:

1. **The format is still pickle-based.**  The container did not change; only the
   default loader behaviour did.
2. **Much of the ecosystem passes ``weights_only=False`` anyway**, because older
   checkpoints will not load otherwise and that is the flag which makes the
   error message go away.  So the scanner flags the **call site** as well as the
   artefact -- :func:`scan_call_sites` is a grep with a high hit rate.
3. **Bypasses of the restricted unpickler have been reported.**  A restricted
   unpickler is a smaller attack surface, not the absence of one.

So the recommendation is not "set the flag".  It is **prefer safetensors**,
which removes the class of problem rather than restricting it.

One more thing worth saying while it runs, because it is what makes this worth
writing rather than buying: the main open-source pickle scanner has itself
carried bypass vulnerabilities rated 9.3 -- extension-based evasion, ZIP CRC
handling, and subclassed module paths that dodge the blocklist.  A blocklist
under adversarial pressure is a layer-one control, with all the properties that
implies.
"""

from __future__ import annotations

import pickletools
import re
import zipfile
from io import BytesIO
from pathlib import Path

from attacklab.supplychain.findings import FAIL, PASS, WARN, Finding

#: Globals a state dict legitimately needs.  Anything else is a finding.
#:
#: An allowlist rather than a blocklist, and the distinction is the module's
#: whole argument: a blocklist under adversarial pressure enumerates the attacks
#: someone thought of, and the scanners that took that approach have their own
#: CVEs to show for it.
SAFE_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        ("collections", "OrderedDict"),
        ("torch", "FloatStorage"),
        ("torch", "LongStorage"),
        ("torch", "HalfStorage"),
        ("torch", "IntStorage"),
        ("torch", "BFloat16Storage"),
        ("torch._utils", "_rebuild_tensor_v2"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
    }
)

#: Opcodes that can call arbitrary code when the stream is loaded.
DANGEROUS_OPCODES: frozenset[str] = frozenset({"REDUCE", "BUILD", "INST", "OBJ", "NEWOBJ"})

#: Extensions that carry a pickle stream.
PICKLE_SUFFIXES: frozenset[str] = frozenset({".pt", ".pth", ".bin", ".ckpt", ".pkl", ".pickle"})
SAFE_SUFFIXES: frozenset[str] = frozenset({".safetensors"})

#: The call-site grep.
#:
#: Requires ``load`` on the same line, and skips this package. The first version
#: matched the flag name anywhere and reported four findings against *this
#: file* -- its own docstring and its own remedy strings. A scanner whose
#: loudest finding is itself is a scanner people learn to ignore.
_WEIGHTS_ONLY_FALSE = re.compile(r"\bload\s*\([^)]*weights_only\s*=\s*False")

#: Paths the call-site grep skips: this package writes *about* the flag.
_GREP_SKIP_PARTS = frozenset(
    {".venv", "node_modules", "__pycache__", ".git", "supplychain"}
)


def _pickle_streams(path: Path) -> list[tuple[str, bytes]]:
    """Every pickle stream in the artefact, named.

    A ``.pt`` file is usually a zip containing ``data.pkl``.  Reading only the
    outer bytes would find nothing, which is how a scanner reports a clean bill
    of health on a hostile file.
    """
    raw = path.read_bytes()
    if zipfile.is_zipfile(BytesIO(raw)):
        streams: list[tuple[str, bytes]] = []
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            for name in archive.namelist():
                if name.endswith((".pkl", ".pickle")):
                    streams.append((name, archive.read(name)))
        return streams
    return [(path.name, raw)]


def _globals_and_opcodes(stream: bytes) -> tuple[list[tuple[str, str]], list[str]]:
    """Read the opcode stream.  Never loads it."""
    found_globals: list[tuple[str, str]] = []
    found_opcodes: list[str] = []
    pending: list[str] = []
    for opcode, argument, _position in pickletools.genops(BytesIO(stream)):
        name = opcode.name
        if name in DANGEROUS_OPCODES:
            found_opcodes.append(name)
        if name == "GLOBAL" and isinstance(argument, str):
            module, _, attribute = argument.partition(" ")
            found_globals.append((module, attribute))
        elif name == "STACK_GLOBAL":
            # The module and attribute were pushed as two separate strings.
            if len(pending) >= 2:
                found_globals.append((pending[-2], pending[-1]))
        elif name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "SHORT_BINSTRING"}:
            if isinstance(argument, str):
                pending.append(argument)
    return found_globals, found_opcodes


def scan_artefact(path: Path) -> list[Finding]:
    """One artefact: report its format, and what its opcode stream references."""
    suffix = path.suffix.lower()
    if suffix in SAFE_SUFFIXES:
        return [
            Finding(
                scanner="scan_model",
                severity=PASS,
                subject=path.name,
                summary=(
                    "safetensors: a tensor container with no code path. Same weights as "
                    "the .pt twin, and nothing to execute."
                ),
                detail={"format": "safetensors", "bytes": path.stat().st_size},
            )
        ]
    if suffix not in PICKLE_SUFFIXES:
        return [
            Finding(
                scanner="scan_model",
                severity=WARN,
                subject=path.name,
                summary=f"unrecognised extension {suffix!r}; not inspected",
                remedy="Name model artefacts with a known extension, or extend PICKLE_SUFFIXES.",
            )
        ]

    findings: list[Finding] = [
        Finding(
            scanner="scan_model",
            severity=WARN,
            subject=path.name,
            summary=(
                "pickle-based format. Since PyTorch 2.6 torch.load defaults to "
                "weights_only=True, which is a real improvement -- but the container is "
                "still pickle, and restricted unpicklers have been bypassed."
            ),
            detail={"format": "pickle", "bytes": path.stat().st_size},
            remedy="Prefer safetensors: it removes the class of problem rather than restricting it.",
        )
    ]
    for name, stream in _pickle_streams(path):
        try:
            found_globals, found_opcodes = _globals_and_opcodes(stream)
        except Exception as exc:  # noqa: BLE001 - an unreadable stream is a finding
            findings.append(
                Finding(
                    scanner="scan_model",
                    severity=FAIL,
                    subject=f"{path.name}:{name}",
                    summary=f"the opcode stream could not be read: {type(exc).__name__}: {exc}",
                    remedy=(
                        "Treat an unparseable artefact as hostile. A stream a scanner "
                        "cannot read is a stream a loader might still execute."
                    ),
                )
            )
            continue
        unexpected = sorted({g for g in found_globals if g not in SAFE_GLOBALS})
        if unexpected:
            findings.append(
                Finding(
                    scanner="scan_model",
                    severity=FAIL,
                    subject=f"{path.name}:{name}",
                    summary=(
                        "the opcode stream references globals a state dict does not need: "
                        + ", ".join(f"{module}.{attribute}" for module, attribute in unexpected)
                    ),
                    detail={"globals": [f"{m}.{a}" for m, a in unexpected]},
                    remedy="Do not load this artefact. Obtain it as safetensors, or from a source you can verify.",
                )
            )
        if found_opcodes:
            findings.append(
                Finding(
                    scanner="scan_model",
                    severity=FAIL,
                    subject=f"{path.name}:{name}",
                    summary=(
                        "the opcode stream contains "
                        + ", ".join(sorted(set(found_opcodes)))
                        + " -- opcodes that call out when the stream is loaded"
                    ),
                    detail={"opcodes": sorted(set(found_opcodes))},
                    remedy="Do not load this artefact.",
                )
            )
    return findings


def scan_call_sites(root: Path) -> list[Finding]:
    """Flag ``weights_only=False`` anywhere in a source tree.

    The artefact is half the problem.  The other half is the call site: much of
    the ecosystem passes this flag because older checkpoints will not load
    without it, and that turns a restricted unpickler back into an unrestricted
    one.  A five-minute grep with a high hit rate.
    """
    findings: list[Finding] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in _GREP_SKIP_PARTS for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if _WEIGHTS_ONLY_FALSE.search(line):
                findings.append(
                    Finding(
                        scanner="scan_model",
                        severity=FAIL,
                        subject=f"{path}:{number}",
                        summary="weights_only=False turns the restricted unpickler back off",
                        remedy=(
                            "Convert the checkpoint to safetensors. If you truly cannot, "
                            "add the specific globals with add_safe_globals and say why in a comment."
                        ),
                    )
                )
    if not findings:
        findings.append(
            Finding(
                scanner="scan_model",
                severity=PASS,
                subject=str(root),
                summary="no weights_only=False call sites",
            )
        )
    return findings


def scan(models_dir: Path, source_root: Path | None = None) -> list[Finding]:
    """Every artefact in ``models_dir``, plus the call-site grep."""
    findings: list[Finding] = []
    if not models_dir.is_dir():
        return [
            Finding(
                scanner="scan_model",
                severity=WARN,
                subject=str(models_dir),
                summary="no model directory; nothing inspected",
                remedy="Build the fixtures with `make fixtures`.",
            )
        ]
    for path in sorted(models_dir.iterdir()):
        if path.is_file():
            findings.extend(scan_artefact(path))
    if source_root is not None:
        findings.extend(scan_call_sites(source_root))
    return findings
