"""Run model-produced Python in a subprocess jail.

Slide 10's right-hand column: what happens when the answer stops being text and
starts being an action.  ``LLM10 Improper Output Handling`` fell five places in
2026, not because it stopped happening but because everything above it got
worse.

Scope is deliberately small -- one tool, ``run_snippet`` -- because everything
about *how* it runs is the lesson.  Five limits, and slide 10 notes that none of
them are AI-specific:

* **Subprocess**, never ``exec`` in-process.  No shared interpreter state.
* **``resource.setrlimit``** for CPU seconds and address space, with a
  wall-clock timeout on top, because a process blocked on I/O burns no CPU.
* **Empty environment.**  No inherited variables, no API keys, no tokens.  *If
  the sandbox holds a long-lived credential, it is not a boundary -- it is a
  container for your credentials.*  Secrets are brokered with a short TTL and
  never placed here.
* **A per-task scratch directory**, wiped after.  No shared volume with the app.
* **Network denied.**  In Compose the sandbox service takes
  ``network_mode: none``.  The allowlist is by *destination* rather than by
  protocol, because exfiltration does not need a shell -- a DNS lookup or an
  image fetch is plenty.

**State the limit plainly, and the README repeats it: a container is not a
security boundary against genuinely hostile code.**  The production answer is a
separate kernel or a microVM.  This demonstrates the *shape* of the control on a
laptop.  Nobody should leave thinking a Docker container is sufficient isolation
for untrusted code.

Three real failure shapes are encoded as fixtures in
:data:`FAILURE_FIXTURES`, because they are the ones that break naive
sandboxing -- and the third is the clearest example in the whole lab of a
control that *looks* deterministic and is not.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from rights_agent.log import get_logger

log = get_logger("attacklab.sandbox")

#: CPU seconds, address space and wall clock.  Numbers in the configuration --
#: which is the whole point of ``LLM06``: a ceiling is not a judgement about
#: text, and it is a denial-of-wallet control as much as a denial-of-service one.
CPU_SECONDS = 2
ADDRESS_SPACE_BYTES = 256 * 1024 * 1024
WALL_CLOCK_SECONDS = 5.0
MAX_OUTPUT_CHARS = 4_000
#: Largest file a snippet may write into its scratch directory.
MAX_FILE_BYTES = 1024 * 1024

#: Prefix the child uses to report which limits took effect. Stripped from
#: stderr by the parent and surfaced separately, so an audience sees the limits
#: that are real rather than the ones the code asked for.
LIMIT_MARKER = "__attacklab_limits__:"

#: Applied inside the child, before it reaches the snippet.
#:
#: In the child rather than via ``preexec_fn`` because that hook is documented
#: as unsafe with threads, and the console is threaded.
#:
#: **Every limit is applied defensively, and the child reports which ones took
#: effect.**  The first version set ``RLIMIT_AS`` to a flat 256MB and every
#: snippet died with ``ValueError: current limit exceeds maximum limit`` -- on
#: this host the hard limit is lower than the number I picked.  The sandbox
#: reported ``ok=False`` for ``print("hello")``, which would have read on stage
#: as a working jail.  A boundary that fails closed for the wrong reason is
#: indistinguishable from one that works, and that is worse than one that
#: visibly does not: so the child clamps each limit to the host's hard limit,
#: skips what the platform refuses, and says which is which.
_PREAMBLE = f"""\
import resource, sys

_applied = []
_skipped = []

def _limit(name, want):
    number = getattr(resource, name, None)
    if number is None:
        _skipped.append(name + " (not on this platform)")
        return
    try:
        soft, hard = resource.getrlimit(number)
    except (ValueError, OSError) as exc:
        _skipped.append(name + " (unreadable: " + type(exc).__name__ + ")")
        return
    target = want if hard in (resource.RLIM_INFINITY, -1) else min(want, hard)
    try:
        # Soft AND hard. Lowering only the soft limit lets the snippet raise it
        # straight back -- the `redefine_its_own_limits` fixture proved exactly
        # that and printed "raised its own limit". A limit the sandboxed code can
        # undo is not a limit. Note this is still enforcement inside the process
        # the model drives, which is the weak part of the design: in production
        # the ceiling belongs to the supervisor.
        resource.setrlimit(number, (target, target))
    except (ValueError, OSError) as exc:
        _skipped.append(name + " (refused: " + type(exc).__name__ + ")")
        return
    _applied.append(name + "=" + str(target))

_limit("RLIMIT_CPU", {CPU_SECONDS})
_limit("RLIMIT_AS", {ADDRESS_SPACE_BYTES})
_limit("RLIMIT_NOFILE", 64)
_limit("RLIMIT_FSIZE", {MAX_FILE_BYTES})

sys.stderr.write("{LIMIT_MARKER}" + ";".join(_applied) + "|" + ";".join(_skipped) + "\\n")
"""

@dataclass(slots=True)
class SandboxResult:
    """What running a snippet produced, and which limit stopped it."""

    ok: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    limit_hit: str = ""
    wall_ms: float = 0.0
    scratch_wiped: bool = True
    notes: list[str] = field(default_factory=list)
    #: Limits the host actually accepted, and the ones it refused.
    limits_applied: list[str] = field(default_factory=list)
    limits_skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "limit_hit": self.limit_hit,
            "wall_ms": round(self.wall_ms, 3),
            "scratch_wiped": self.scratch_wiped,
            "notes": list(self.notes),
            "limits_applied": list(self.limits_applied),
            "limits_skipped": list(self.limits_skipped),
        }


def run_snippet(code: str, *, timeout_s: float = WALL_CLOCK_SECONDS) -> SandboxResult:
    """Execute ``code`` in a jailed subprocess.  Never raises for snippet failure."""
    scratch = Path(tempfile.mkdtemp(prefix="attacklab-sandbox-"))
    started = time.perf_counter()
    notes = [
        "subprocess, not exec: no shared interpreter state",
        "env={} passed: nothing inherited, so no key and no token is reachable",
        f"cwd is a per-task scratch directory ({scratch.name}), wiped after",
        f"wall-clock timeout {timeout_s}s on top, because blocked I/O burns no CPU",
        # Said plainly, because the alternative is a demo that claims a control
        # it does not have. On a laptop this process can open a socket; only the
        # compose service's `network_mode: none` stops it, and a run outside
        # compose has no network control at all.
        "network: NOT denied by this code -- only by the compose service "
        "(network_mode: none). Outside compose, a snippet can reach the network.",
    ]
    try:
        completed = subprocess.run(  # noqa: S603 - a jailed interpreter, by design
            [sys.executable, "-I", "-S", "-c", _PREAMBLE + code],
            cwd=scratch,
            # Empty, not sanitised. A sanitised environment is a list of the
            # variables somebody remembered.
            env={},
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        wall_ms = (time.perf_counter() - started) * 1_000
        stderr, applied, skipped = _split_limit_report(completed.stderr)
        limit = ""
        if completed.returncode != 0:
            if "MemoryError" in stderr or "Cannot allocate" in stderr:
                limit = "address space"
            elif completed.returncode in {-9, -24} or "CPU time limit" in stderr:
                limit = "cpu seconds"
            else:
                limit = "snippet error"
        return SandboxResult(
            ok=completed.returncode == 0,
            exit_code=completed.returncode,
            stdout=completed.stdout[:MAX_OUTPUT_CHARS],
            stderr=stderr[:MAX_OUTPUT_CHARS],
            limit_hit=limit,
            wall_ms=wall_ms,
            notes=notes,
            limits_applied=applied,
            limits_skipped=skipped,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(
            ok=False,
            exit_code=-1,
            stderr=f"wall-clock timeout after {timeout_s}s",
            limit_hit="wall clock",
            wall_ms=(time.perf_counter() - started) * 1_000,
            notes=notes,
        )
    finally:
        # Wiped whichever way it went. A scratch directory that survives a
        # failure is a shared volume nobody declared.
        shutil.rmtree(scratch, ignore_errors=True)


def _split_limit_report(stderr: str) -> tuple[str, list[str], list[str]]:
    """Pull the child's limit report out of its stderr."""
    kept: list[str] = []
    applied: list[str] = []
    skipped: list[str] = []
    for line in (stderr or "").splitlines():
        if line.startswith(LIMIT_MARKER):
            payload = line[len(LIMIT_MARKER) :]
            left, _, right = payload.partition("|")
            applied = [item for item in left.split(";") if item]
            skipped = [item for item in right.split(";") if item]
            continue
        kept.append(line)
    return "\n".join(kept), applied, skipped


#: The three failure shapes worth encoding, from real disclosures.
#:
#: Each is a snippet plus the lesson.  They are here as *fixtures for the
#: tests*, not as exploits: each one fails harmlessly inside the jail, and the
#: point is which limit catches it.
FAILURE_FIXTURES: dict[str, dict[str, str]] = {
    "read_a_credential": {
        "code": (
            "import os\n"
            "leaky = [k for k in os.environ if any(t in k.upper() for t in "
            "('KEY', 'TOKEN', 'SECRET', 'PASSWORD'))]\n"
            "print('env vars:', len(os.environ), '| credential-shaped:', len(leaky))\n"
        ),
        "lesson": (
            "CVE-2026-25592's lesson for this column: the boundary was fine and the "
            "tool surface was not. A sandbox is only as good as the list of host-side "
            "functions the model can reach, and enumerating that list is a different "
            "job from building the sandbox -- usually nobody's."
        ),
        # Not zero: the platform re-adds a couple of its own (on macOS,
        # __CF_USER_TEXT_ENCODING and PWD). Claiming "0 variables" was wrong, and
        # the honest claim is the one that matters: none of them is a credential.
        "expect": "credential-shaped: 0 -- nothing inherited, so there is nothing to read",
    },
    "redefine_its_own_limits": {
        "code": (
            "import resource\n"
            "try:\n"
            "    resource.setrlimit(resource.RLIMIT_CPU, (999, 999))\n"
            "    print('raised its own limit')\n"
            "except (ValueError, OSError) as exc:\n"
            "    print('refused:', type(exc).__name__)\n"
        ),
        "lesson": (
            "A 2025 CVE against a widely used coding-agent CLI showed the agent's own "
            "output influencing the limits of its sandbox. The rule that falls out: "
            "enforce the boundary in a process the model cannot influence, never "
            "inside the one it drives. Here the hard limit is already lowered, so the "
            "child cannot raise it -- but note that the enforcement being *inside* the "
            "child is the weak part of this design, and in production the limits "
            "belong to the supervisor."
        ),
        "expect": "refused: soft and hard were both lowered, so the child cannot raise them",
    },
    "allowlisted_command_carries_the_payload": {
        "code": "print('an allowlisted name is not an allowlisted capability')",
        "lesson": (
            "A 2026 CVE against a major AI editor poisoned the execution environment so "
            "that auto-approved commands delivered the attack; the allowlist actively "
            "helped. **An allowlist keyed on a command name is a judgement about a "
            "label, not about a capability** -- layer one wearing layer four's clothes. "
            "It is the clearest example in this lab of a control that looks "
            "deterministic and is not."
        ),
        "expect": "runs: the snippet is harmless, and the name told you nothing about that",
    },
}


def sandbox_limits() -> list[dict[str, str]]:
    """The five limits, for the console's sandbox column."""
    return [
        {
            "limit": "Separate kernel or microVM",
            "state": "NOT IMPLEMENTED",
            "note": (
                "A container is not a security boundary against untrusted code, and "
                "model-generated code is untrusted code. This lab uses a subprocess "
                "and says so; the production answer is a separate kernel."
            ),
        },
        {
            "limit": "Network deny-by-default, allowlisted by destination",
            "state": "compose only -- NOT enforced outside it",
            "note": (
                "By destination rather than protocol: exfiltration does not need a "
                "shell, and a DNS lookup is plenty."
            ),
        },
        {
            "limit": "Secrets brokered, short TTL, never in the environment",
            "state": "env={} passed; 0 credential-shaped variables",
            "note": (
                "If the sandbox holds a long-lived key it is not a boundary, it is a "
                "container for your credentials."
            ),
        },
        {
            "limit": "Scratch filesystem, wiped per task",
            "state": "mkdtemp per call, rmtree in finally",
            "note": "No shared volume with the app.",
        },
        {
            "limit": "Wall-clock, memory and token ceilings",
            "state": (
                f"{WALL_CLOCK_SECONDS}s wall, {CPU_SECONDS}s CPU, "
                f"{ADDRESS_SPACE_BYTES // (1024 * 1024)}MB requested"
            ),
            "note": (
                "LLM06. A denial-of-wallet control as much as a denial-of-service one. "
                "Read SandboxResult.limits_skipped on the day: the address-space limit "
                "is refused on some hosts, and a ceiling the host declined is not a "
                "ceiling however confidently the code asked for it."
            ),
        },
    ]
