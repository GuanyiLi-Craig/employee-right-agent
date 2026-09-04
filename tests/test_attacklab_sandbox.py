"""The subprocess jail, and what it honestly does not do.

The most important assertions here are the negative ones.  **A container is not
a security boundary against genuinely hostile code**, this lab uses a
subprocess, and a test suite that implied otherwise would be the problem it is
supposed to demonstrate.
"""

from __future__ import annotations

import pytest

from attacklab.sandbox.runner import (
    ADDRESS_SPACE_BYTES,
    FAILURE_FIXTURES,
    LIMIT_MARKER,
    run_snippet,
    sandbox_limits,
)


def test_a_harmless_snippet_runs() -> None:
    """The first version failed *every* snippet, and that reads as a working jail.

    It set ``RLIMIT_AS`` to a flat 256MB; on a host whose hard limit is lower
    the child died in the preamble with ``ValueError: current limit exceeds
    maximum limit``, before it reached the code. ``print("hello")`` came back
    ``ok=False``. A boundary that fails closed for the wrong reason is
    indistinguishable from one that works, which is worse than one that visibly
    does not.
    """
    result = run_snippet('print("hello")')
    assert result.ok, f"exit {result.exit_code}: {result.stderr}"
    assert result.stdout.strip() == "hello"


def test_the_child_reports_which_limits_the_host_accepted() -> None:
    """A ceiling the host declined is not a ceiling, however confidently asked for."""
    result = run_snippet("pass")
    assert result.limits_applied, "no limit was applied and nothing said so"
    assert LIMIT_MARKER not in result.stderr, "the limit report leaked into stderr"


def test_the_environment_carries_no_credential() -> None:
    """Not "empty" -- the platform re-adds a couple of its own.

    The honest claim is the one that matters: nothing inherited, so no key and
    no token is reachable. Claiming zero variables was simply wrong on macOS.
    """
    result = run_snippet(FAILURE_FIXTURES["read_a_credential"]["code"])
    assert result.ok, result.stderr
    assert "credential-shaped: 0" in result.stdout


def test_a_snippet_cannot_raise_its_own_limits() -> None:
    """Soft *and* hard, or the snippet puts them back.

    The fixture printed "raised its own limit" when only the soft limit was
    lowered. Note what this still does not fix: enforcement lives inside the
    process the model drives, and in production the ceiling belongs to the
    supervisor.
    """
    result = run_snippet(FAILURE_FIXTURES["redefine_its_own_limits"]["code"])
    assert result.ok, result.stderr
    assert "refused" in result.stdout


def test_a_spinning_snippet_is_stopped() -> None:
    result = run_snippet("while True: pass", timeout_s=3.0)
    assert not result.ok
    assert result.limit_hit in {"cpu seconds", "wall clock"}


def test_a_huge_allocation_is_stopped() -> None:
    result = run_snippet("x = [0] * 10**9", timeout_s=6.0)
    assert not result.ok
    assert result.limit_hit in {"address space", "cpu seconds", "wall clock"}


def test_the_scratch_directory_is_wiped_whichever_way_it_went() -> None:
    """A scratch directory that survives a failure is an undeclared shared volume."""
    from pathlib import Path

    written = run_snippet(
        "from pathlib import Path\n"
        "Path('marker.txt').write_text('x')\n"
        "print(Path('marker.txt').resolve())\n"
    )
    assert written.ok, written.stderr
    assert not Path(written.stdout.strip()).exists()
    assert written.scratch_wiped


def test_a_snippet_error_is_a_result_not_an_exception() -> None:
    result = run_snippet("raise ValueError('deliberate')")
    assert not result.ok
    assert "ValueError" in result.stderr
    assert result.limit_hit == "snippet error"


def test_the_limits_panel_admits_what_is_not_enforced() -> None:
    """Two of the five limits are honestly incomplete, and the panel says so.

    Nobody should leave thinking a Docker container is sufficient isolation for
    untrusted code, and nobody should think this process denies the network --
    only the compose service does.
    """
    rows = {row["limit"]: row for row in sandbox_limits()}
    assert "NOT IMPLEMENTED" in rows["Separate kernel or microVM"]["state"]
    assert "NOT enforced" in rows["Network deny-by-default, allowlisted by destination"]["state"]


def test_the_network_is_not_denied_by_this_code() -> None:
    """Asserted, because the alternative is a demo claiming a control it lacks.

    On a laptop this process can open a socket. If this test ever starts
    failing because the network became unreachable, that is the compose service
    doing its job -- not this module.
    """
    result = run_snippet(
        "import socket\n"
        "socket.setdefaulttimeout(2)\n"
        "try:\n"
        "    socket.getaddrinfo('example.com', 80)\n"
        "    print('resolved')\n"
        "except Exception as exc:\n"
        "    print('blocked:', type(exc).__name__)\n"
    )
    assert result.ok, result.stderr
    assert result.stdout.strip() in {"resolved", "blocked: gaierror", "blocked: timeout"}


@pytest.mark.parametrize("key", sorted(FAILURE_FIXTURES))
def test_every_failure_fixture_carries_its_lesson(key: str) -> None:
    """A fixture without a lesson is a snippet, and a snippet teaches nothing."""
    fixture = FAILURE_FIXTURES[key]
    assert fixture["lesson"].strip()
    assert fixture["expect"].strip()
    assert fixture["code"].strip()


def test_the_requested_address_space_is_sane() -> None:
    assert 64 * 1024 * 1024 <= ADDRESS_SPACE_BYTES <= 1024 * 1024 * 1024
