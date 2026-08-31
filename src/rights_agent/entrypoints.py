"""One place that decides how an operator error reaches an operator.

There are two kinds of failure in a command-line tool. A **bug** should print a
traceback, because the traceback is the useful part. An **operator error** --
retention below the statutory floor, no index built yet, an embedder that does
not match the one the index was built with -- has a known fix, and printing
sixty lines of stack in front of it buries the one sentence that helps.

Every console script wraps its entry function with :func:`operator_error_exit`,
so the distinction is made once rather than remembered eight times.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable, Sequence
from typing import TypeVar

from rights_agent.audit import AuditError
from rights_agent.config import ConfigError
from rights_agent.document.parser import ParserError
from rights_agent.embedding import EmbedderError
from rights_agent.store import IndexNotBuiltError, StoreError

#: Failures whose message *is* the fix.
OPERATOR_ERRORS: tuple[type[Exception], ...] = (
    ConfigError,
    IndexNotBuiltError,
    StoreError,
    EmbedderError,
    ParserError,
    AuditError,
)

#: Exit code for an operator error. Distinct from 1 so a script can tell "you
#: configured this wrongly" from "it crashed".
OPERATOR_EXIT = 2

#: 128 + SIGINT, the shell convention.
INTERRUPT_EXIT = 130

_F = TypeVar("_F", bound=Callable[..., int])


def operator_error_exit(main: _F) -> _F:
    """Print operator errors as one line and exit 2; let bugs raise."""

    @functools.wraps(main)
    def wrapper(argv: Sequence[str] | None = None) -> int:
        try:
            return main(argv)
        except OPERATOR_ERRORS as exc:
            print(f"error: {exc}", file=sys.stderr)
            return OPERATOR_EXIT
        except KeyboardInterrupt:
            print("interrupted", file=sys.stderr)
            return INTERRUPT_EXIT

    return wrapper  # type: ignore[return-value]
