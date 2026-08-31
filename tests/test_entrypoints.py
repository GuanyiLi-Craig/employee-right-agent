"""Operator errors reach operators as one line, not sixty."""

from __future__ import annotations

import pytest

from rights_agent.config import ConfigError
from rights_agent.entrypoints import (
    INTERRUPT_EXIT,
    OPERATOR_ERRORS,
    OPERATOR_EXIT,
    operator_error_exit,
)
from rights_agent.store import IndexNotBuiltError


def test_a_successful_command_passes_its_code_through() -> None:
    @operator_error_exit
    def main(argv=None) -> int:
        return 0

    assert main([]) == 0


@pytest.mark.parametrize("error", OPERATOR_ERRORS)
def test_every_operator_error_becomes_exit_two(
    error: type[Exception], capsys: pytest.CaptureFixture[str]
) -> None:
    @operator_error_exit
    def main(argv=None) -> int:
        raise error("the fix goes here")

    assert main([]) == OPERATOR_EXIT
    captured = capsys.readouterr()
    assert captured.err.strip() == "error: the fix goes here"
    assert not captured.out, "an operator error belongs on stderr"


def test_a_bug_still_raises() -> None:
    """A traceback is the useful part of a bug; suppressing it hides the cause."""

    @operator_error_exit
    def main(argv=None) -> int:
        raise KeyError("a genuine bug")

    with pytest.raises(KeyError):
        main([])


def test_interrupt_uses_the_shell_convention(capsys: pytest.CaptureFixture[str]) -> None:
    @operator_error_exit
    def main(argv=None) -> int:
        raise KeyboardInterrupt

    assert main([]) == INTERRUPT_EXIT
    assert "interrupted" in capsys.readouterr().err


def test_the_wrapper_keeps_the_function_identity() -> None:
    @operator_error_exit
    def main(argv=None) -> int:
        """Docstring survives."""
        return 0

    assert main.__name__ == "main"
    assert main.__doc__ == "Docstring survives."


def test_config_and_index_errors_are_both_covered() -> None:
    assert ConfigError in OPERATOR_ERRORS
    assert IndexNotBuiltError in OPERATOR_ERRORS


def test_every_console_script_is_guarded() -> None:
    """The distinction is made once, not remembered eight times."""
    import importlib

    entry_points = {
        "rights_agent.cli": "ask",
        "rights_agent.pipelines.hierarchical": "main",
        "rights_agent.pipelines.simple": "main",
        "rights_agent.pipelines.compare": "main",
        "rights_agent.demo.app": "main",
        "rights_agent.tools.corpus": "main",
        "rights_agent.tools.goldens": "main",
        "rights_agent.tools.evaluate": "main",
    }
    for module_name, attribute in entry_points.items():
        function = getattr(importlib.import_module(module_name), attribute)
        assert getattr(function, "__wrapped__", None) is not None, (
            f"{module_name}:{attribute} is not wrapped with operator_error_exit"
        )
