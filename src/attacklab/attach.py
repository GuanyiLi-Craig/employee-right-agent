"""Install the lab into the assistant.

    from attacklab.attach import attach
    registry = Registry()
    attach(registry)

That is the whole integration.  The assistant is unmodified and unforked; it
calls seven hooks that do nothing until this function replaces them.
"""

from __future__ import annotations

from attacklab.controls.input_scan import HEURISTIC
from attacklab.controls.stack import ControlStack
from attacklab.registry import Registry
from rights_agent import hooks


def attach(
    registry: Registry, *, scan_mode: str = HEURISTIC, judge: bool = True
) -> ControlStack:
    """Install a control stack over ``registry``.  Returns it."""
    stack = ControlStack(registry, scan_mode=scan_mode, judge=judge)
    hooks.install(stack)
    return stack


def detach() -> None:
    """Put :class:`~rights_agent.hooks.NullHooks` back.

    Used by the tests to prove the assistant's behaviour is unchanged when the
    lab is not installed, which is the property that makes shipping the hooks
    permanently defensible.
    """
    hooks.reset()
