"""The hook contract: seven points, and byte-identical behaviour when unused.

Two properties make it safe to ship :mod:`rights_agent.hooks` in the assistant
permanently, and both are asserted here rather than argued for:

* with :class:`~rights_agent.hooks.NullHooks` installed, every hook returns its
  argument **unchanged and by identity**;
* the call sites **dereference at call time**, so :func:`install` actually takes
  effect.

The second one is not hypothetical.  Every call site was written as ``from
rights_agent.hooks import HOOKS``, which binds the *value* at import time -- so
``install()`` rebound the module global and every call site kept calling the
``NullHooks`` it had captured.  The failure was completely silent: the console's
toggles flipped, the spans recorded the snapshot, the tool list shrank, and not
one control ran.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from rights_agent import hooks
from rights_agent.config import DEFAULT_REGION
from rights_agent.hooks import (
    AnswerContext,
    Decision,
    ModelRef,
    NullHooks,
    Principal,
    Verdict,
    default_model_ref,
)

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture(autouse=True)
def _reset_hooks():
    """Every test starts and ends with NullHooks installed."""
    hooks.reset()
    yield
    hooks.reset()


# --------------------------------------------------------------------------- #
# The no-op contract
# --------------------------------------------------------------------------- #
def test_null_hooks_returns_the_question_by_identity() -> None:
    question = "What does the document say about bereavement leave?"
    assert NullHooks().on_question(question) is question


def test_null_hooks_returns_the_block_list_by_identity() -> None:
    blocks: list[Any] = [object(), object()]
    assert NullHooks().on_context(blocks) is blocks


def test_null_hooks_returns_the_prompt_strings_by_identity() -> None:
    system, user = "SYSTEM", "USER"
    out_system, out_user = NullHooks().on_prompt(system, user)
    assert out_system is system
    assert out_user is user


def test_null_hooks_allows_every_answer() -> None:
    verdict = NullHooks().on_answer("anything", AnswerContext(question="q"))
    assert verdict.allowed
    assert not verdict.control


def test_null_hooks_allows_every_tool_call() -> None:
    decision = NullHooks().on_tool_call("send_notification", {"to": "anywhere"}, Principal("s"))
    assert decision.allowed


def test_null_hooks_returns_the_log_record_by_identity() -> None:
    record: dict[str, Any] = {"question": "q"}
    assert NullHooks().on_log(record) is record


def test_null_hooks_resolves_the_configured_model() -> None:
    ref = NullHooks().resolve_model("generate", "eu-west-2", "stub-local")
    assert ref.model == "stub-local"
    assert ref.region == "eu-west-2"


def test_the_configured_model_wins_over_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bug the third argument exists to fix.

    The spec gave this hook ``(role, region)`` only, so the no-op re-read
    ``RIGHTS_MODEL`` -- and silently replaced a per-request override with the
    process default. Three tests in ``test_deepseek.py`` caught it immediately,
    which is the only reason it is not still there.
    """
    monkeypatch.setenv("RIGHTS_MODEL", "gpt-4o")
    assert default_model_ref("generate", None, "claude-sonnet-5").model == "claude-sonnet-5"
    assert default_model_ref("generate", None).model == "gpt-4o"


def test_a_model_reference_always_carries_a_region(monkeypatch: pytest.MonkeyPatch) -> None:
    """There is no code path that produces a reference without one.

    Region is part of a model's identity, not a deployment note: a reference
    with no region is a routing decision nobody made.
    """
    monkeypatch.delenv("RIGHTS_REGION", raising=False)
    assert default_model_ref("generate", None).region == DEFAULT_REGION
    assert default_model_ref("generate", "").region == DEFAULT_REGION


def test_the_default_region_agrees_with_the_settings_module() -> None:
    """Two constants, one value.

    :mod:`rights_agent.hooks` cannot import :mod:`rights_agent.config` -- the
    import runs the other way -- so the value is mirrored, and a mirror needs a
    test or it becomes two values.
    """
    assert hooks.DEFAULT_REGION == DEFAULT_REGION


# --------------------------------------------------------------------------- #
# install() has to reach the call sites
# --------------------------------------------------------------------------- #
class _RecordingHooks(NullHooks):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def on_question(self, q: str) -> str:
        self.calls.append("on_question")
        return f"rewritten: {q}"

    def on_log(self, record: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("on_log")
        return record


def test_install_replaces_the_module_binding() -> None:
    recording = _RecordingHooks()
    previous = hooks.install(recording)
    assert isinstance(previous, NullHooks)
    assert hooks.HOOKS is recording
    assert hooks.installed() is recording


def test_no_call_site_binds_the_hook_value_at_import_time() -> None:
    """``from rights_agent.hooks import HOOKS`` must not appear anywhere.

    This is the silent bug, asserted structurally. The correct form is ``from
    rights_agent import hooks`` and then ``hooks.HOOKS.on_question(...)``, so
    the lookup happens per call and :func:`install` takes effect.

    Parsed rather than grepped, so the sentence in this docstring does not
    count as a violation of it.
    """
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "rights_agent.hooks":
                continue
            for alias in node.names:
                if alias.name == "HOOKS":
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not offenders, (
        f"these modules bind the hook value at import time: {offenders}. "
        "install() will not reach them. Use `from rights_agent import hooks` and "
        "`hooks.HOOKS.<hook>(...)` so the lookup happens per call."
    )


def test_every_hook_in_the_protocol_is_implemented_by_the_no_op() -> None:
    """Seven hooks, and there is no eighth.

    If a hook is added to the protocol and not to :class:`NullHooks`, the
    assistant crashes the first time nothing is installed -- which is the common
    case.
    """
    expected = {
        "on_question",
        "on_context",
        "on_prompt",
        "on_answer",
        "on_tool_call",
        "on_log",
        "resolve_model",
    }
    implemented = {
        name for name in dir(NullHooks) if not name.startswith("_") and callable(getattr(NullHooks, name))
    }
    assert implemented == expected, (
        f"NullHooks implements {sorted(implemented)}; the contract is {sorted(expected)}. "
        "If you need an eighth hook, the design is wrong."
    )


# --------------------------------------------------------------------------- #
# The decision types
# --------------------------------------------------------------------------- #
def test_a_verdict_carries_what_to_say_instead() -> None:
    """A blocked answer still owes the user a sentence."""
    verdict = Verdict.block("reason", control="output_verify", layer=3, replacement="Sorry.")
    assert not verdict.allowed
    assert verdict.replacement == "Sorry."


def test_the_three_tool_outcomes_are_distinct() -> None:
    """``needs_approval`` is not a soft deny.

    The call does not happen until a person says so, and the difference between
    that and a denial is who is accountable for the next step.
    """
    allow = Decision.allow()
    deny = Decision.deny("no grant")
    ask = Decision.needs_approval_for("irreversible")
    assert (allow.allowed, allow.denied, allow.needs_approval) == (True, False, False)
    assert (deny.allowed, deny.denied, deny.needs_approval) == (False, True, False)
    assert (ask.allowed, ask.denied, ask.needs_approval) == (False, False, True)


def test_a_model_reference_reports_region_and_jurisdiction_separately() -> None:
    """Residency is where the bytes sit; sovereignty is who can compel disclosure."""
    ref = ModelRef(
        provider="anthropic", model="claude-sonnet-5", region="eu-west-1",
        provider_jurisdiction="US",
    )
    payload = ref.to_dict()
    assert payload["region"] == "eu-west-1"
    assert payload["provider_jurisdiction"] == "US"
