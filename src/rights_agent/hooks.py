"""The seven hook points a security lab attaches to.

This module ships with the assistant and does nothing.  It exists so that the
Session 6 attack lab can wrap the request path **without forking the
assistant**: an audience watching a control fail has to be watching the same
system it trusts, or the demonstration proves nothing about that system.

Seven points, and there is no eighth:

===================  ==========================  ==================================
Hook                 Called from                 Called when
===================  ==========================  ==================================
``on_question``      :mod:`~rights_agent.graph`   before retrieval, on the raw string
``on_context``       :mod:`~rights_agent.retrieval`  on the block list, before assembly
``on_prompt``        :mod:`~rights_agent.llm`     on system+user, immediately before the call
``on_answer``        :mod:`~rights_agent.graph`   on the draft answer, before return or log
``on_tool_call``     the lab's tool dispatcher    before every tool invocation
``on_log``           :mod:`~rights_agent.audit`   on every record, before the hash chain
``resolve_model``    :mod:`~rights_agent.llm`     when choosing a model for a role
===================  ==========================  ==================================

Two properties make this safe to ship here permanently.

**With :class:`NullHooks` installed, behaviour is byte-identical.** Every no-op
returns its argument unchanged, and ``tests/test_hooks.py`` asserts the identity
for each one rather than trusting the reading.

**The hooks are synchronous and pure-ish.** They may read configuration and emit
spans; they may not mutate agent state.  A control that needs memory keeps its
own, which is why the request context in the lab belongs to the lab.

Inbound hooks **rewrite**, outbound hooks **judge**.  That split is in the
signatures and is not an accident: a control that both changes a value and
reports a verdict is a control that silently alters behaviour when it was
supposed to be observing.  ``on_question``, ``on_context``, ``on_prompt`` and
``on_log`` return a replacement value; ``on_answer`` and ``on_tool_call`` return
a verdict and change nothing.

The one exception is :func:`resolve_model`, which may raise
:class:`ResidencyError`.  Residency is a routing *precondition* rather than a
judgement about content: when the region is unsupported there is no request to
carry on with, and returning a "denied" value that some caller might treat as
advisory is how a residency guarantee turns into a nearest-region fallback.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime, types only
    from rights_agent.retrieval import Doc

#: One retrieved block, as the hooks see it.  Aliased rather than redeclared:
#: the lab operates on the assistant's own retrieval results, and a parallel
#: type would drift from them.
Block = "Doc"


class ResidencyError(RuntimeError):
    """The requested region cannot serve this role.

    Raised, not returned, and never caught by a fallback.  See the module
    docstring, and :mod:`attacklab.controls.residency` for the argument.
    """


# --------------------------------------------------------------------------- #
# Decision types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Verdict:
    """The outcome of an outbound judgement.

    ``replacement`` is what the user should be told instead when ``allowed`` is
    false.  A blocked answer still owes the user a sentence.
    """

    allowed: bool
    reason: str = ""
    control: str = ""
    layer: int = 0
    replacement: str = ""

    @classmethod
    def allow(cls, reason: str = "") -> Verdict:
        return cls(allowed=True, reason=reason)

    @classmethod
    def block(
        cls, reason: str, *, control: str = "", layer: int = 0, replacement: str = ""
    ) -> Verdict:
        return cls(
            allowed=False,
            reason=reason,
            control=control,
            layer=layer,
            replacement=replacement,
        )


#: The three outcomes a gate can return.  ``needs_approval`` is not a soft deny:
#: the call does not happen until a human says so, and the difference between
#: that and a denial is who is accountable for the next step.
ALLOW = "allow"
DENY = "deny"
NEEDS_APPROVAL = "needs_approval"


@dataclass(frozen=True, slots=True)
class Decision:
    """The outcome of a tool-call authorisation."""

    outcome: str = ALLOW
    reason: str = ""
    control: str = ""
    grant_id: str = ""
    #: Set when the denial was caused by one parameter rather than the tool.
    parameter: str = ""

    @property
    def allowed(self) -> bool:
        return self.outcome == ALLOW

    @property
    def denied(self) -> bool:
        return self.outcome == DENY

    @property
    def needs_approval(self) -> bool:
        return self.outcome == NEEDS_APPROVAL

    @classmethod
    def allow(cls, *, grant_id: str = "", reason: str = "") -> Decision:
        return cls(outcome=ALLOW, grant_id=grant_id, reason=reason)

    @classmethod
    def deny(cls, reason: str, *, control: str = "", parameter: str = "") -> Decision:
        return cls(outcome=DENY, reason=reason, control=control, parameter=parameter)

    @classmethod
    def needs_approval_for(cls, reason: str, *, control: str = "") -> Decision:
        return cls(outcome=NEEDS_APPROVAL, reason=reason, control=control)


@dataclass(frozen=True, slots=True)
class Principal:
    """Who the request is *for*.

    ``subject`` is the end user, never the service account.  A broker that mints
    grants for the service is a broker that has granted the union of everything
    anyone may do.
    """

    subject: str
    roles: frozenset[str] = frozenset()
    tenant: str = "default"
    region: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "roles": sorted(self.roles),
            "tenant": self.tenant,
            "region": self.region,
        }


@dataclass(frozen=True, slots=True)
class ModelRef:
    """A model, with its region as part of its identity.

    ``region`` is where the bytes are processed.  ``provider_jurisdiction`` is
    whose courts can compel disclosure, which is a question about the provider's
    corporate home and not about the datacentre's postcode.  The two disagree
    often enough that showing only the first is misleading.
    """

    provider: str
    model: str
    region: str
    endpoint: str = ""
    provider_jurisdiction: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "region": self.region,
            "endpoint": self.endpoint,
            "provider_jurisdiction": self.provider_jurisdiction,
        }


@dataclass(frozen=True, slots=True)
class AnswerContext:
    """Everything an outbound check needs that is not the answer itself."""

    question: str
    context: str = ""
    citations: tuple[str, ...] = ()
    #: Ids of blocks some earlier control flagged.  Citation integrity is only
    #: worth anything once this is non-empty -- see :mod:`attacklab.controls.output_verify`.
    flagged_ids: frozenset[str] = frozenset()
    request_id: str = ""
    intent: str = ""
    docs: tuple[Mapping[str, Any], ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #
@runtime_checkable
class Hooks(Protocol):
    """What the assistant calls.  Implemented by the lab's control stack."""

    def on_question(self, q: str) -> str: ...

    def on_context(self, blocks: Sequence[Doc]) -> Sequence[Doc]: ...

    def on_prompt(self, system: str, user: str) -> tuple[str, str]: ...

    def on_answer(self, answer: str, ctx: AnswerContext) -> Verdict: ...

    def on_tool_call(self, name: str, params: dict[str, Any], principal: Principal) -> Decision: ...

    def on_log(self, record: dict[str, Any]) -> dict[str, Any]: ...

    def resolve_model(
        self, role: str, region: str | None, configured: str = ""
    ) -> ModelRef: ...


#: Region used when nothing configures one.
#:
#: A default rather than an optional: :func:`default_model_ref` must not be able
#: to produce a reference with no region, because a reference with no region is
#: a routing decision nobody made.
DEFAULT_REGION = "eu-west-2"

#: Provider corporate jurisdiction, for the sovereignty column.  Not a legal
#: opinion -- a label the console can show next to the region so the difference
#: is visible.
PROVIDER_JURISDICTION: dict[str, str] = {
    "local": "n/a (in process)",
    "openai": "US",
    "anthropic": "US",
    "deepseek": "CN",
}


def provider_for(model: str) -> str:
    """Provider name implied by a model id, for the reference's identity."""
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    if model.startswith("deepseek"):
        return "deepseek"
    return "local"


def default_model_ref(
    role: str, region: str | None = None, configured: str = ""
) -> ModelRef:
    """The reference :class:`NullHooks` returns: whatever is configured.

    ``configured`` is the caller's own model id, and it wins.  The spec for this
    hook took ``(role, region)`` only, and that shape has a bug in it: the
    caller passes a :class:`~rights_agent.config.Settings` whose ``model`` may
    have been overridden per request, so a hook that re-reads the environment
    silently *replaces* that override with the process default.  Three tests
    caught it immediately.  The third argument is optional, so a two-argument
    call still works, and the environment is only consulted when nobody has an
    opinion.
    """
    model = configured or os.environ.get("RIGHTS_MODEL", "stub-local")
    if role == "judge" and not configured:
        model = os.environ.get("RIGHTS_JUDGE_MODEL") or model
    provider = provider_for(model)
    return ModelRef(
        provider=provider,
        model=model,
        region=region or os.environ.get("RIGHTS_REGION") or DEFAULT_REGION,
        provider_jurisdiction=PROVIDER_JURISDICTION.get(provider, "unknown"),
    )


class NullHooks:
    """Every hook returns its argument.  The installed default.

    Deliberately not an ABC subclass and deliberately trivial: this is the code
    path every request takes when nobody is running the lab, and it must be
    obvious by reading that it changes nothing.
    """

    def on_question(self, q: str) -> str:
        return q

    def on_context(self, blocks: Sequence[Doc]) -> Sequence[Doc]:
        return blocks

    def on_prompt(self, system: str, user: str) -> tuple[str, str]:
        return system, user

    def on_answer(self, answer: str, ctx: AnswerContext) -> Verdict:
        return Verdict.allow()

    def on_tool_call(self, name: str, params: dict[str, Any], principal: Principal) -> Decision:
        return Decision.allow()

    def on_log(self, record: dict[str, Any]) -> dict[str, Any]:
        return record

    def resolve_model(self, role: str, region: str | None, configured: str = "") -> ModelRef:
        return default_model_ref(role, region, configured)


#: The installed implementation.
#:
#: **Call sites must dereference this at call time** -- ``from rights_agent
#: import hooks`` then ``hooks.HOOKS.on_question(...)``, never ``from
#: rights_agent.hooks import HOOKS``.  The second form binds the *value* at
#: import time, so :func:`install` rebinds this global and every call site keeps
#: calling the ``NullHooks`` it captured.  That bug is completely silent: the
#: console's toggles flip, the spans record the snapshot, the tool list shrinks,
#: and not one control runs.  It cost an afternoon; ``tests/test_hooks.py``
#: asserts against it now.
HOOKS: Hooks = NullHooks()


def install(hooks: Hooks) -> Hooks:
    """Install an implementation.  Returns the one it replaced."""
    # One process-wide hook point, by design: the assistant has one request path.
    global HOOKS
    previous = HOOKS
    HOOKS = hooks
    return previous


def reset() -> None:
    """Reinstall :class:`NullHooks`.  For tests, and the console's reset."""
    install(NullHooks())


def installed() -> Hooks:
    """The current implementation.  Read through a function so tests can patch."""
    return HOOKS
