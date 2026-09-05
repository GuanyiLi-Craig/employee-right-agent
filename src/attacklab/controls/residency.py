"""Residency and sovereignty.

Hook: ``resolve_model``.

* **Region travels with the model id.**  ``resolve_model("judge", "eu-west-1")``
  returns a reference whose region is intrinsic.  There is no code path in this
  lab or in the assistant that produces a model reference without one -- see
  :func:`rights_agent.hooks.default_model_ref`, which takes a region and
  defaults it rather than allowing ``None`` through.
* **Fail closed.**  An unsupported region raises
  :class:`~rights_agent.hooks.ResidencyError`, the request is refused, and the
  refusal is logged.  **Never** fall back to the nearest available region.  That
  is the beat demo 5 ends on: *a system that routes to the nearest available
  region is a system with no residency guarantee.*

  Two lines elsewhere do the enforcing, and neither is in this file.
  :func:`rights_agent.llm.resolve_model` is called by ``generate`` on every
  request, *before* the client is chosen and whether or not a caller injected
  one -- the first version checked inside ``make_client``, which the lab's own
  client selection skips entirely, so demo 5 answered from an unsupported
  region without complaint.  And the explicit ``except ResidencyError: raise``
  in ``make_client`` stops the existing "any failure means use the stub"
  fallback from turning a refusal into a quiet answer.
* **All three transfers.**  Inference, trace export and eval dataset access each
  go through :func:`check_transfer`.  Teams remember the first and forget the
  other two, because a trace does not feel like production data even though it
  contains the user's question verbatim.
* **Sovereignty is a separate field.**  Residency is where the bytes sit.
  Sovereignty is about who can compel disclosure, which is a question about the
  provider's corporate jurisdiction rather than the datacentre's postcode.  The
  console shows both in the same row, and they sometimes disagree -- which does
  more than a paragraph of explanation.
"""

from __future__ import annotations

from typing import Any

from attacklab.context import RequestContext
from attacklab.controls.base import Control, ControlResult
from rights_agent.hooks import (
    PROVIDER_JURISDICTION,
    ModelRef,
    ResidencyError,
    provider_for,
)

#: Regions this deployment is permitted to process in.
#:
#: A short list on purpose.  A permitted-region list that contains everything is
#: a configuration nobody has thought about.
PERMITTED_REGIONS: frozenset[str] = frozenset({"eu-west-1", "eu-west-2"})

#: The three transfers.  Named, because the check has to be applied to each one
#: and "we host in the EU" is a claim about only the first.
TRANSFER_INFERENCE = "inference"
TRANSFER_TRACE_EXPORT = "trace_export"
TRANSFER_EVAL_DATASET = "eval_dataset"
TRANSFERS: tuple[str, ...] = (TRANSFER_INFERENCE, TRANSFER_TRACE_EXPORT, TRANSFER_EVAL_DATASET)

TRANSFER_NOTES: dict[str, str] = {
    TRANSFER_INFERENCE: "the question and the retrieved provisions reach the model",
    TRANSFER_TRACE_EXPORT: "the span carries the question verbatim to the trace store",
    TRANSFER_EVAL_DATASET: "the golden set is real questions, held for reuse",
}

#: Where a provider processes, per region, when it does.  ``None`` means the
#: provider has no presence there -- which is a residency answer, not an error
#: to be worked around.
PROVIDER_REGIONS: dict[str, frozenset[str]] = {
    "local": frozenset({"eu-west-1", "eu-west-2", "us-east-1", "ap-southeast-2"}),
    "anthropic": frozenset({"eu-west-1", "us-east-1"}),
    "openai": frozenset({"eu-west-1", "us-east-1"}),
    "deepseek": frozenset({"cn-north-1"}),
}


def check_transfer(transfer: str, region: str) -> str:
    """Empty string when permitted; the refusal reason otherwise."""
    if transfer not in TRANSFERS:
        return f"unrecognised transfer {transfer!r}"
    if region not in PERMITTED_REGIONS:
        return (
            f"{transfer} in {region!r} is not permitted "
            f"(permitted: {', '.join(sorted(PERMITTED_REGIONS))})"
        )
    return ""


def resolve(role: str, region: str, configured: str) -> ModelRef:
    """A model reference for ``role``, or a refusal.

    Raises rather than returning a denial, and that is the one place in the lab
    where a control raises.  The reason is in :mod:`rights_agent.hooks`: this is
    a routing *precondition*, not a judgement about content.  When the region is
    unsupported there is no request to carry on with, and a "denied" return
    value is something a caller can treat as advisory.  A residency guarantee
    that a caller can treat as advisory is not one.
    """
    provider = provider_for(configured)
    reason = check_transfer(TRANSFER_INFERENCE, region)
    if reason:
        raise ResidencyError(reason)
    available = PROVIDER_REGIONS.get(provider, frozenset())
    if available and region not in available:
        raise ResidencyError(
            f"{provider} does not serve {configured!r} from {region!r} "
            f"(it serves {', '.join(sorted(available))}). Refusing rather than routing "
            "to the nearest available region: that fallback is what a residency "
            "guarantee has to exclude."
        )
    return ModelRef(
        provider=provider,
        model=configured,
        region=region,
        provider_jurisdiction=PROVIDER_JURISDICTION.get(provider, "unknown"),
    )


def transfer_table(region: str, configured: str) -> list[dict[str, Any]]:
    """All three transfers, for the console's residency panel."""
    provider = provider_for(configured)
    rows: list[dict[str, Any]] = []
    for transfer in TRANSFERS:
        reason = check_transfer(transfer, region)
        rows.append(
            {
                "transfer": transfer,
                "note": TRANSFER_NOTES[transfer],
                "region": region,
                "permitted": not reason,
                "reason": reason,
                "provider": provider,
                "provider_jurisdiction": PROVIDER_JURISDICTION.get(provider, "unknown"),
            }
        )
    return rows


class Residency(Control):
    """Cross-cutting.  **Deterministic**: a region either is on the list or is not."""

    key = "residency"
    layer = 0
    deterministic = True

    def _run(self, value: Any, ctx: RequestContext) -> ControlResult:
        """``value`` is ``(role, region, configured)``.

        Note the shape: this returns a *result carrying the reference*, and the
        stack raises.  The control itself keeps the "never raise" rule; the
        stack's ``resolve_model`` is the one place that converts a refusal into
        an exception, and it does so because the hook's contract says it may.
        """
        role, region, configured = value
        reason = check_transfer(TRANSFER_INFERENCE, region)
        if reason:
            return ControlResult(passed=False, reason=reason, matched=region)
        try:
            ref = resolve(role, region, configured)
        except ResidencyError as exc:
            return ControlResult(passed=False, reason=str(exc), matched=region)
        return ControlResult(
            passed=True,
            reason=(
                f"{ref.model} in {ref.region}; provider jurisdiction "
                f"{ref.provider_jurisdiction}"
            ),
            mutated=ref,
        )
