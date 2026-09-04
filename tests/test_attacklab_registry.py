"""The toggle registry, and the request context that snapshots it.

Two requirements that are easy to get wrong and both show on stage:

* a toggle flipped **mid-request** must not produce a half-defended run;
* ``controls={}`` must mean *nothing on*, not *whatever the registry says*.

The second one was a real bug: the override merged onto the snapshot, and an
empty dict is falsy, so every "run this bare" probe silently inherited the
console's toggles. The tally then reported a 0% block rate while the panel
showed two of three payloads contained.
"""

from __future__ import annotations

import threading

import pytest

from attacklab.context import request_context
from attacklab.registry import (
    PRESETS,
    RUNTIME_KEYS,
    SPECS,
    Registry,
    enabled_keys,
    snapshot_attribute,
)


def test_everything_starts_off() -> None:
    """``ALL OFF`` is the default, so demo 1 needs no setup."""
    registry = Registry()
    assert not any(registry.snapshot().values())
    assert registry.preset == "all_off"


def test_an_unknown_control_raises_before_anything_changes() -> None:
    registry = Registry()
    with pytest.raises(KeyError, match="unknown control"):
        registry.set("nonexistent", True)
    with pytest.raises(KeyError, match="unknown control"):
        registry.update({"input_scan": True, "nonexistent": True})
    assert not registry.enabled("input_scan"), (
        "a rejected batch update changed state anyway, so a typo half-applies"
    )


def test_the_presets_are_complete_states_not_patches() -> None:
    """``LEAST PRIVILEGE ONLY`` must turn layers 1-3 *off*, not leave them.

    Demo 4's rhetorical force depends on the audience seeing layers 1-3 off
    while the payload is still contained.
    """
    registry = Registry()
    registry.apply_preset("all_on")
    state = registry.apply_preset("least_privilege")
    assert state["tool_broker"] is True
    assert not any(state[key] for key in ("input_scan", "provenance", "output_verify"))


def test_a_preset_is_recognised_after_toggling_back_to_it() -> None:
    registry = Registry()
    registry.apply_preset("layers_1_3")
    registry.set("input_scan", False)
    assert registry.preset == "custom"
    registry.set("input_scan", True)
    assert registry.preset == "layers_1_3"


def test_reset_returns_to_all_off() -> None:
    registry = Registry()
    registry.apply_preset("all_on")
    assert not any(registry.reset().values())


def test_the_snapshot_attribute_is_sorted_and_joined() -> None:
    """It lands on every span, so it has to be stable across runs."""
    registry = Registry()
    registry.update({"tool_broker": True, "input_scan": True})
    assert snapshot_attribute(registry.snapshot()) == "input_scan,tool_broker"
    assert enabled_keys(registry.snapshot()) == ("input_scan", "tool_broker")


def test_a_toggle_flipped_mid_request_does_not_change_that_request() -> None:
    """The reason :class:`~attacklab.context.RequestContext` exists.

    A half-defended run is confusing to watch and impossible to explain, so the
    registry is read once, at request start, and the controls read the context.
    """
    registry = Registry()
    with request_context(registry) as ctx:
        assert not ctx.get("input_scan")
        registry.set("input_scan", True)
        assert not ctx.get("input_scan"), "the request saw a toggle flipped after it started"
    with request_context(registry) as later:
        assert later.get("input_scan"), "the next request did not pick the change up"


def test_an_empty_override_means_nothing_on() -> None:
    """``controls={}`` is how the harness says "run this bare"."""
    registry = Registry()
    registry.apply_preset("all_on")
    with request_context(registry, overrides={}) as ctx:
        assert not any(ctx.controls.values()), (
            "an empty override inherited the registry, so every bare probe came back "
            "already-defended"
        )


def test_a_partial_override_is_the_complete_control_set() -> None:
    """"Only tool_broker" has to mean only tool_broker."""
    registry = Registry()
    registry.apply_preset("all_on")
    with request_context(registry, overrides={"tool_broker": True}) as ctx:
        assert ctx.controls["tool_broker"] is True
        assert not any(v for k, v in ctx.controls.items() if k != "tool_broker")


def test_no_override_uses_the_registry() -> None:
    registry = Registry()
    registry.apply_preset("layers_1_3")
    with request_context(registry) as ctx:
        assert ctx.get("input_scan") and ctx.get("provenance") and ctx.get("output_verify")


def test_the_registry_is_thread_safe() -> None:
    """The console and the assistant are on different threads."""
    registry = Registry()
    errors: list[BaseException] = []

    def hammer() -> None:
        try:
            for _ in range(200):
                registry.apply_preset("all_on")
                registry.snapshot()
                registry.reset()
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    assert set(registry.snapshot()) == {spec.key for spec in SPECS}


def test_each_request_gets_its_own_fence_nonce() -> None:
    """A fixed delimiter can be forged by an attacker who has read your source.

    And this attacker has: the payload arrived in a document, and whoever wrote
    it could read the repository.
    """
    registry = Registry()
    with request_context(registry) as first, request_context(registry) as second:
        assert first.nonce != second.nonce
        assert len(first.nonce) >= 16


def test_only_two_runtime_controls_are_deterministic() -> None:
    """The session's argument, as a table.

    **Two of the six per-request controls**, and the count is worth pinning
    because the presenter says it aloud on slide 7. Counting the whole spec list
    instead gives four of eight, which is a different and much weaker claim --
    ``supplychain`` runs before deployment and ``rule_of_two`` is a reading of
    the configuration, so neither is a control an attacker meets.
    """
    runtime = [spec for spec in SPECS if spec.key in RUNTIME_KEYS]
    assert len(runtime) == 6, f"there are now {len(runtime)} runtime controls, not six"
    deterministic = sorted(spec.key for spec in runtime if spec.deterministic)
    assert deterministic == ["residency", "tool_broker"], (
        f"deterministic runtime controls are {deterministic}; the deck says two of six"
    )


def test_the_pre_deploy_and_design_time_controls_are_not_runtime() -> None:
    """The console must not imply that these gate a request.

    One runs before deployment; the other is a reading of the configuration.
    """
    assert "supplychain" not in RUNTIME_KEYS
    assert "rule_of_two" not in RUNTIME_KEYS


def test_every_preset_names_only_real_controls() -> None:
    known = {spec.key for spec in SPECS}
    for name, wanted in PRESETS.items():
        unknown = sorted(set(wanted) - known)
        assert not unknown, f"preset {name!r} names controls that do not exist: {unknown}"
