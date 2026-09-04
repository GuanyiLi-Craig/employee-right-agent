"""The toggle registry: the spine of the lab.

Everything else plugs in here.  Four requirements, each of which is easy to get
wrong and each of which shows on stage when you do:

* **Thread-safe.** The console and the assistant are on different threads.
* **Read once per request.** A toggle flipped mid-request must not produce a
  half-defended run: that is confusing to watch and impossible to explain.
  Enforced by :class:`~attacklab.context.RequestContext`, which snapshots this
  registry at request start and is the only thing the controls read.
* **Snapshotted onto every span and every audit record**, so scrolling back
  through Phoenix tells you which controls were on for each trace.
* **Presets**, because the presenter is talking, not aiming a mouse.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

#: Layer 0 is cross-cutting; the pre-deploy and design-time controls are not
#: per-request at all and carry sentinel layers so the console can group them.
LAYER_CROSS_CUTTING = 0
LAYER_PRE_DEPLOY = -1
LAYER_DESIGN_TIME = -2


@dataclass(frozen=True, slots=True)
class ControlSpec:
    """What a control is, for the console and the report.

    ``deterministic`` is not decoration.  Two of the five runtime controls are
    deterministic and the rest are a model or a heuristic judging text; the
    console renders the difference, and that table is the session's argument.
    """

    key: str
    label: str
    layer: int
    deterministic: bool
    default: bool = False
    notes: str = ""

    @property
    def layer_label(self) -> str:
        if self.layer == LAYER_PRE_DEPLOY:
            return "pre-deploy"
        if self.layer == LAYER_DESIGN_TIME:
            return "design-time"
        if self.layer == LAYER_CROSS_CUTTING:
            return "cross-cutting"
        return f"layer {self.layer}"

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "layer": self.layer,
            "layer_label": self.layer_label,
            "deterministic": self.deterministic,
            "default": self.default,
            "notes": self.notes,
        }


#: The seven controls, in the order the console shows them.
SPECS: tuple[ControlSpec, ...] = (
    ControlSpec(
        key="input_scan",
        label="Input scanning",
        layer=1,
        deterministic=False,
        notes=(
            "Heuristic lexicon, or a model classifier. Catches the obvious and misses "
            "the polite rephrase — which is the lesson, not a bug."
        ),
    ),
    ControlSpec(
        key="provenance",
        label="Provenance fencing",
        layer=2,
        deterministic=False,
        notes=(
            "Structural and nearly free. Not a boundary: the model still receives the "
            "text. It gives the model a signal it did not otherwise have."
        ),
    ),
    ControlSpec(
        key="output_verify",
        label="Output verification",
        layer=3,
        deterministic=False,
        notes=(
            "Easier than input scanning because the attack has narrowed: on the way "
            "out it must be a specific action against a specific target."
        ),
    ),
    ControlSpec(
        key="tool_broker",
        label="Tool scoping (policy gate)",
        layer=4,
        deterministic=True,
        notes=(
            "The only guarantee in the lab. Grants derive from the caller, parameters "
            "are authorisation, deny by default, and every denial is logged."
        ),
    ),
    ControlSpec(
        key="pii_mask",
        label="PII tokenisation",
        layer=LAYER_CROSS_CUTTING,
        deterministic=False,
        notes=(
            "Regex plus validators: a floor, not a solution. Masks inbound at capture "
            "and scans outbound, because a model can reproduce PII it inferred."
        ),
    ),
    ControlSpec(
        key="residency",
        label="Residency + sovereignty",
        layer=LAYER_CROSS_CUTTING,
        deterministic=True,
        notes=(
            "Fails closed. A system that routes to the nearest available region is a "
            "system with no residency guarantee."
        ),
    ),
    ControlSpec(
        key="supplychain",
        label="Supply-chain scanning",
        layer=LAYER_PRE_DEPLOY,
        deterministic=True,
        notes=(
            "Pre-deployment, not per-request. Model format, index content, skill "
            "manifests, digest pinning, and the tool surface itself."
        ),
    ),
    ControlSpec(
        key="rule_of_two",
        label="Rule of Two",
        layer=LAYER_DESIGN_TIME,
        deterministic=True,
        notes=(
            "Not a runtime filter — a property of the configuration. Private data, "
            "untrusted content, outward communication: hold at most two."
        ),
    ),
)

#: Keys that gate a per-request code path.  ``supplychain`` and ``rule_of_two``
#: are not among them, and the console must not imply they are: one runs before
#: deployment, the other is a reading of the configuration.
RUNTIME_KEYS: frozenset[str] = frozenset(
    {"input_scan", "provenance", "output_verify", "tool_broker", "pii_mask", "residency"}
)

#: One-click states.  Demo 4's rhetorical force depends on ``least_privilege``
#: showing layers 1-3 visibly *off* while the payload is still contained.
PRESETS: dict[str, dict[str, bool]] = {
    "all_off": {},
    "layers_1_3": {"input_scan": True, "provenance": True, "output_verify": True},
    "least_privilege": {"tool_broker": True},
    "all_on": {spec.key: True for spec in SPECS},
}

PRESET_LABELS: dict[str, str] = {
    "all_off": "ALL OFF (demo 1)",
    "layers_1_3": "LAYERS 1-3 (demo 2)",
    "least_privilege": "LEAST PRIVILEGE ONLY (demo 4)",
    "all_on": "ALL ON",
}


class Registry:
    """Which controls are on.  Mutated by the console, read by the request."""

    def __init__(self, specs: tuple[ControlSpec, ...] | list[ControlSpec] = SPECS) -> None:
        self._lock = threading.Lock()
        self._specs = {spec.key: spec for spec in specs}
        self._state = {spec.key: spec.default for spec in specs}
        self._preset = "all_off"

    # ---- reading ----------------------------------------------------------
    @property
    def specs(self) -> tuple[ControlSpec, ...]:
        return tuple(self._specs.values())

    def spec(self, key: str) -> ControlSpec:
        try:
            return self._specs[key]
        except KeyError:
            raise KeyError(f"unknown control {key!r}; known: {sorted(self._specs)}") from None

    def enabled(self, key: str) -> bool:
        with self._lock:
            return self._state[self.spec(key).key]

    def snapshot(self) -> dict[str, bool]:
        with self._lock:
            return dict(self._state)

    @property
    def preset(self) -> str:
        with self._lock:
            return self._preset

    # ---- writing ----------------------------------------------------------
    def set(self, key: str, on: bool) -> None:
        spec = self.spec(key)
        with self._lock:
            self._state[spec.key] = bool(on)
            self._preset = self._match_preset(self._state)

    def update(self, values: dict[str, bool]) -> dict[str, bool]:
        for key in values:
            self.spec(key)  # raise before changing anything
        with self._lock:
            self._state.update({key: bool(on) for key, on in values.items()})
            self._preset = self._match_preset(self._state)
            return dict(self._state)

    def apply_preset(self, name: str) -> dict[str, bool]:
        if name not in PRESETS:
            raise KeyError(f"unknown preset {name!r}; known: {sorted(PRESETS)}")
        wanted = PRESETS[name]
        with self._lock:
            self._state = {key: bool(wanted.get(key, False)) for key in self._state}
            self._preset = name
            return dict(self._state)

    def reset(self) -> dict[str, bool]:
        """Back to ``ALL OFF``, so demo 1 is one keypress away."""
        return self.apply_preset("all_off")

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _match_preset(state: dict[str, bool]) -> str:
        for name, wanted in PRESETS.items():
            if all(state.get(key, False) == wanted.get(key, False) for key in state):
                return name
        return "custom"


def enabled_keys(snapshot: dict[str, bool]) -> tuple[str, ...]:
    """Sorted enabled keys, for span attributes and audit records."""
    return tuple(sorted(key for key, on in snapshot.items() if on))


def snapshot_attribute(snapshot: dict[str, bool]) -> str:
    """The ``attacklab.controls`` span attribute: sorted, comma-joined."""
    return ",".join(enabled_keys(snapshot))
