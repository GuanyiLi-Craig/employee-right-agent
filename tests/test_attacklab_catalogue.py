"""The catalogue's own invariants, and the audit-record schema change.

No index needed: these are properties of the data and of the record shape.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from attacklab.attacks.catalogue import (
    BY_ID,
    CATALOGUE,
    OWASP_TO_ASI,
    RUNNABLE,
    SESSION_RISKS,
    Goal,
    Vector,
    indirect_payloads,
)
from attacklab.attacks.corpus import splice, splice_all
from attacklab.console import incident
from attacklab.registry import SPECS
from rights_agent.audit import AUDIT_SCHEMA, AuditLog, AuditRecord, verify_records


# --------------------------------------------------------------------------- #
# The catalogue
# --------------------------------------------------------------------------- #
def test_the_catalogue_is_the_advertised_size() -> None:
    """Roughly twenty, because each one has to be individually explicable.

    A hundred payloads is a number, not a catalogue.
    """
    assert 18 <= len(CATALOGUE) <= 26
    assert len(RUNNABLE) == len(CATALOGUE) - 2, "the two cross-modal gaps are not runnable"


def test_every_payload_file_exists() -> None:
    """Payloads live in files: greppable, diffable, reviewable."""
    for payload in CATALOGUE:
        assert payload.path.is_file(), f"{payload.id}: {payload.path} is missing"
        assert payload.text.strip(), f"{payload.id} is empty"


def test_every_payload_carries_both_identifiers() -> None:
    """The 2026 LLM list owns the model as a component; ASI owns it as an actor.

    Labelling both is how students learn where the line is.
    """
    for payload in CATALOGUE:
        assert payload.owasp in OWASP_TO_ASI, f"{payload.id}: unknown OWASP id {payload.owasp!r}"
        assert payload.asi, f"{payload.id} has no ASI identifier"


def test_the_five_session_risks_are_the_attack_chain_in_order() -> None:
    """Slide 4's selection rule, as data."""
    assert [owasp for owasp, _label, _beat in SESSION_RISKS] == [
        "LLM01:2026",
        "LLM02:2026",
        "LLM03:2026",
        "LLM04:2026",
        "LLM10:2026",
    ]
    beats = [beat for _o, _l, beat in SESSION_RISKS]
    assert beats == [
        "how it gets in",
        "what leaks",
        "what it can do",
        "what you downloaded",
        "what finally runs",
    ]


def test_unbounded_consumption_is_labelled_as_session_seven() -> None:
    """Slide 4 hands LLM06 to session 7; the payloads are here as the foreshadow."""
    assert "LLM06:2026" not in {owasp for owasp, _l, _b in SESSION_RISKS}
    assert "LLM06:2026" in OWASP_TO_ASI
    consume = [p for p in RUNNABLE if p.goal is Goal.CONSUME]
    assert consume and all(p.owasp == "LLM06:2026" for p in consume)


def test_every_expectation_names_a_real_control() -> None:
    known = {spec.key for spec in SPECS}
    for payload in CATALOGUE:
        unknown = sorted((payload.expect_blocked_by | payload.expect_evades) - known)
        assert not unknown, f"{payload.id} names controls that do not exist: {unknown}"


def test_no_control_is_claimed_to_both_stop_and_evade_a_payload() -> None:
    for payload in CATALOGUE:
        both = sorted(payload.expect_blocked_by & payload.expect_evades)
        assert not both, f"{payload.id} claims {both} both stops and evades it"


def test_residency_appears_in_no_expectation_set() -> None:
    """It is a routing precondition and has nothing to say about payload content.

    Listing it as evaded by all twenty would be noise diluting the two fields
    that do the pedagogical work.
    """
    for payload in CATALOGUE:
        assert "residency" not in payload.expect_blocked_by | payload.expect_evades


def test_the_rephrases_are_declared_to_evade_the_input_scanner() -> None:
    """Demo 2's beat, declared in the data as well as asserted in the gate."""
    for payload_id in ("p07", "p08", "p09", "p10"):
        payload = BY_ID[payload_id]
        assert "input_scan" in payload.expect_evades, (
            f"{payload_id} no longer claims to evade input_scan, so demo 2 has no beat"
        )
        assert "output_verify" in payload.expect_blocked_by


def test_a_payload_with_no_expectations_says_what_contains_it() -> None:
    """An empty ``expect_blocked_by`` must never read as an oversight."""
    for payload in RUNNABLE:
        if payload.expect_blocked_by or payload.expect_evades:
            continue
        assert payload.expect_bare_containment, (
            f"{payload.id} claims nothing stops it and nothing contains it either"
        )
        assert payload.expect_bare_containment in {"audit_hook", "budget", "no_directive"}


def test_every_runnable_payload_has_a_question() -> None:
    """A direct injection rides on something a person genuinely wanted to know."""
    for payload in RUNNABLE:
        assert payload.question, f"{payload.id} has no question"


def test_every_payload_carries_an_explicable_note() -> None:
    """Twenty payloads, each individually explicable. A note is how you tell."""
    for payload in CATALOGUE:
        assert len(payload.note) > 40, f"{payload.id}'s note is too thin to say on stage"


def test_the_documented_gaps_are_documented_rather_than_faked() -> None:
    """The assistant has no image or audio channel, and faking one would claim coverage."""
    gaps = [p for p in CATALOGUE if p.documented_gap]
    assert len(gaps) == 2
    for payload in gaps:
        assert "NOT EXERCISED" in payload.text
        assert not payload.expect_blocked_by and not payload.expect_evades


# --------------------------------------------------------------------------- #
# The splice
# --------------------------------------------------------------------------- #
def test_no_two_indirect_payloads_share_a_host_section() -> None:
    """If they did, demo 1's question would retrieve two poisons.

    The input scanner would catch whichever was crude, and demo 1 would report
    a containment nobody arranged.
    """
    hosts = [p.host_section for p in indirect_payloads()]
    assert len(hosts) == len(set(hosts)), f"duplicated hosts: {hosts}"


def test_splice_all_refuses_a_duplicated_host() -> None:
    """Asserted rather than trusted, because the failure would be silent.

    Two payloads on one provision means demo 1's question retrieves both, the
    scanner catches whichever is crude, and the panel reports a containment
    nobody arranged. That is a wrong number rather than an error, which is the
    kind that survives a rehearsal.
    """
    first, second = indirect_payloads()[0], indirect_payloads()[1]
    clashing = replace(second, host_section=first.host_section)
    with pytest.raises(RuntimeError, match="claimed by more than one payload"):
        splice_all("corpus text", [first, clashing])


def test_a_missing_host_section_is_a_clear_error() -> None:
    direct = next(p for p in RUNNABLE if p.vector is Vector.DIRECT)
    with pytest.raises(RuntimeError, match="no host_section"):
        splice("some corpus text", direct)


def test_a_host_section_absent_from_the_corpus_is_a_clear_error() -> None:
    """Loosening the pattern would splice the payload somewhere arbitrary.

    And a payload spliced somewhere arbitrary is not retrieved by its question,
    so the demo shows nothing -- quietly.
    """
    with pytest.raises(RuntimeError, match="could not find"):
        splice_all("a corpus with none of the expected headings", indirect_payloads())


def test_the_committed_dataset_matches_the_catalogue() -> None:
    """``evals/adversarial.jsonl`` is derived, so it can go stale.

    It is committed for a reviewer's benefit -- the whole containment matrix in
    one readable place, and a diff that makes a change in expectations visible
    in a pull request. Committing a generated file means it can drift, so the
    freshness is asserted rather than hoped for.

    Note the gate asserts against the *catalogue*, not against this file: a gate
    reading a generated artefact would be asserting that the generator ran.
    """
    from attacklab.attacks.dataset import main as dataset_main

    assert dataset_main(["--check"]) == 0, (
        "evals/adversarial.jsonl is out of date. Regenerate it with `make dataset`, "
        "in the same commit as the catalogue change."
    )


# --------------------------------------------------------------------------- #
# The audit schema change
# --------------------------------------------------------------------------- #
def test_the_schema_was_bumped_for_the_controls_field() -> None:
    assert AUDIT_SCHEMA == 2
    assert "controls" in AuditRecord.__dataclass_fields__


def test_an_empty_controls_mapping_is_omitted_from_the_hash() -> None:
    """Canonicalisation, not an exemption.

    Absent and empty mean the same thing, so they must hash the same -- and that
    is what lets a record written under schema 1 still verify.
    """
    record = AuditRecord(sequence=0, previous_hash="0" * 64, question="q")
    assert "controls" not in record.payload()
    with_controls = AuditRecord(
        sequence=0, previous_hash="0" * 64, question="q", controls={"enabled": []}
    )
    assert "controls" in with_controls.payload()


def test_a_schema_one_row_still_verifies(tmp_path: Path) -> None:
    """The compatibility claim, against a row that predates the field.

    Written by hand rather than fetched, so the test does not depend on an
    artefact somebody might clean up.
    """
    legacy = AuditRecord(sequence=0, previous_hash="0" * 64, schema=1, question="q").sealed()
    row = json.loads(legacy.to_json())
    row.pop("controls", None)
    verification = verify_records([row])
    assert verification.ok, verification.reason


def test_editing_a_populated_controls_mapping_breaks_the_hash(tmp_path: Path) -> None:
    """The property that matters: a populated mapping is covered."""
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append(question="q", controls={"enabled": ["tool_broker"], "blocked_by": []})
    rows = log.read()
    assert verify_records(rows).ok
    rows[0]["controls"]["blocked_by"] = ["input_scan"]
    assert not verify_records(rows).ok


def test_a_hook_may_not_extend_the_audit_schema(tmp_path: Path) -> None:
    """Named explicitly, because the fix is in the hook and not in the record."""
    from rights_agent import hooks
    from rights_agent.audit import AuditError
    from rights_agent.hooks import NullHooks

    class _Inventive(NullHooks):
        def on_log(self, record):  # type: ignore[no-untyped-def]
            return {**record, "invented_field": 1}

    log = AuditLog(tmp_path / "audit.jsonl")
    hooks.install(_Inventive())
    try:
        with pytest.raises(AuditError, match="no place for"):
            log.append(question="q")
    finally:
        hooks.reset()


# --------------------------------------------------------------------------- #
# Slide 11
# --------------------------------------------------------------------------- #
def test_the_incident_columns_are_three_and_three() -> None:
    """Slide 11 has exactly three each, and the split is the argument.

    Kept as data so nobody can quietly add a fourth item to the flattering
    column.
    """
    assert len(incident.FAILED) == 3
    assert len(incident.HELD) == 3


def test_every_failure_names_the_same_pattern() -> None:
    assert {row["pattern"] for row in incident.FAILED} == {"one credential had too much power"}


def test_every_control_that_held_was_a_scope_decision() -> None:
    """None of them needed to understand the attacker's prompt."""
    assert {row["pattern"] for row in incident.HELD} == {"a scope decision made in advance"}


def test_the_counterfactual_is_stated_rather_than_left_to_the_audience() -> None:
    """Say it before someone who has read the report says it for you."""
    text = incident.COUNTERFACTUAL
    assert "none were running" in text
    assert "orders of magnitude" in text
    assert "hindsight" in text
