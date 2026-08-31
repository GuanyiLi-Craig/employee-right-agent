"""The hash-chained audit record, and what it does and does not prove."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import pytest

from rights_agent.audit import (
    GENESIS_HASH,
    RETENTION_FLOOR_DAYS,
    AuditError,
    AuditLog,
    AuditRecord,
    RetentionPolicy,
    expired,
    fingerprint,
    redact,
    verify_records,
)


@pytest.fixture
def log(tmp_path: Path) -> AuditLog:
    audit = AuditLog(tmp_path / "audit.jsonl")
    for index in range(4):
        audit.append(
            ts=f"2026-08-28T10:0{index}:00Z",
            request_id=f"r{index}",
            session_id="s1",
            actor="alex",
            question=f"question {index}",
            question_sha256=fingerprint(f"question {index}"),
            answered=True,
            index_version="parser-3+hashing-bow-512+abcd1234",
            sources=[{"id": "l1", "citation": "s.19(1)", "version": "v1", "sha256": "ab" * 32}],
        )
    return audit


# --------------------------------------------------------------------------- #
# The chain
# --------------------------------------------------------------------------- #
def test_the_first_record_links_to_genesis(log: AuditLog) -> None:
    rows = log.read()
    assert rows[0]["sequence"] == 0
    assert rows[0]["previous_hash"] == GENESIS_HASH


def test_each_record_links_to_the_one_before(log: AuditLog) -> None:
    rows = log.read()
    for earlier, later in pairwise(rows):
        assert later["previous_hash"] == earlier["record_hash"]
        assert later["sequence"] == earlier["sequence"] + 1


def test_a_clean_chain_verifies_from_genesis(log: AuditLog) -> None:
    result = log.verify()
    assert result.ok
    assert result.records == result.verified == 4
    assert result.broken_at is None
    assert "CHAIN INTACT" in result.render()


def test_editing_the_first_record_breaks_the_whole_chain(log: AuditLog) -> None:
    """A decision from an hour ago, and everything after it becomes unverifiable."""
    log.tamper(0, "question")
    result = log.verify()
    assert not result.ok
    assert result.broken_at == 0
    assert result.verified == 0
    rendered = result.render()
    assert "CHAIN BROKEN at record 0" in rendered
    assert "every record after 0 is now unverifiable" in rendered.lower()


def test_editing_a_later_record_is_located_precisely(log: AuditLog) -> None:
    log.tamper(2, "answered", False)
    result = log.verify()
    assert result.broken_at == 2
    assert result.verified == 2, "records before the edit still verify"


def test_removing_a_record_is_detected(log: AuditLog) -> None:
    rows = log.read()
    del rows[1]
    log.path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    result = log.verify()
    assert not result.ok
    assert "inserted or removed" in result.reason


def test_reordering_records_is_detected(log: AuditLog) -> None:
    rows = log.read()
    rows[1], rows[2] = rows[2], rows[1]
    log.path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert not log.verify().ok


def test_a_rewritten_store_verifies_and_that_is_the_honest_limit(tmp_path: Path) -> None:
    """Tamper-EVIDENT, not tamper-proof.

    Someone who can rewrite the whole store recomputes every hash and produces a
    chain that verifies. This test exists so the limitation is asserted rather
    than only claimed in a docstring -- it is the reason the mitigation is to
    anchor outside the store.
    """
    log = AuditLog(tmp_path / "audit.jsonl")
    for index in range(3):
        log.append(request_id=f"r{index}", question=f"q{index}")

    rows = log.read()
    rows[0]["question"] = "a question that was never asked"
    previous = GENESIS_HASH
    for row in rows:
        row["previous_hash"] = previous
        record = AuditRecord(**{k: v for k, v in row.items() if k in AuditRecord.__dataclass_fields__})
        sealed = record.sealed()
        row["record_hash"] = sealed.record_hash
        previous = sealed.record_hash
    log.path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    assert log.verify().ok, "a full-store rewrite is undetectable from inside the store"


def test_empty_log_is_reported_as_empty(tmp_path: Path) -> None:
    result = AuditLog(tmp_path / "missing.jsonl").verify()
    assert result.records == 0
    assert "empty" in result.render()


def test_a_corrupt_line_is_an_error_not_a_skipped_row(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append(request_id="r0")
    with path.open("a") as handle:
        handle.write("{not json\n")
    with pytest.raises(AuditError, match="investigated, not skipped"):
        log.read()


def test_every_verification_states_the_limitation(log: AuditLog) -> None:
    for rendered in (log.verify().render(), (log.tamper(0), log.verify().render())[1]):
        assert "tamper-evident, not tamper-proof" in rendered.lower()
        assert "anchor outside the store" in rendered.lower()


# --------------------------------------------------------------------------- #
# Content
# --------------------------------------------------------------------------- #
def test_the_record_answers_who_what_and_how(log: AuditLog) -> None:
    row = log.read()[0]
    for who in ("actor", "tenant", "role", "lawful_basis"):
        assert who in row
    for what in ("question", "answered", "refused", "citations"):
        assert what in row
    for how in ("index_version", "prompt_version", "model", "sufficiency", "sources"):
        assert how in row


def test_sources_are_ids_versions_and_hashes_not_text(log: AuditLog) -> None:
    """An id plus a hash proves a source changed; the version is what lets you
    reconstruct what it said."""
    source = log.read()[0]["sources"][0]
    assert set(source) == {"id", "citation", "version", "sha256"}
    assert "text" not in source and "content" not in source
    assert len(source["sha256"]) == 64


def test_the_hash_covers_the_content_but_not_itself() -> None:
    record = AuditRecord(sequence=0, previous_hash=GENESIS_HASH, question="q").sealed()
    assert "record_hash" not in record.payload()
    assert record.record_hash == record.compute_hash()
    assert record.compute_hash() != AuditRecord(
        sequence=0, previous_hash=GENESIS_HASH, question="q2"
    ).compute_hash()


def test_the_hash_is_stable_across_key_order() -> None:
    """Canonical JSON: the hash depends on content, not on serialisation order."""
    left = AuditRecord(sequence=0, previous_hash=GENESIS_HASH, question="q", actor="a")
    right = AuditRecord(previous_hash=GENESIS_HASH, sequence=0, actor="a", question="q")
    assert left.compute_hash() == right.compute_hash()


# --------------------------------------------------------------------------- #
# Redaction at capture
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("contact jo.smith+x@acme.co.uk", "contact [email]"),
        ("my number is AB123456C", "my number is [nino]"),
        ("call 07700 900123", "call [phone]"),
        ("payroll 000123456789", "payroll [digits]"),
    ],
)
def test_direct_identifiers_are_masked(text: str, expected: str) -> None:
    assert redact(text) == expected


def test_redaction_leaves_legal_references_alone() -> None:
    """Over-redacting a corpus of numbered provisions would destroy the record."""
    text = "What does section 18 say about the 56 day window in Part 4?"
    assert redact(text) == text


def test_identity_survives_redaction() -> None:
    """The fingerprint is of the *original*, so a question can be proved without
    being stored."""
    question = "Does my contract with a@b.com cover this?"
    assert fingerprint(question) != fingerprint(redact(question))
    assert len(fingerprint(question)) == 64


def test_redaction_happens_before_the_hash_is_taken(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append(question=redact("email a@b.com"), question_sha256=fingerprint("email a@b.com"))
    row = log.read()[0]
    assert "a@b.com" not in json.dumps(row)
    assert row["question_sha256"] == fingerprint("email a@b.com")


# --------------------------------------------------------------------------- #
# Retention and anchoring
# --------------------------------------------------------------------------- #
def test_the_policy_records_both_boundaries() -> None:
    policy = RetentionPolicy()
    assert policy.floor_days == RETENTION_FLOOR_DAYS
    assert "Articles 19 and 26(6)" in policy.floor_basis
    assert "storage limitation" in policy.ceiling_basis
    assert policy.purpose


def test_a_period_below_the_floor_is_rejected_with_its_basis() -> None:
    with pytest.raises(ValueError, match="Articles 19 and 26"):
        RetentionPolicy(configured_days=30).validate()


def test_a_longer_period_is_allowed_because_the_floor_is_a_floor() -> None:
    RetentionPolicy(configured_days=2_555).validate()


def test_expired_records_are_reported_not_deleted(log: AuditLog) -> None:
    """Deleting audit records is a decision with a legal basis attached."""
    policy = RetentionPolicy(configured_days=RETENTION_FLOOR_DAYS)
    stale = expired(log.read(), policy, now_iso="2028-01-01T00:00:00Z")
    assert stale == [0, 1, 2, 3]
    assert expired(log.read(), policy, now_iso="2026-08-28T12:00:00Z") == []
    assert log.verify().records == 4, "nothing was removed"


def test_a_checkpoint_carries_the_head_hash_and_says_where_it_belongs(
    log: AuditLog, tmp_path: Path
) -> None:
    checkpoint = log.write_checkpoint(tmp_path / "checkpoint.json")
    assert checkpoint["records"] == 4
    assert checkpoint["head_hash"] == log.verify().head_hash
    assert "separately controlled" in checkpoint["note"]


def test_verify_records_accepts_plain_dicts(log: AuditLog) -> None:
    assert verify_records(log.read()).ok
    assert verify_records([]).records == 0


def test_tampering_a_missing_record_is_refused(log: AuditLog) -> None:
    with pytest.raises(AuditError, match="does not exist"):
        log.tamper(99)
    with pytest.raises(AuditError, match="no field"):
        log.tamper(0, "not_a_field")


def test_clear_restarts_the_chain_from_genesis(log: AuditLog) -> None:
    log.clear()
    assert log.verify().records == 0
    record = log.append(request_id="fresh")
    assert record.sequence == 0 and record.previous_hash == GENESIS_HASH
