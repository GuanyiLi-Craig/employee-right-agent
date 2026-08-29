"""A hash-chained audit record: one row per decision, verifiable from genesis.

Each record carries the hash of the record before it, so an edit cannot be made
*silently* between the decision and the audit.  This is **tamper-evident, not
tamper-proof**: it catches a local edit, and anyone who can rewrite the whole
store can recompute every subsequent hash and produce a chain that verifies.
The mitigation is to anchor outside the store -- see :func:`write_checkpoint`.

**This field list is an engineering design, not a legal schema.**  The EU AI
Act's Article 12 is a *logging-capability* duty: it requires high-risk systems
to technically allow automatic recording of events over the system's lifetime.
It does not prescribe these fields, and it is not the source of a retention
number.  Retention has two boundaries and neither comes from Article 12:

* **AI Act Articles 19 and 26(6)** -- providers and deployers keep the logs the
  system automatically generates, to the extent they control them, for a period
  appropriate to the intended purpose and *generally at least six months*,
  unless other Union or national law requires otherwise.  Six months is a
  **floor**, not a fixed period; sector law routinely requires longer.
* **GDPR** -- a storage-limitation *principle*, not a universal maximum: keep
  personal data no longer than necessary and be able to justify the period.

Scope, since it is easy to overstate: certain employment uses are *listed* in
Annex III (recruitment and candidate selection, decisions affecting the
employment relationship, task allocation, worker monitoring and evaluation).
Where such a system is classified high-risk the obligations apply; Article 6(3)
allows a provider to document an assessment that a listed system does not pose a
significant risk.  "All employment AI is high-risk" is wrong.

Two design choices worth reading the code for:

* **Sources are recorded as id + version + hash, never as a copy of the text.**
  An id and a hash prove the source *changed*; they do not let you reconstruct
  what it said.  The version does.  If your corpus is not versioned, hashing
  gives you detection without recovery -- worth knowing before an audit rather
  than during one.
* **Redaction happens at capture, not on read.**  A redaction applied when the
  UI renders has already been exported by anyone with an API key.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from rights_agent.log import get_logger

log = get_logger("audit")

#: ``previous_hash`` of the first record.  A fixed, recognisable genesis value
#: means "this is the start of the chain" cannot be confused with "the previous
#: hash is missing".
GENESIS_HASH = "0" * 64

AUDIT_SCHEMA = 1

#: The AI Act Articles 19/26(6) floor, in days.  A floor, not a policy: the
#: deployed period must be justified against the purpose and against sector law.
RETENTION_FLOOR_DAYS = 183


class AuditError(RuntimeError):
    """Raised when the audit store cannot be read or appended to."""


# --------------------------------------------------------------------------- #
# Redaction at capture
# --------------------------------------------------------------------------- #
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
    # UK National Insurance number.
    (re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b"), "[nino]"),
    (re.compile(r"\b(?:\+?44|0)(?:\s?\d){9,10}\b"), "[phone]"),
    # Any other long digit run: account numbers, case references, payroll ids.
    (re.compile(r"\b\d{9,}\b"), "[digits]"),
)


def redact(text: str) -> str:
    """Mask obvious direct identifiers.

    Deliberately conservative and deliberately not a PII detector: it removes
    the identifiers that show up in employment questions, and the record keeps a
    hash of the *original* so a question can still be proved without storing it.
    """
    out = text
    for pattern, replacement in _REDACTIONS:
        out = pattern.sub(replacement, out)
    return out


def fingerprint(text: str) -> str:
    """SHA-256 of the unredacted text, so identity survives redaction."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class AuditRecord:
    """One decision, in the shape an engineer would actually build.

    Grouped the way the question is usually asked: **who** asked, **what** the
    system did, **how** the answer was produced.  "How" is the mechanism --
    which index, which documents at which version, which model, which prompt
    version, which thresholds.  It is not *why the model reasoned as it did*,
    and no amount of logging provides that.
    """

    # -- chain ---------------------------------------------------------------
    sequence: int
    previous_hash: str
    record_hash: str = ""
    schema: int = AUDIT_SCHEMA

    # -- when ----------------------------------------------------------------
    ts: str = ""
    request_id: str = ""
    session_id: str = ""

    # -- who -----------------------------------------------------------------
    actor: str = "anonymous"
    tenant: str = "default"
    role: str = "reader"
    lawful_basis: str = "legitimate_interests"

    # -- what ----------------------------------------------------------------
    question: str = ""          # redacted at capture
    question_sha256: str = ""   # fingerprint of the original
    answered: bool = False
    refused: bool = False
    refusal_reason: str = ""
    citations: list[str] = field(default_factory=list)

    # -- how -----------------------------------------------------------------
    index_version: str = ""
    embedding_model: str = ""
    parser_version: str = ""
    prompt_version: str = ""
    model: str = ""
    #: What was asked for, and whether something else answered. A record that
    #: names only the configured model cannot answer "which model produced this"
    #: after a silent failover -- which is the question it exists for.
    requested_model: str = ""
    fallback: bool = False
    intent: str = ""
    route: str = ""
    sufficiency: float = 0.0
    sufficiency_threshold: float = 0.0
    attempts: int = 0
    #: ``{id, citation, version, sha256}`` per source.  Ids and versions, never text.
    sources: list[dict[str, Any]] = field(default_factory=list)

    # -- accountability ------------------------------------------------------
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    e2e_ms: float = 0.0
    degraded: bool = False
    error: str = ""
    trace_id: str = ""
    trace_span_id: str = ""

    def payload(self) -> dict[str, Any]:
        """Everything the hash covers: the record minus its own hash."""
        data = asdict(self)
        data.pop("record_hash", None)
        return data

    def compute_hash(self) -> str:
        """SHA-256 over a canonical JSON encoding of :meth:`payload`.

        Canonical means sorted keys and no insignificant whitespace, so the hash
        depends on the *content* and not on how a writer happened to serialise
        it.
        """
        canonical = json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def sealed(self) -> "AuditRecord":
        return replace(self, record_hash=self.compute_hash())

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ChainVerification:
    """Result of walking the chain from genesis."""

    ok: bool
    records: int
    verified: int
    broken_at: int | None = None
    reason: str = ""
    head_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        lines: list[str] = []
        if self.records == 0:
            return "audit log is empty — ask a question first"
        if self.ok:
            lines.append(f"CHAIN INTACT — {self.verified}/{self.records} records verified from genesis")
            lines.append(f"head hash  {self.head_hash}")
            lines.append("")
            lines.append(
                "Each record carries the hash of the one before it, so an edit cannot be "
                "made silently between the decision and the audit."
            )
        else:
            lines.append(f"CHAIN BROKEN at record {self.broken_at}")
            lines.append(f"reason     {self.reason}")
            lines.append(f"verified   {self.verified}/{self.records} before the break")
            lines.append("")
            lines.append(
                f"Every record after {self.broken_at} is now unverifiable: each one's hash "
                "depends on the one before it."
            )
        lines.append(
            "Tamper-evident, not tamper-proof: this detects a local edit. Someone who can "
            "rewrite the whole store can recompute every subsequent hash. Anchor outside "
            "the store — sign checkpoints, publish a periodic root hash to a separately "
            "controlled system, or write to WORM storage."
        )
        return "\n".join(lines)


def verify_records(rows: Sequence[dict[str, Any]]) -> ChainVerification:
    """Walk ``rows`` in order, checking sequence, linkage and each record's hash."""
    previous = GENESIS_HASH
    verified = 0
    for index, row in enumerate(rows):
        try:
            stored_hash = str(row.get("record_hash") or "")
            record = AuditRecord(**{k: v for k, v in row.items() if k in _FIELDS})
        except TypeError as exc:
            return ChainVerification(
                ok=False,
                records=len(rows),
                verified=verified,
                broken_at=index,
                reason=f"record is not readable: {exc}",
            )
        if record.sequence != index:
            return ChainVerification(
                ok=False,
                records=len(rows),
                verified=verified,
                broken_at=index,
                reason=f"sequence is {record.sequence}, expected {index} (a record was inserted or removed)",
            )
        if record.previous_hash != previous:
            return ChainVerification(
                ok=False,
                records=len(rows),
                verified=verified,
                broken_at=index,
                reason=f"previous_hash does not match the record before it",
            )
        recomputed = record.compute_hash()
        if recomputed != stored_hash:
            return ChainVerification(
                ok=False,
                records=len(rows),
                verified=verified,
                broken_at=index,
                reason=(
                    "record_hash does not match its contents — a field was edited after "
                    "the record was written"
                ),
            )
        previous = stored_hash
        verified += 1
    return ChainVerification(
        ok=True, records=len(rows), verified=verified, head_hash=previous
    )


_FIELDS = frozenset(AuditRecord.__dataclass_fields__)


# --------------------------------------------------------------------------- #
# The log
# --------------------------------------------------------------------------- #
class AuditLog:
    """Append-only, hash-chained JSONL.

    Appends are serialised: the chain is an ordering, so two concurrent writers
    would produce two records claiming the same predecessor.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._head_hash: str | None = None
        self._count: int | None = None

    # ---- reading ----------------------------------------------------------
    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            for number, line in enumerate(
                self.path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise AuditError(
                        f"{self.path}:{number} is not valid JSON: {exc}. A corrupt audit "
                        "store must be investigated, not skipped."
                    ) from exc
        except OSError as exc:
            raise AuditError(f"cannot read the audit log at {self.path}: {exc}") from exc
        return rows

    def verify(self) -> ChainVerification:
        return verify_records(self.read())

    def count(self) -> int:
        if self._count is None:
            self._load_head()
        return self._count or 0

    def head_hash(self) -> str:
        if self._head_hash is None:
            self._load_head()
        return self._head_hash or GENESIS_HASH

    def reload(self) -> None:
        """Forget the cached head, re-reading it on next use.

        Needed after anything rewrites the file underneath this object -- the
        tamper demonstration, a restore, an external prune.
        """
        with self._lock:
            self._head_hash = None
            self._count = None

    def _load_head(self) -> None:
        rows = self.read()
        self._count = len(rows)
        self._head_hash = str(rows[-1]["record_hash"]) if rows else GENESIS_HASH

    # ---- appending --------------------------------------------------------
    def append(self, **fields: Any) -> AuditRecord:
        """Seal and append one record.  Returns the sealed record."""
        with self._lock:
            if self._head_hash is None or self._count is None:
                self._load_head()
            record = AuditRecord(
                sequence=self._count or 0,
                previous_hash=self._head_hash or GENESIS_HASH,
                **fields,
            ).sealed()
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(record.to_json() + "\n")
            except OSError as exc:
                # An audit record that cannot be written is a compliance event,
                # not a debug line -- but it must not lose the user's answer.
                log.error("could not append audit record %s: %s", record.sequence, exc)
                raise AuditError(f"cannot append to the audit log at {self.path}: {exc}") from exc
            self._head_hash = record.record_hash
            self._count = record.sequence + 1
            return record

    def clear(self) -> None:
        """Delete the store.  Only for tests and the dashboard's reset."""
        with self._lock:
            if self.path.exists():
                self.path.unlink()
            self._head_hash = GENESIS_HASH
            self._count = 0

    # ---- the demonstration ------------------------------------------------
    def tamper(
        self, sequence: int = 0, field_name: str = "question", value: Any = None
    ) -> dict[str, Any]:
        """Edit one field of one record **without** recomputing hashes.

        This exists to be caught.  It is the demonstration from the talk: edit a
        decision from an hour ago and watch verification name the record.  It is
        never called by the request path.
        """
        rows = self.read()
        if not rows:
            raise AuditError("the audit log is empty; ask a question first")
        if not 0 <= sequence < len(rows):
            raise AuditError(f"record {sequence} does not exist (log has {len(rows)})")
        row = rows[sequence]
        if field_name not in row:
            raise AuditError(f"records have no field {field_name!r}")
        before = row[field_name]
        if value is None:
            value = (
                f"{before} [EDITED]" if isinstance(before, str) else not before
                if isinstance(before, bool)
                else before
            )
        row[field_name] = value
        with self._lock:
            temporary = self.path.with_suffix(".jsonl.tmp")
            temporary.write_text(
                "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
            self._head_hash = None
            self._count = None
        log.warning("audit record %s field %r edited for demonstration", sequence, field_name)
        return {"sequence": sequence, "field": field_name, "before": before, "after": value}

    # ---- anchoring --------------------------------------------------------
    def write_checkpoint(self, path: Path) -> dict[str, Any]:
        """Write a root-hash checkpoint, the anchor the chain needs.

        A checkpoint published to a separately controlled system is what turns
        "tamper-evident within this file" into "tamper-evident against a rewrite
        of this file".  Writing it next to the log demonstrates the shape; in
        production it goes somewhere the log's writer cannot reach.
        """
        verification = self.verify()
        checkpoint = {
            "records": verification.records,
            "head_hash": verification.head_hash,
            "chain_ok": verification.ok,
            "note": (
                "Publish this to a separately controlled system. A checkpoint stored "
                "beside the log it protects is a demonstration, not a control."
            ),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return checkpoint


# --------------------------------------------------------------------------- #
# Retention
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    """The two boundaries, written down so each can be defended.

    There is no single correct number.  The floor comes from AI Act Articles 19
    and 26(6); the ceiling is whatever GDPR's storage-limitation principle lets
    you justify for the purpose.  Both are recorded here rather than encoded as
    one constant, because "we keep it for N days" is only defensible with the
    reason attached.
    """

    floor_days: int = RETENTION_FLOOR_DAYS
    configured_days: int = RETENTION_FLOOR_DAYS
    purpose: str = "demonstrate and investigate automated decisions about employment rights"
    floor_basis: str = "EU AI Act Articles 19 and 26(6): appropriate to purpose, generally at least six months"
    ceiling_basis: str = "GDPR storage limitation: no longer than necessary for the purpose, and justifiable"

    def validate(self) -> None:
        if self.configured_days < self.floor_days:
            raise ValueError(
                f"configured retention of {self.configured_days} days is below the "
                f"{self.floor_days}-day floor ({self.floor_basis})"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def expired(
    rows: Iterable[dict[str, Any]], policy: RetentionPolicy, now_iso: str
) -> list[int]:
    """Sequence numbers older than the configured period.

    Reported rather than acted on: deleting audit records is a decision with a
    legal basis attached, not a background job.
    """
    from datetime import datetime, timedelta, timezone

    def parse(value: str) -> Any:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    reference = parse(now_iso) or datetime.now(timezone.utc)
    cutoff = reference - timedelta(days=policy.configured_days)
    out: list[int] = []
    for row in rows:
        stamp = parse(str(row.get("ts") or ""))
        if stamp is not None and stamp < cutoff:
            out.append(int(row.get("sequence", -1)))
    return out
