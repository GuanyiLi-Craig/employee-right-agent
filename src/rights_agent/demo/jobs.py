"""Background jobs behind the dashboard's controls.

One worker thread, one job at a time, an append-only output log.  The UI polls a
snapshot; nothing it does blocks the server.  Job exceptions land **in the
output log**, not in a terminal nobody is projecting -- a traceback the room
cannot see is a traceback that did not happen.

Every job calls the same modules the tests call.  There is no demo-only path.
"""

from __future__ import annotations

import random
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rights_agent.agent import Agent
from rights_agent.analysis import drift_report
from rights_agent.audit import AuditError
from rights_agent.config import PRICING, Settings
from rights_agent.costs import render_repricing, reprice_components
from rights_agent.datasets import require_datasets_dir
from rights_agent.log import get_logger
from rights_agent.metrics import MetricsSink
from rights_agent.tools.evaluate import (
    EVALS_DIR,
    calibration_report,
    gate_report,
    read_jsonl,
)

log = get_logger("demo.jobs")

MAX_OUTPUT_ENTRIES = 40

#: The gate always evaluates the offline stub, whatever is configured for
#: serving. Kept in step with ``evals/conftest.py``, which pins the same model
#: for the same reason.
# One definition, in the gate's own module: three call sites pinned this
# independently and the CLI's had drifted to no pin at all.
from rights_agent.tools.evaluate import GATE_MODEL  # noqa: E402

#: Default sizes for the traffic controls.  Named so the button labels and the
#: job agree: an "Incident · 18" button that sends 12 requests is a small lie
#: that costs you the room's trust at exactly the wrong moment.
BASELINE_REQUESTS = 24
INCIDENT_REQUESTS = 18
INTENT_SHIFT_REQUESTS = 20


@dataclass
class OutputEntry:
    job: str
    started_at: float
    finished_at: float
    ok: bool
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "job": self.job,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round(self.finished_at - self.started_at, 2),
            "ok": self.ok,
            "text": self.text,
        }


@dataclass
class JobState:
    name: str = ""
    running: bool = False
    started_at: float = 0.0
    progress: str = ""
    done: int = 0
    total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "running": self.running,
            "elapsed_s": round(time.time() - self.started_at, 1) if self.running else 0.0,
            "progress": self.progress,
            "done": self.done,
            "total": self.total,
        }


class JobRunner:
    """Runs one job at a time on a worker thread."""

    def __init__(self, agent: Agent, settings: Settings, evals_dir: Path = EVALS_DIR) -> None:
        self.agent = agent
        self.settings = settings
        self.evals_dir = evals_dir
        self.state = JobState()
        self.output: list[OutputEntry] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._gate: Agent | None = None
        self.jobs: dict[str, Callable[[dict[str, Any]], str]] = {
            "baseline_traffic": self._baseline_traffic,
            "incident_traffic": self._incident_traffic,
            "shift_intents": self._shift_intents,
            "ci_gate": self._ci_gate,
            "calibrate_judge": self._calibrate_judge,
            "drift_report": self._drift_report,
            "reprice": self._reprice,
            "verify_audit": self._verify_audit,
            "tamper_audit": self._tamper_audit,
            "reset": self._reset,
        }

    # ---- lifecycle --------------------------------------------------------
    def start(self, name: str, params: dict[str, Any] | None = None) -> tuple[bool, str]:
        """Begin a job unless one is already running."""
        if name not in self.jobs:
            return False, f"unknown job {name!r}; expected one of {sorted(self.jobs)}"
        with self._lock:
            if self.state.running:
                return False, f"{self.state.name} is still running"
            self.state = JobState(name=name, running=True, started_at=time.time())
        self._thread = threading.Thread(
            target=self._run, args=(name, params or {}), name=f"job-{name}", daemon=True
        )
        self._thread.start()
        return True, f"started {name}"

    def _run(self, name: str, params: dict[str, Any]) -> None:
        started = time.time()
        try:
            text = self.jobs[name](params)
            ok = True
        except Exception as exc:  # noqa: BLE001 - the output panel is where this belongs
            ok = False
            text = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=6)}"
            log.exception("job %s failed", name)
        finally:
            with self._lock:
                self.output.append(
                    OutputEntry(
                        job=name,
                        started_at=started,
                        finished_at=time.time(),
                        ok=ok,
                        text=text,
                    )
                )
                del self.output[:-MAX_OUTPUT_ENTRIES]
                self.state = JobState()

    def _progress(self, label: str) -> Callable[[int, int], None]:
        def report(done: int, total: int) -> None:
            with self._lock:
                self.state.progress = f"{label} {done}/{total}"
                self.state.done = done
                self.state.total = total

        return report

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "job": self.state.to_dict(),
                "output": [entry.to_dict() for entry in reversed(self.output[-8:])],
                "jobs_available": sorted(self.jobs),
            }

    # ---- jobs -------------------------------------------------------------
    @property
    def _datasets(self) -> Path:
        """The dataset directory for the embedder the live index was built with.

        Resolved per call rather than at construction: an operator can rebuild
        the index under a different embedder while the dashboard is running, and
        reading the previous embedder's golden set would quietly measure the
        wrong thing.
        """
        manifest = self.agent.deps.retriever.manifest if self.agent.deps.retriever else None
        embedder = manifest.embedding_model if manifest else self.settings.embedder
        return require_datasets_dir(self.evals_dir, embedder)

    def _questions(self) -> list[str]:
        rows = read_jsonl(self._datasets / "golden.jsonl")
        return [str(row["question"]) for row in rows if not row.get("should_refuse")]

    def _traffic(self, params: dict[str, Any], *, degraded: bool, label: str) -> str:
        count = int(params.get("count", BASELINE_REQUESTS))
        questions = self._questions()
        if not questions:
            return "no golden questions available"
        # One seed for both traffic modes, so baseline and degraded ask the *same*
        # questions.  If the question mix moved too, a drift report cannot tell
        # whether the model changed or the traffic did -- which is the confound
        # the "shift intents" control exists to demonstrate deliberately.
        chooser = random.Random(1234)
        selection = [chooser.choice(questions) for _ in range(count)]
        report = self._progress(label)

        previous = self.agent.degraded
        self.agent.set_degraded(degraded)
        session = f"{label}-{int(time.time())}"
        try:
            refused = 0
            costs = 0.0
            for index, question in enumerate(selection, start=1):
                answer = self.agent.ask(
                    question, session_id=session, user_id="demo", remember=False
                )
                refused += int(answer.refused)
                costs += answer.metrics.cost_usd
                report(index, len(selection))
        finally:
            self.agent.set_degraded(previous)
        summary = (
            f"{len(selection)} requests as session {session}\n"
            f"mode: {'degraded fallback' if degraded else 'healthy'}\n"
            f"refused: {refused}\ncost: ${costs:.6f}"
        )
        if not degraded:
            return (
                f"{summary}\n\n"
                "Percentiles, never a mean. Note the refusal rate is a quality metric, not "
                "an error rate.\n\nNow run the incident: it asks the SAME questions, so the "
                "only thing that changes is how they are answered."
            )
        return (
            f"{summary}\n\n"
            "Read the two panels together. TTFT p50 jumps and citation coverage collapses in "
            "the same window.\n\nSeparately, each is the kind of thing a busy team closes as "
            '"no repro". Together they point at one story.\n\n'
            "Be precise about what that establishes: the two series tell you WHEN something "
            "changed and that the symptoms share a cause. They do not prove which model "
            "answered — the trace does.\n\n"
            + self._quality_caveat()
        )

    def _quality_caveat(self) -> str:
        """State what the panels actually show, not what they usually show.

        The two metrics do not move together, and *how far* groundedness moves
        depends on the model: an extractive one lifts sentences verbatim and a
        lexical judge stays happy, while a capable one paraphrases and the same
        judge marks it down. Reading a scripted number off a screen that
        disagrees with it is worse than saying nothing, so this measures.
        """
        rows = MetricsSink(self.settings.metrics_path).read()
        healthy = [r for r in rows if not r.get("degraded") and r.get("scores")]
        degraded = [r for r in rows if r.get("degraded") and r.get("scores")]
        if not healthy or not degraded:
            return (
                "Run the baseline first: the comparison needs healthy traffic in the same "
                "window."
            )

        def mean_of(window: list[dict[str, Any]], metric: str) -> float:
            values = [float(r["scores"][metric]) for r in window if metric in r["scores"]]
            return sum(values) / len(values) if values else 0.0

        moves = {
            metric: (mean_of(healthy, metric), mean_of(degraded, metric))
            for metric in ("citation_coverage", "groundedness")
        }
        lines = [
            "The two quality signals did not move together:",
            *(
                f"  {metric:<18} {before:.3f} -> {after:.3f}  ({after - before:+.3f})"
                for metric, (before, after) in moves.items()
            ),
            "",
        ]
        citation_drop = moves["citation_coverage"][0] - moves["citation_coverage"][1]
        ground_drop = moves["groundedness"][0] - moves["groundedness"][1]
        if ground_drop < citation_drop / 3:
            lines.append(
                "Groundedness barely moved: this fallback still lifts sentences out of the "
                "retrieved text, so a lexical check stays happy while the answers become "
                "uncitable."
            )
        else:
            lines.append(
                "Groundedness moved too, and for a reason worth naming: a capable model "
                "paraphrases rather than quoting, and a lexical judge cannot see paraphrase "
                "— the blind spot the calibration demo puts a number on. Citation coverage "
                "is still the signal that collapses furthest and fastest."
            )
        lines.append(
            "Either way: one headline quality number would have hidden this. That is the "
            "argument for a small family of signals rather than a single score."
        )
        return "\n".join(lines)

    def _baseline_traffic(self, params: dict[str, Any]) -> str:
        params.setdefault("count", BASELINE_REQUESTS)
        return self._traffic(params, degraded=False, label="baseline")

    def _incident_traffic(self, params: dict[str, Any]) -> str:
        params.setdefault("count", INCIDENT_REQUESTS)
        return self._traffic(params, degraded=True, label="incident")

    def _shift_intents(self, params: dict[str, Any]) -> str:
        """Send traffic weighted toward one topic.

        The point: a quality change can be a change in the *questions*.  Without
        an intent mix on every row you cannot tell the two apart.

        Intent is classified by keyword, deliberately.  An embedding classifier
        has better coverage, but a label that changes when you re-run it makes
        every time series meaningless -- and if you group metrics by a dimension,
        that dimension has to be stable.
        """
        intent = str(params.get("intent", "enforcement"))
        count = int(params.get("count", INTENT_SHIFT_REQUESTS))
        rows = read_jsonl(self._datasets / "golden.jsonl")
        matching = [
            str(row["question"])
            for row in rows
            if row.get("intent") == intent and not row.get("should_refuse")
        ]
        if not matching:
            available = sorted({str(row.get("intent")) for row in rows})
            return f"no golden questions with intent {intent!r}; available: {available}"
        report = self._progress(f"{intent} traffic")
        session = f"intent-shift-{intent}-{int(time.time())}"
        for index in range(1, count + 1):
            self.agent.ask(
                matching[index % len(matching)],
                session_id=session,
                user_id="demo",
                remember=False,
            )
            report(index, count)
        return (
            f"{count} requests, all intent={intent!r}, as session {session}\n\n"
            "Now run the drift report: the intent mix moved. A quality delta in this window "
            "is a change in the questions, not necessarily in the system."
        )

    def _gate_agent(self) -> Agent:
        """An agent pinned to the offline stub, for the gate only.

        The gate must reproduce what CI asserts, and CI is offline. Running the
        golden set through whatever hosted model happens to be configured would
        make this button slow, non-deterministic and billable -- and it would no
        longer be the same check that blocks a merge.

        It shares the retriever, the audit chain and the metrics sink with the
        serving agent, so the index is opened once and the integrity gate still
        verifies the chain the rest of the process is writing to.
        """
        if self._gate is None:
            from rights_agent.graph import AgentDeps

            settings = self.settings.with_overrides(model=GATE_MODEL)
            assert self.agent.deps.retriever is not None
            self._gate = Agent(
                settings,
                deps=AgentDeps(settings=settings, retriever=self.agent.deps.retriever),
                sink=self.agent.sink,
                audit=self.agent.audit,
                conversations=self.agent.conversations,
                init_tracing=False,
            )
        return self._gate

    def _ci_gate(self, params: dict[str, Any]) -> str:
        report = gate_report(
            self._gate_agent(),
            evals_dir=self.evals_dir,
            progress=self._progress("golden rows"),
        )
        note = ""
        if self.settings.model != GATE_MODEL:
            note = (
                f"\n\nRun against {GATE_MODEL}, not {self.settings.model}: the gate has to "
                "reproduce what CI asserts, and CI is offline. A merge gate that depends on a "
                "hosted provider's availability, latency and price list is not a gate."
            )
        return report.render() + note

    def _calibrate_judge(self, params: dict[str, Any]) -> str:
        return calibration_report(
            rows=read_jsonl(self._datasets / "calibration.jsonl"),
            threshold=float(params.get("threshold", 0.7)),
        ).render()

    def _drift_report(self, params: dict[str, Any]) -> str:
        rows = MetricsSink(self.settings.metrics_path).read()
        return drift_report(rows, split=float(params.get("split", 0.5))).render()

    def _reprice(self, params: dict[str, Any]) -> str:
        """The same recorded traffic at two prices.  Nothing is re-run."""
        left = str(params.get("baseline_model", "claude-haiku-4-5"))
        right = str(params.get("model", "claude-sonnet-5"))
        for model in (left, right):
            if model not in PRICING:
                return f"unknown model {model!r}; the table knows {sorted(PRICING)}"
        rows = MetricsSink(self.settings.metrics_path).read()
        if not rows:
            return "no recorded requests to reprice; generate some traffic first"
        requests_per_day = int(
            params.get("requests_per_day", self.settings.projection_requests_per_day)
        )
        rendered = render_repricing(
            reprice_components(
                rows, left, requests_per_day=requests_per_day,
                judge_sample_rate=self.settings.judge_sample_rate,
                judge_model=self.settings.judge_model,
            ),
            reprice_components(
                rows, right, requests_per_day=requests_per_day,
                judge_sample_rate=self.settings.judge_sample_rate,
                judge_model=self.settings.judge_model,
            ),
        )
        return (
            f"{rendered}\n\n"
            "Recorded tokens plus a price table is a cost model, with no new "
            "instrumentation — and it makes routing easy questions to the cheap model an "
            "arithmetic question rather than a taste one.\n\n"
            "Every figure came from one dated table. Change the table and the whole model "
            "moves; that property is the difference between a cost model and a spreadsheet."
        )

    # ---- the audit chain --------------------------------------------------
    def _verify_audit(self, params: dict[str, Any]) -> str:
        verification = self.agent.audit.verify()
        checkpoint = self.agent.audit.write_checkpoint(self.settings.audit_checkpoint_path)
        return (
            f"{verification.render()}\n\n"
            f"checkpoint written to {self.settings.audit_checkpoint_path.name}: "
            f"{checkpoint['records']} records, head {checkpoint['head_hash'][:16]}…"
        )

    def _tamper_audit(self, params: dict[str, Any]) -> str:
        """Verify, edit one field of one record, verify again.

        The record edited defaults to the **first** one -- a decision from an
        hour ago -- because that is what makes the consequence visible: every
        record after it becomes unverifiable too.
        """
        audit = self.agent.audit
        before = audit.verify()
        if before.records == 0:
            return "the audit log is empty — ask a question or run some traffic first"
        sequence = int(params.get("sequence", 0))
        field_name = str(params.get("field", "question"))
        try:
            edit = audit.tamper(sequence, field_name)
        except AuditError as exc:
            return str(exc)
        after = audit.verify()
        return (
            "BEFORE\n"
            f"{before.render()}\n\n"
            f"EDIT   record {edit['sequence']}, field {edit['field']!r}\n"
            f"       {edit['before']!r}\n"
            f"    -> {edit['after']!r}\n\n"
            "AFTER\n"
            f"{after.render()}\n\n"
            "This is not a blockchain and does not need to be. Each record carries the hash "
            "of the one before it, so an edit cannot be made SILENTLY between the decision "
            "and the audit.\n\n"
            "Run Reset to rebuild a clean chain."
        )

    def _reset(self, params: dict[str, Any]) -> str:
        MetricsSink(self.settings.metrics_path).clear()
        self.agent.audit.clear()
        self.agent.conversations.clear()
        self.agent.set_degraded(False)
        return (
            "metrics cleared, audit chain restarted from genesis, transcripts dropped, "
            "degraded mode switched off"
        )
