"""Run the golden set as an experiment, and calibrate the judge.

    uv run python -m rights_agent evaluate            # local summary
    uv run python -m rights_agent evaluate --phoenix  # also push to Phoenix

Phoenix's dataset and experiment API moves between releases, so every call into
it is wrapped and falls back to a local summary.  **The lesson survives without
the UI**; a broken import must not break the demo.

The same functions back the dashboard's buttons, so what the room sees is what
the tests assert.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Mapping, Sequence

from rights_agent.agent import Agent, AgentAnswer
from rights_agent.config import PROMPT_VERSION, Settings, settings as load_settings
from rights_agent.datasets import require_datasets_dir
from rights_agent.judges import Calibration, HeuristicJudge, Judge, JudgeScores, calibrate
from rights_agent.log import configure_logging, get_logger
from rights_agent.metrics import percentile
from rights_agent.store import load_manifest
from rights_agent.entrypoints import operator_error_exit

log = get_logger("tools.evaluate")

#: The model the gate runs against, whatever is serving the chat.
#:
#: A merge gate that depends on a provider's availability, latency and price
#: list is not a gate. The pytest suite and the dashboard's gate job both pinned
#: this; the CLI did not, so ``evaluate --gate`` graded a non-deterministic
#: hosted model against thresholds measured on the stub and reported GATE FAILED
#: for a p10 that moves between runs.
GATE_MODEL = "stub-local"



EVALS_DIR = Path("evals")
DIMENSIONS = ("groundedness", "citation_coverage", "context_relevance", "answer_relevance")


def shippable(scores: JudgeScores) -> float:
    """Supported *and* attributable -- the pair that decides shippability."""
    return min(scores.groundedness, scores.citation_coverage)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate the eval datasets with "
            "`uv run python -m rights_agent goldens --write-baseline`."
        )
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------------- #
# Local experiment
# --------------------------------------------------------------------------- #
@dataclass
class RowResult:
    id: str
    question: str
    intent: str
    expected: list[str]
    should_refuse: bool
    known_failure: bool
    refused: bool
    citations: list[str]
    retrieved: list[str]
    sufficiency: float
    scores: dict[str, float]
    cost_usd: float
    e2e_ms: float
    citation_hit: bool
    #: Carried so the Phoenix experiment can record *this* run rather than
    #: provoking another one.
    answer: str = ""
    #: The model that actually served, which is not always the one configured.
    model: str = ""

    @classmethod
    def from_answer(cls, row: dict[str, Any], answer: AgentAnswer) -> "RowResult":
        retrieved = [str(doc.get("citation", "")) for doc in answer.docs]
        provisions = {citation.split("(")[0].strip() for citation in retrieved}
        expected = list(row.get("must_cite") or [])
        return cls(
            id=str(row["id"]),
            question=str(row["question"]),
            intent=str(row.get("intent", "")),
            expected=expected,
            should_refuse=bool(row.get("should_refuse")),
            known_failure=bool(row.get("known_failure")),
            refused=answer.refused,
            citations=list(answer.citations),
            retrieved=retrieved,
            sufficiency=round(answer.sufficiency, 4),
            scores=dict(answer.scores),
            cost_usd=answer.metrics.cost_usd,
            e2e_ms=answer.metrics.e2e_ms,
            citation_hit=bool(expected)
            and all(citation.split("(")[0] in provisions for citation in expected),
            answer=answer.answer,
            model=answer.metrics.model,
        )


@dataclass
class ExperimentSummary:
    index_version: str
    prompt_version: str
    model: str
    rows: int
    answerable: int
    refusal_rows: int
    known_failures: int
    citation_hit_rate: float
    refusal_accuracy: float
    quality: dict[str, dict[str, float]]
    per_intent: dict[str, dict[str, float]]
    cost_usd_total: float
    e2e_ms_p95: float
    phoenix: str = "not attempted"
    results: list[RowResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["results"] = [asdict(result) for result in self.results]
        return payload

    def render(self) -> str:
        lines = [
            f"index_version      {self.index_version}",
            f"prompt_version     {self.prompt_version}",
            f"model              {self.model}",
            f"rows               {self.rows} "
            f"({self.answerable} answerable, {self.refusal_rows} refusal, "
            f"{self.known_failures} known failure)",
            f"citation hit rate  {self.citation_hit_rate:.3f}  (answerable rows)",
            f"refusal accuracy   {self.refusal_accuracy:.3f}  (out-of-scope rows refused)",
            "",
            f"{'dimension':<20}{'mean':>8}{'p10':>8}{'min':>8}",
        ]
        for name in DIMENSIONS:
            stats = self.quality.get(name, {})
            lines.append(
                f"{name:<20}{stats.get('mean', 0):>8.3f}{stats.get('p10', 0):>8.3f}"
                f"{stats.get('min', 0):>8.3f}"
            )
        lines += ["", f"{'intent':<16}{'rows':>6}{'groundedness':>14}{'citations':>11}"]
        for intent, stats in sorted(self.per_intent.items()):
            lines.append(
                f"{intent:<16}{int(stats['rows']):>6}{stats['groundedness']:>14.3f}"
                f"{stats['citation_coverage']:>11.3f}"
            )
        lines += [
            "",
            f"total cost         ${self.cost_usd_total:.6f}",
            f"e2e p95            {self.e2e_ms_p95:.1f} ms",
            f"phoenix            {self.phoenix}",
        ]
        return "\n".join(lines)


def run_experiment_locally(
    agent: Agent,
    golden_rows: Sequence[dict[str, Any]],
    *,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> ExperimentSummary:
    """Run every golden row and aggregate.  No Phoenix required."""
    rows = list(golden_rows)[:limit] if limit else list(golden_rows)
    results: list[RowResult] = []
    for index, row in enumerate(rows, start=1):
        answer = agent.ask(
            str(row["question"]), session_id=f"experiment-{row['id']}", remember=False
        )
        results.append(RowResult.from_answer(row, answer))
        if progress is not None:
            progress(index, len(rows))

    answerable = [r for r in results if not r.should_refuse and not r.known_failure]
    refusal_rows = [r for r in results if r.should_refuse]

    quality: dict[str, dict[str, float]] = {}
    for name in DIMENSIONS:
        values = [r.scores[name] for r in answerable if name in r.scores]
        quality[name] = {
            "mean": round(mean(values), 4) if values else 0.0,
            "p10": round(percentile(values, 0.10), 4),
            "min": round(min(values), 4) if values else 0.0,
            "n": len(values),
        }

    per_intent: dict[str, dict[str, float]] = {}
    for result in answerable:
        bucket = per_intent.setdefault(
            result.intent or "unknown", {"rows": 0.0, "groundedness": 0.0, "citation_coverage": 0.0}
        )
        bucket["rows"] += 1
        bucket["groundedness"] += result.scores.get("groundedness", 0.0)
        bucket["citation_coverage"] += result.scores.get("citation_coverage", 0.0)
    for bucket in per_intent.values():
        count = bucket["rows"] or 1
        bucket["groundedness"] = round(bucket["groundedness"] / count, 4)
        bucket["citation_coverage"] = round(bucket["citation_coverage"] / count, 4)

    return ExperimentSummary(
        index_version=agent.index_version,
        prompt_version=PROMPT_VERSION,
        model=agent.settings.model,
        rows=len(results),
        answerable=len(answerable),
        refusal_rows=len(refusal_rows),
        known_failures=sum(1 for r in results if r.known_failure),
        citation_hit_rate=round(
            sum(1 for r in answerable if r.citation_hit) / len(answerable), 4
        )
        if answerable
        else 0.0,
        refusal_accuracy=round(
            sum(1 for r in refusal_rows if r.refused) / len(refusal_rows), 4
        )
        if refusal_rows
        else 1.0,
        quality=quality,
        per_intent=per_intent,
        cost_usd_total=round(sum(r.cost_usd for r in results), 6),
        e2e_ms_p95=round(percentile([r.e2e_ms for r in results], 0.95), 2),
        results=results,
    )


# --------------------------------------------------------------------------- #
# Phoenix datasets and experiments
# --------------------------------------------------------------------------- #
def push_to_phoenix(
    agent: Agent,
    golden_rows: Sequence[dict[str, Any]],
    settings: Settings,
    results: Sequence[RowResult],
) -> str:
    """Upload the golden set and record ``results`` as a Phoenix experiment.

    Takes the results of the run that has already happened rather than running
    the set again. Re-running produced two different sets of numbers for the
    same command -- a hosted model does not repeat itself, so the table printed
    to the terminal disagreed with the experiment in Phoenix by a couple of
    points -- and charged twice for the privilege. The whole argument for the
    trace is that it is the same event you are looking at.

    Returns a human-readable status.  Tagged with ``index_version`` and
    ``prompt_version`` so two runs are comparable -- an experiment you cannot
    attribute to an index is an experiment you cannot act on.
    """
    by_question = {result.question: result for result in results}
    try:
        from phoenix.client import Client

        client = Client(base_url=settings.phoenix_endpoint)
        dataset_name = f"golden-{agent.index_version}"
        # Each example is nested: ``input`` / ``output`` / ``metadata``. The
        # ``*_keys`` arguments belong to the dataframe and CSV paths, and flat
        # rows passed with them are rejected outright -- the kind of thing that
        # only shows up the first time you point this at a real server.
        examples = [
            {
                "input": {"question": str(row["question"])},
                "output": {
                    "expected_citations": ", ".join(row.get("must_cite") or []),
                    "should_refuse": bool(row.get("should_refuse")),
                },
                "metadata": {
                    "id": str(row["id"]),
                    "intent": str(row.get("intent", "")),
                    "known_failure": bool(row.get("known_failure")),
                    "note": str(row.get("note", "")),
                    "index_version": agent.index_version,
                    "prompt_version": PROMPT_VERSION,
                },
            }
            for row in golden_rows
            if row["question"] in by_question
        ]
        dataset = client.datasets.create_dataset(
            name=dataset_name,
            examples=examples,
            dataset_description=(
                f"Golden set for {agent.index_version}. Asserts citations, not prose: "
                "wording changes when models change, the source does not."
            ),
        )

        def task(example: Any) -> dict[str, Any]:
            question = str(_example_input(example).get("question", ""))
            result = by_question.get(question)
            if result is None:
                # Never silently re-ask: that is the bug this signature exists to
                # remove, and it would show up as two answers to one question.
                raise KeyError(f"no local result for {question!r}")
            return {
                "answer": result.answer,
                "citations": ", ".join(result.citations),
                "refused": result.refused,
                # Which model actually served, so an experiment cannot be
                # attributed to a model that fell back.
                "model": result.model,
                "cost_usd": result.cost_usd,
                "e2e_ms": result.e2e_ms,
                **result.scores,
            }

        experiment = client.experiments.run_experiment(
            dataset=dataset,
            task=task,
            evaluators=[_evaluator_groundedness, _evaluator_citation_coverage],
            experiment_name=f"golden-{agent.index_version}",
            experiment_metadata={
                "index_version": agent.index_version,
                "prompt_version": PROMPT_VERSION,
                "model": agent.settings.model,
            },
        )
        # ``RanExperiment`` is a mapping with ``experiment_id`` / ``dataset_id``,
        # and the URL helper needs both. Read defensively: this is the surface
        # that moves between releases, which is why the whole function is wrapped.
        experiment_id = _field(experiment, "experiment_id")
        dataset_id = _field(experiment, "dataset_id") or _field(dataset, "id")
        url = ""
        if experiment_id and dataset_id:
            try:
                url = client.experiments.get_experiment_url(
                    dataset_id=str(dataset_id), experiment_id=str(experiment_id)
                )
            except Exception as exc:  # noqa: BLE001 - a missing link is not a failure
                log.debug("could not build the experiment URL: %s", exc)
        runs = len(_field(experiment, "task_runs") or ())
        evaluations = len(_field(experiment, "evaluation_runs") or ())
        lines = [
            f"uploaded {len(examples)} rows to dataset {dataset_name!r}",
            f"experiment ran {runs} tasks with {evaluations} evaluations",
        ]
        if url:
            lines.append(f"open {_browsable(url)}")
        return " · ".join(lines)
    except Exception as exc:  # noqa: BLE001 - deliberate: the API surface moves
        log.warning("Phoenix experiment unavailable: %s", exc)
        return f"unavailable ({type(exc).__name__}: {exc}); local summary used instead"


#: Hosts that only resolve inside the compose network. A link nobody can click
#: is worse than a path, so the path is offered instead.
_INTERNAL_HOSTS = ("phoenix", "host.docker.internal")


def _browsable(url: str) -> str:
    """Make a URL usable from the machine running the browser.

    Inside compose the Phoenix client talks to ``http://phoenix:6006``, a name
    that resolves only on that network. Handing a presenter that link wastes a
    minute on stage, so the path is given relative to whatever their Phoenix URL
    is.
    """
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    if parts.hostname in _INTERNAL_HOSTS:
        path = parts.path + (f"?{parts.query}" if parts.query else "")
        return f"<your Phoenix UI>{path}"
    return url


def _field(obj: Any, name: str) -> Any:
    """Read ``name`` from a mapping or an object, whichever Phoenix returned."""
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _example_input(example: Any) -> dict[str, Any]:
    """Read an example's input across Phoenix client shapes."""
    for attribute in ("input", "inputs"):
        value = getattr(example, attribute, None)
        if isinstance(value, dict):
            return value
    if isinstance(example, dict):
        for key in ("input", "inputs"):
            if isinstance(example.get(key), dict):
                return example[key]
        return example
    return {}


def _evaluator_groundedness(output: Any = None, **_: Any) -> float:
    if isinstance(output, dict):
        return float(output.get("groundedness", 0.0))
    return 0.0


def _evaluator_citation_coverage(output: Any = None, **_: Any) -> float:
    if isinstance(output, dict):
        return float(output.get("citation_coverage", 0.0))
    return 0.0


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationReport:
    judge: str
    full: Calibration
    clean_only: Calibration
    groundedness_only: Calibration

    def to_dict(self) -> dict[str, Any]:
        return {
            "judge": self.judge,
            "full": self.full.to_dict(),
            "clean_only": self.clean_only.to_dict(),
            "groundedness_only": self.groundedness_only.to_dict(),
        }

    def render(self) -> str:
        lines = [
            f"judge              {self.judge}",
            "",
            f"{'set':<24}{'n':>4}{'kappa':>9}{'agreement':>11}",
            f"{'clean examples only':<24}{self.clean_only.n:>4}"
            f"{self.clean_only.kappa:>9.3f}{self.clean_only.agreement:>11.3f}",
            f"{'plus realistic cases':<24}{self.full.n:>4}"
            f"{self.full.kappa:>9.3f}{self.full.agreement:>11.3f}",
            f"{'groundedness alone':<24}{self.groundedness_only.n:>4}"
            f"{self.groundedness_only.kappa:>9.3f}{self.groundedness_only.agreement:>11.3f}",
            "",
            "Read those three in order.",
            "",
            "Clean examples alone give a perfect kappa -- and tell you nothing. That figure is",
            "a property of the examples, not of the judge, and it is exactly how teams end up",
            "trusting an evaluator that cannot do its job.",
            "",
            "The judge did not get worse when the realistic cases were added. The test got",
            "honest.",
            "",
            "The third line is the payoff: gating on groundedness ALONE is worse than gating",
            "on groundedness AND citation coverage together. Two complementary signals beat",
            "either alone, because they fail on different cases -- combining two signals that",
            "fail the same way buys nothing.",
            "",
            "Kappa corrects for chance: if 90% of answers are good, a judge that always says",
            "'good' scores 90% agreement and is worth nothing. Kappa removes that floor.",
            "",
            "disagreements:",
        ]
        for entry in self.full.disagreements:
            lines.append(
                f"  {entry['id']:<6} human {entry['human_label']} machine {entry['machine_label']} "
                f"score {float(entry['score']):.2f}  {entry['note']}"
            )
        if not self.full.disagreements:
            lines.append("  (none -- suspicious: are the hard cases still in the set?)")
        return "\n".join(lines)


def calibration_report(
    judge: Judge | None = None,
    rows: Sequence[dict[str, Any]] | None = None,
    threshold: float = 0.7,
) -> CalibrationReport:
    judge = judge or HeuristicJudge()
    rows = rows if rows is not None else read_jsonl(EVALS_DIR / "calibration.jsonl")
    clean = [row for row in rows if str(row["id"]).startswith("c")]
    return CalibrationReport(
        judge=getattr(judge, "name", "unknown"),
        full=calibrate(judge, rows, scorer=shippable, threshold=threshold),
        clean_only=calibrate(judge, clean, scorer=shippable, threshold=threshold),
        groundedness_only=calibrate(judge, rows, threshold=threshold),
    )


# --------------------------------------------------------------------------- #
# The gate, reported rather than asserted
# --------------------------------------------------------------------------- #
@dataclass
class Gate:
    name: str
    observed: float
    threshold: float
    passed: bool
    note: str = ""


@dataclass
class GateReport:
    """The same numbers ``pytest evals/`` asserts, rendered instead of raised.

    The suites remain the gate -- this is for showing the room what the gate is
    looking at, and for a dashboard button that must not shell out to pytest in
    a slim container.  Both read the same per-embedder ``baseline.json``.
    """

    index_version: str
    gates: list[Gate]
    summary: ExperimentSummary
    #: The model the gate actually graded with, which is deliberately not the
    #: model serving the chat. Printed because a threshold measured against one
    #: model says nothing about another.
    gate_model: str = GATE_MODEL

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_version": self.index_version,
            "gate_model": self.gate_model,
            "passed": self.passed,
            "gates": [asdict(gate) for gate in self.gates],
        }

    def render(self) -> str:
        lines = [
            f"index_version      {self.index_version}",
            f"gate model         {self.gate_model}",
            "",
            f"{'gate':<28}{'observed':>10}{'threshold':>11}  result",
        ]
        for gate in self.gates:
            verdict = "PASS" if gate.passed else "FAIL"
            lines.append(
                f"{gate.name:<28}{gate.observed:>10.3f}{gate.threshold:>11.3f}  {verdict}"
                + (f"   {gate.note}" if gate.note else "")
            )
        lines += ["", "GATE PASSED" if self.passed else "GATE FAILED"]
        lines.append("`uv run pytest evals/ -q` is the gate; these are the numbers it asserts.")
        return "\n".join(lines)


def _datasets_for(agent: Agent, evals_dir: Path) -> Path:
    """The dataset directory matching the index the agent is actually reading.

    Taken from the manifest rather than from settings: settings say what was
    *preferred*, the manifest says what the vectors in the index were built with,
    and only the second one decides which golden rows are true.
    """
    manifest = agent.deps.retriever.manifest if agent.deps.retriever else None
    embedder = manifest.embedding_model if manifest else agent.settings.embedder
    return require_datasets_dir(evals_dir, embedder)


def gate_report(
    agent: Agent,
    *,
    evals_dir: Path = EVALS_DIR,
    progress: Callable[[int, int], None] | None = None,
) -> GateReport:
    datasets = _datasets_for(agent, evals_dir)
    baseline = json.loads((datasets / "baseline.json").read_text(encoding="utf-8"))
    thresholds = {key: float(value) for key, value in baseline["quality_thresholds"].items()}
    golden = read_jsonl(datasets / "golden.jsonl")
    calibration = read_jsonl(datasets / "calibration.jsonl")

    report = calibration_report(rows=calibration)
    summary = run_experiment_locally(agent, golden, progress=progress)
    gate_model = agent.settings.model

    gates = [
        # The instrument is gated before anything measured with it.
        Gate(
            "judge kappa",
            report.full.kappa,
            thresholds["judge_kappa"],
            report.full.kappa >= thresholds["judge_kappa"],
            "gate the instrument first",
        ),
        Gate(
            "groundedness mean",
            summary.quality["groundedness"]["mean"],
            thresholds["groundedness_mean"],
            summary.quality["groundedness"]["mean"] >= thresholds["groundedness_mean"],
        ),
        Gate(
            "groundedness p10",
            summary.quality["groundedness"]["p10"],
            thresholds["groundedness_p10"],
            summary.quality["groundedness"]["p10"] >= thresholds["groundedness_p10"],
            "the tail, not the mean",
        ),
        Gate(
            "citation coverage mean",
            summary.quality["citation_coverage"]["mean"],
            thresholds["citation_coverage_mean"],
            summary.quality["citation_coverage"]["mean"] >= thresholds["citation_coverage_mean"],
        ),
        Gate(
            "context relevance mean",
            summary.quality["context_relevance"]["mean"],
            thresholds["context_relevance_mean"],
            summary.quality["context_relevance"]["mean"] >= thresholds["context_relevance_mean"],
            "blames retrieval",
        ),
        Gate(
            "answer relevance mean",
            summary.quality["answer_relevance"]["mean"],
            thresholds["answer_relevance_mean"],
            summary.quality["answer_relevance"]["mean"] >= thresholds["answer_relevance_mean"],
        ),
        Gate("refusal accuracy", summary.refusal_accuracy, 1.0, summary.refusal_accuracy >= 1.0),
        Gate(
            "citation hit rate",
            summary.citation_hit_rate,
            1.0,
            summary.citation_hit_rate >= 1.0,
            "expected citations retrieved",
        ),
        Gate(
            "known failures",
            float(summary.known_failures),
            float(len(baseline["known_failures"])),
            summary.known_failures <= len(baseline["known_failures"]),
            "must not grow",
        ),
    ]
    # Integrity: the audit chain those requests just extended must still verify
    # from genesis. Testing your own observability is not paranoia -- an audit
    # log nobody verifies is a file, not a control.
    verification = agent.audit.verify()
    gates.append(
        Gate(
            "audit chain verifies",
            float(verification.verified),
            float(verification.records),
            verification.ok,
            "from genesis"
            if verification.ok
            else f"broken at record {verification.broken_at}",
        )
    )
    return GateReport(
        index_version=agent.index_version,
        gates=gates,
        summary=summary,
        gate_model=gate_model,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@operator_error_exit
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the golden set and report quality.")
    parser.add_argument("--evals-dir", type=Path, default=EVALS_DIR)
    parser.add_argument("--limit", type=int, default=None, help="only run the first N rows")
    parser.add_argument(
        "--phoenix", action="store_true", help="also upload a dataset and run an experiment"
    )
    parser.add_argument("--calibration", action="store_true", help="only report judge calibration")
    parser.add_argument("--gate", action="store_true", help="report the CI gate's numbers")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    configure_logging("WARNING" if args.quiet else None)

    if args.calibration:
        # No agent on this path, so read the embedder from the manifest directly.
        manifest = load_manifest(load_settings())
        embedder = manifest.embedding_model if manifest else load_settings().embedder
        datasets = require_datasets_dir(args.evals_dir, embedder)
        report = calibration_report(rows=read_jsonl(datasets / "calibration.jsonl"))
        print(json.dumps(report.to_dict(), indent=2) if args.json else report.render())
        return 0

    settings = load_settings()
    agent = Agent(settings)
    if args.gate:
        # Pinned, like evals/conftest.py and demo/jobs.py. Reported, not silent:
        # a number graded by a different model than the one serving is a number
        # you have to know the provenance of.
        gate_agent = agent if settings.model == GATE_MODEL else Agent(
            settings.with_overrides(model=GATE_MODEL)
        )
        report = gate_report(gate_agent, evals_dir=args.evals_dir)
        print(json.dumps(report.to_dict(), indent=2) if args.json else report.render())
        return 0 if report.passed else 1
    golden = read_jsonl(_datasets_for(agent, args.evals_dir) / "golden.jsonl")
    summary = run_experiment_locally(agent, golden, limit=args.limit)
    if args.phoenix:
        summary.phoenix = push_to_phoenix(agent, golden, settings, summary.results)
    print(json.dumps(summary.to_dict(), indent=2) if args.json else summary.render())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
