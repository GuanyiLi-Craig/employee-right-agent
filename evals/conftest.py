"""Shared fixtures for both eval suites.

The suites import the *same* modules the demo runner does.  There is no
demo-only code path, so what the room sees is what these tests assert.

The golden set is run **once** per session and shared: 37 requests against a
local index is a few seconds, but re-running it per test would make the suite
slow enough that people stop running it -- and a suite nobody runs is worse than
no suite.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from rights_agent.agent import Agent, AgentAnswer
from rights_agent.audit import AuditLog
from rights_agent.config import Settings, reload_settings
from rights_agent.document.parser import parse_corpus
from rights_agent.metrics import MetricsSink
from rights_agent.retrieval import Retriever
from rights_agent.store import IndexNotBuiltError, load_manifest

EVALS_DIR = Path(__file__).parent


# Resolution lives in the package, not here, so the dashboard's jobs, the CLI's
# gate report and this suite cannot disagree about where a dataset is.
from rights_agent.datasets import require_datasets_dir  # noqa: E402

MISSING_INDEX_HINT = (
    "no index found. The embedding pipeline is a separate step:\n"
    "    uv run rights-ingest --no-onnx\n"
    "or, with Docker:\n"
    "    docker compose run --rm ingest"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        pytest.fail(
            f"{path} is missing. Generate the eval datasets with:\n"
            "    uv run python -m rights_agent goldens --write-baseline"
        )
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            pytest.fail(f"{path}:{number} is not valid JSON: {exc}")
    return rows


#: The gate always evaluates the offline stub.
#:
#: Not a convenience: a merge gate whose result depends on a hosted provider's
#: availability, key or price list is not a gate. With ``RIGHTS_MODEL`` pointing
#: at a hosted model, this suite would pass or fail according to whether a key
#: happened to be present in the environment -- which is how a green build stops
#: meaning anything. Evaluate a hosted model deliberately and separately:
#:
#:     RIGHTS_MODEL=deepseek-v4-flash uv run python -m rights_agent evaluate
# One definition, in the gate's own module. Three call sites pinned this
# independently and the CLI's had drifted to no pin at all.
from rights_agent.tools.evaluate import GATE_MODEL  # noqa: E402


@pytest.fixture(scope="session")
def settings() -> Settings:
    return reload_settings().with_overrides(model=GATE_MODEL)


@pytest.fixture(scope="session")
def manifest(settings: Settings):
    found = load_manifest(settings)
    if found is None:
        pytest.fail(MISSING_INDEX_HINT)
    return found


@pytest.fixture(scope="session")
def retriever(settings: Settings) -> Retriever:
    try:
        return Retriever(settings)
    except IndexNotBuiltError:
        pytest.fail(MISSING_INDEX_HINT)


@pytest.fixture(scope="session")
def tree(settings: Settings):
    return parse_corpus(settings.corpus_path).tree


@pytest.fixture(scope="session")
def dataset_dir(manifest) -> Path:
    """The datasets belong to one retrieval config, so they are stored under it.

    Which rows are ``known_failure``, and what a realistic quality floor is, are
    both properties of the embedder: the same 30 questions retrieve their
    expected citation 83% of the time on the hashing bag-of-words, 90% on MiniLM
    and 100% on ``text-embedding-3-small``.  One shared golden set would be
    wrong for at least two of the three, and wrong in the direction that looks
    like a regression in whichever one did not generate it.
    """
    from rights_agent.datasets import DatasetsMissingError

    try:
        return require_datasets_dir(EVALS_DIR, manifest.embedding_model)
    except DatasetsMissingError as exc:
        pytest.fail(str(exc))


@pytest.fixture(scope="session")
def golden_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    return _read_jsonl(dataset_dir / "golden.jsonl")


@pytest.fixture(scope="session")
def calibration_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    return _read_jsonl(dataset_dir / "calibration.jsonl")


@pytest.fixture(scope="session")
def baseline(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "baseline.json"
    if not path.exists():
        pytest.fail(
            f"{path} is missing. It records which golden rows are known to fail and the "
            "quality gates. Create it with:\n"
            "    python -m rights_agent goldens --write-baseline"
        )
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class GoldenResult:
    """A golden row paired with what the agent actually did with it."""

    row: dict[str, Any]
    answer: AgentAnswer

    @property
    def id(self) -> str:
        return str(self.row.get("id", ""))

    @property
    def must_cite(self) -> list[str]:
        return list(self.row.get("must_cite") or [])

    @property
    def should_refuse(self) -> bool:
        return bool(self.row.get("should_refuse"))

    @property
    def known_failure(self) -> bool:
        return bool(self.row.get("known_failure"))

    def retrieved_citations(self) -> set[str]:
        return {str(doc.get("citation", "")) for doc in self.answer.docs}

    def retrieved_provisions(self) -> set[str]:
        """Citations reduced to their provision, so ``s.7(2)`` satisfies ``s.7``."""
        return {citation.split("(")[0].strip() for citation in self.retrieved_citations()}


@pytest.fixture(scope="session")
def eval_agent(settings: Settings, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Agent]:
    """An agent whose metrics and audit records go to temporary files.

    Both are still written and read back -- the sink and the chain are part of
    what is under test -- they just do not touch the operator's ``runs/``. The
    audit log especially: a gate whose result depends on whether someone ran the
    tamper demonstration this morning is not a gate.
    """
    artefacts = tmp_path_factory.mktemp("artefacts")
    try:
        agent = Agent(
            settings,
            sink=MetricsSink(artefacts / "metrics.jsonl"),
            audit=AuditLog(artefacts / "audit.jsonl"),
            init_tracing=False,
        )
    except IndexNotBuiltError:
        pytest.fail(MISSING_INDEX_HINT)
    yield agent
    agent.sink.clear()
    agent.audit.clear()


@pytest.fixture(scope="session")
def golden_results(
    eval_agent: Agent, golden_rows: Sequence[dict[str, Any]]
) -> list[GoldenResult]:
    """Run every golden row once, in a fresh session per row."""
    results: list[GoldenResult] = []
    for row in golden_rows:
        answer = eval_agent.ask(
            str(row["question"]),
            session_id=f"eval-{row['id']}",
            user_id="eval",
            remember=False,
        )
        results.append(GoldenResult(row=row, answer=answer))
    return results


@pytest.fixture(scope="session")
def answerable_results(golden_results: Sequence[GoldenResult]) -> list[GoldenResult]:
    """Rows that are expected to produce a cited answer."""
    return [
        result
        for result in golden_results
        if not result.should_refuse and not result.known_failure
    ]
