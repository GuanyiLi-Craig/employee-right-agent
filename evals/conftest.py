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


# --------------------------------------------------------------------------- #
# Session 6 — the adversarial gate.
#
# Separate labs, separate artefacts, and separate skips from the session 5
# fixtures above: hanging the adversarial suite off the golden-set fixtures
# would make one suite's missing index look like the other suite's failure.
# --------------------------------------------------------------------------- #
# `evals` is not a package, and conftest.py is loaded by path with its own
# directory on sys.path -- so this is a plain module import, not a
# package-qualified one. `from evals.attacklab_gate import ...` fails with
# ModuleNotFoundError.
from attacklab_gate import MISSING_TWINS, read_thresholds  # noqa: E402


@pytest.fixture(scope="session")
def gate_thresholds() -> dict[str, Any]:
    """``evals/thresholds.json``.

    Thresholds are data, not code, and every one of them carries a note saying
    where the number came from. A gate whose numbers nobody can source is a gate
    nobody can argue with, and the first time it goes red someone loosens it
    rather than investigating.
    """
    return read_thresholds()


@pytest.fixture(scope="session")
def lab(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    """A lab on the poisoned index, writing its artefacts to a temp directory.

    Temp artefacts for the same reason the session 5 agent gets a temporary
    audit log: a gate whose result depends on whether somebody ran an attack
    this morning is not a gate.

    The layer-3 judge is **left on**, and that is deliberate after getting it
    wrong: it was disabled first, on the reasoning that a gate must not depend
    on a provider's availability. But check 4 runs offline here -- it is priced
    against the real price list and answered by a deterministic simulation, with
    no network call -- so turning it off did not make the gate independent, it
    made the gate measure a different system.

    And it measured it misleadingly. ``p07``'s only container at layer 3 is
    check 4, so with the judge off the suite reported the polite rephrase as
    getting past layers 1 *and* 3 -- which is a real finding
    (``test_only_the_judge_catches_the_polite_rephrase`` asserts it directly)
    but not what ``expect_blocked_by`` describes.
    """
    from attacklab.attacks.corpus import twin_built
    from attacklab.lab import Lab
    from attacklab.registry import Registry
    from rights_agent.config import reload_settings

    # Re-read the environment first.
    #
    # `tests/conftest.py::isolated_settings` points RIGHTS_RUNS_DIR at a tmp
    # directory and calls `reload_settings()` in its teardown -- which runs
    # *before* monkeypatch restores the variable, so the process-wide cache is
    # left holding the tmp path for the rest of the session. The unit tests run
    # first (`testpaths = ["tests", "evals"]`), so by the time these fixtures
    # ask where the indexes are, the answer is a directory pytest deleted.
    #
    # The symptom was eleven adversarial tests skipping with "the indexes have
    # not been built" in a full run and passing when the file was run alone,
    # which is the most misleading shape a skip can have.
    reload_settings()
    if not twin_built(True):
        pytest.skip(MISSING_TWINS)
    made = Lab(
        Registry(),
        poisoned=True,
        artefacts=tmp_path_factory.mktemp("attacklab-poisoned"),
    )
    yield made
    made.audit.clear()


@pytest.fixture(scope="session")
def clean_lab(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    """A lab on the clean index.  The false-positive rate is measured here.

    A false-positive rate is by definition measured on inputs where nothing is
    wrong. Measured against the poisoned index, an ordinary question about shift
    notice retrieves the document ``p05`` lives in, the scanner correctly flags
    it, and the suite records a false positive against a perfectly reasonable
    question -- which is how a 4% rate reads as 100%.
    """
    from attacklab.attacks.corpus import twin_built
    from attacklab.lab import Lab
    from attacklab.registry import Registry
    from rights_agent.config import reload_settings

    # See the note in the `lab` fixture: the settings cache can be left holding
    # a deleted tmp directory by the unit suite's teardown order.
    reload_settings()
    if not twin_built(False):
        pytest.skip(MISSING_TWINS)
    made = Lab(
        Registry(),
        poisoned=False,
        artefacts=tmp_path_factory.mktemp("attacklab-clean"),
    )
    yield made
    made.audit.clear()
