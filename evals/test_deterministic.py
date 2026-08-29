"""The suite that blocks the merge.

Structural assertions only.  **Nothing here asks a model for an opinion --
which is exactly why it is allowed to fail a build.**  Every assertion in this
file has the same answer on every machine, every run, with the network off.

Aggregate quality thresholds live in ``test_quality.py``; the separation is
deliberate.  Mixing a flaky judged metric into a blocking gate produces a suite
people learn to re-run until it goes green, and a gate nobody trusts is not a
gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from rights_agent.config import LEAF_COLLECTION, PARENT_COLLECTION, SIMPLE_COLLECTION, Settings
from rights_agent.document.nodes import (
    KIND_INSERTED,
    KIND_SUBSECTION,
    Node,
    citation_resolves,
    leaves as tree_leaves,
    stats,
)
from rights_agent.document.parser import TreeExpectations, validate_tree
from rights_agent.embedding import EmbedderError, HashingEmbedder, assert_embedder_matches
from rights_agent.llm import parse_context
from rights_agent.retrieval import Doc, Retriever, format_context, sufficiency
from rights_agent.store import (
    IndexManifest,
    StoreError,
    chroma_client,
    collection_count,
    open_collection,
)

from conftest import GoldenResult


# --------------------------------------------------------------------------- #
# The dataset and the index have to be talking about the same document
# --------------------------------------------------------------------------- #
def test_the_golden_set_was_generated_for_this_index(
    manifest: IndexManifest, baseline: dict[str, Any]
) -> None:
    """One legible failure instead of twenty misleading ones.

    Expected citations name provisions, and a provision exists in exactly one
    document.  Point the gate at a different corpus and every row fails on a
    citation that was never in the index -- which reads as a retrieval
    regression, and sends whoever is on the hook for it into the retriever.
    Change the embedder and a subtler version happens: the rows still exist, but
    which of them are ``known_failure`` is no longer true.
    """
    stamp = baseline.get("generated_for")
    assert stamp, (
        "evals/baseline.json has no 'generated_for' stamp: regenerate the "
        "datasets with `python -m rights_agent goldens --write-baseline` so the "
        "gate can tell which index they belong to"
    )
    regenerate = "`python -m rights_agent goldens --write-baseline`"
    assert stamp["corpus_sha8"] == manifest.corpus_sha[:8], (
        f"the datasets were generated for {stamp['corpus']} "
        f"({stamp['corpus_sha8']}) but the index holds "
        f"{Path(manifest.corpus_path).name} ({manifest.corpus_sha[:8]}). "
        f"Either point RIGHTS_CORPUS back at the corpus the datasets describe, "
        f"or regenerate them for this one with {regenerate}."
    )
    # The embedder decides which provisions a question retrieves, so it decides
    # which rows are ``known_failure`` -- three of them start passing on MiniLM
    # that do not on the hashing bag-of-words. Checked separately from the corpus
    # because the two mismatches have different fixes and the message should say
    # which one happened.
    assert stamp["index_version"] == manifest.index_version, (
        f"the datasets were generated against {stamp['index_version']} but the "
        f"index is {manifest.index_version}. The known_failure list is a property "
        f"of the retrieval config, not of the corpus alone. Either set "
        f"RIGHTS_EMBEDDER back to match, or regenerate with {regenerate}."
    )


# --------------------------------------------------------------------------- #
# The index
# --------------------------------------------------------------------------- #
def test_manifest_is_complete(manifest: IndexManifest) -> None:
    assert manifest.index_version, "index_version must be recorded"
    assert manifest.index_version.count("+") == 2, (
        f"index_version {manifest.index_version!r} should be "
        "<parser>+<embedder>+<corpus hash>"
    )
    assert manifest.embedding_model
    assert manifest.parser_version
    assert manifest.corpus_sha and len(manifest.corpus_sha) >= 8
    assert manifest.build_seconds >= 0
    assert manifest.collections.get(LEAF_COLLECTION, 0) > 0
    assert manifest.collections.get(PARENT_COLLECTION, 0) > 0


def test_leaf_count_is_sane(manifest: IndexManifest, tree: Node) -> None:
    """Leaf rows should track the subsection count within 10%."""
    subsections = stats(tree).get(KIND_SUBSECTION, 0)
    leaves = manifest.collections[LEAF_COLLECTION]
    assert leaves == len(tree_leaves(tree))
    assert abs(leaves - subsections) <= 0.10 * subsections, (
        f"{leaves} leaves against {subsections} subsections is more than 10% apart"
    )


def test_collections_match_the_manifest(settings: Settings, manifest: IndexManifest) -> None:
    client = chroma_client(settings)
    for name, expected in manifest.collections.items():
        assert collection_count(client, name) == expected


def test_retriever_embedder_matches_the_manifest(
    retriever: Retriever, manifest: IndexManifest
) -> None:
    assert retriever.embedder_name == manifest.embedding_model
    assert retriever.index_version == manifest.index_version


def test_embedder_mismatch_is_fatal() -> None:
    """Pitfall 1: querying across embedders returns confident nonsense, silently."""
    with pytest.raises(EmbedderError, match="confident nonsense"):
        assert_embedder_matches("onnx-all-MiniLM-L6-v2", "hashing-bow-512")
    assert_embedder_matches("hashing-bow-512", "hashing-bow-512")


class _MislabelledEmbedder(HashingEmbedder):
    """Same vectors, different name -- stands in for the wrong model entirely.

    Chroma 1.5 will open a collection with whatever embedder it is handed and
    say nothing, so this asserts *our* guard, not Chroma's.
    """

    @staticmethod
    def name() -> str:
        return "not-the-index-embedder"


def test_opening_a_collection_with_the_wrong_embedder_is_refused(settings: Settings) -> None:
    client = chroma_client(settings)
    with pytest.raises(StoreError, match="confident nonsense"):
        open_collection(client, LEAF_COLLECTION, _MislabelledEmbedder())


# --------------------------------------------------------------------------- #
# The tree
# --------------------------------------------------------------------------- #
def test_tree_shape_gate(tree: Node) -> None:
    counts = validate_tree(tree, TreeExpectations())
    assert counts["part"] >= 5
    assert counts["section"] >= 150
    assert counts["subsection"] >= 900


def test_every_leaf_has_a_resolvable_breadcrumb_and_citation(tree: Node) -> None:
    for node in tree_leaves(tree):
        breadcrumb = node.breadcrumb()
        assert ">" in breadcrumb, f"{node.citation()} has no ancestors in its breadcrumb"
        assert breadcrumb.split(" > ")[0], "breadcrumb must start at the document"
        citation = node.citation()
        assert citation and citation != node.kind, f"{breadcrumb} produced no citation"


def test_inserted_provisions_are_not_attributed_to_their_host(tree: Node) -> None:
    """Pitfall 7: attributing an inserted provision to the host section makes
    every citation point at the wrong document."""
    inserted = [node for node in tree.walk() if node.kind == KIND_INSERTED]
    assert inserted, "the corpus is expected to contain inserted provisions"
    for node in inserted:
        citation = node.citation()
        assert "as inserted by" in citation, citation
        assert node.host_document, f"{citation} does not name the host document"
        assert node.inserted_by, f"{citation} does not name the inserting section"


def test_leaf_ids_are_unique_and_ordinal_prefixed(
    settings: Settings, manifest: IndexManifest
) -> None:
    """Pitfall 8: citation plus page is not unique, and Chroma ignores duplicates."""
    client = chroma_client(settings)
    collection = client.get_collection(LEAF_COLLECTION)
    ids = collection.get(include=[])["ids"]
    assert len(ids) == len(set(ids)) == manifest.collections[LEAF_COLLECTION]
    assert all(identifier.startswith("l") and "::" in identifier for identifier in ids)


def test_every_leaf_embeds_its_breadcrumb_first(settings: Settings) -> None:
    """The breadcrumb IS the embedded text, not metadata stored beside it."""
    client = chroma_client(settings)
    collection = client.get_collection(LEAF_COLLECTION)
    sample = collection.get(limit=200, include=["documents", "metadatas"])
    assert sample["documents"]
    for document, metadata in zip(sample["documents"], sample["metadatas"]):
        breadcrumb = str(metadata["breadcrumb"])
        assert document.startswith(breadcrumb), (
            f"embedded text for {metadata['citation']} does not start with its breadcrumb"
        )


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_rebuilding_produces_identical_ids(settings: Settings, tmp_path: Path) -> None:
    """Two consecutive ingests must produce identical chunk ids (§9.5)."""
    from rights_agent.document.parser import parse_corpus
    from rights_agent.pipelines.hierarchical import build_rows

    first = build_rows(parse_corpus(settings.corpus_path), "v1")
    second = build_rows(parse_corpus(settings.corpus_path), "v1")
    assert first[0]["ids"] == second[0]["ids"]
    assert first[1]["ids"] == second[1]["ids"]
    assert first[0]["documents"] == second[0]["documents"]


def test_hashing_embedder_is_stable_across_instances() -> None:
    """``hash()`` is salted per process; this embedder must not be."""
    left = HashingEmbedder().embed_documents(["guaranteed hours offer"])
    right = HashingEmbedder().embed_documents(["guaranteed hours offer"])
    assert [list(map(float, row)) for row in left] == [list(map(float, row)) for row in right]


def test_intent_classification_is_stable(golden_rows: Sequence[dict[str, Any]]) -> None:
    """The recorded intent is the classifier's own output; drift shows up here."""
    from rights_agent.graph import classify_intent

    for row in golden_rows:
        assert classify_intent(str(row["question"])) == row["intent"], (
            f"intent for {row['id']} changed; regenerate the golden set deliberately"
        )


# --------------------------------------------------------------------------- #
# Retrieval behaviour
# --------------------------------------------------------------------------- #
def test_context_is_never_empty_for_non_empty_docs(retriever: Retriever) -> None:
    """Pitfall 3: truncate an oversized block, never skip it."""
    docs = retriever.search("What does the document say about bereavement leave?")
    assert docs
    for budget in (500, 1_000, 6_000):
        context = format_context(docs, budget_chars=budget, min_block_chars=200)
        assert context.strip(), f"empty context at budget {budget}"
        assert len(context) <= budget * 1.2, "assembled context overshot its budget"


def test_oversized_single_block_is_truncated_not_dropped() -> None:
    giant = Doc(
        id="l1",
        citation="s.1(1)",
        breadcrumb="Doc > Part 1 > s.1",
        text="x" * 36_000,
        score=0.9,
    )
    context = format_context([giant], budget_chars=2_000, min_block_chars=400)
    assert context, "a single oversized block produced an empty context"
    assert "s.1(1)" in context
    assert "truncated" in context


def test_expansion_respects_the_parent_cap(retriever: Retriever, settings: Settings) -> None:
    """Pitfall 4: one provision must not eat the whole budget."""
    docs = retriever.search("What does the document say about enforcement notices?")
    for doc in docs:
        if doc.expanded:
            assert len(doc.text) <= settings.max_parent_chars
        if doc.metadata.get("expand_skipped"):
            assert doc.metadata.get("parent_chars", 0) > settings.max_parent_chars


def test_sufficiency_scores_the_original_question(retriever: Retriever) -> None:
    """Pitfall 11: refinement must not be able to hide its own failure."""
    original = "Which football team won the league in 1998?"
    docs = retriever.search(original)
    rewritten = "employment rights guaranteed hours worker employer"
    assert sufficiency(docs, original) < sufficiency(docs, rewritten), (
        "scoring the rewritten query would let refinement mask an off-topic question"
    )


def test_simple_pipeline_produces_no_citations(settings: Settings) -> None:
    """§8.2: confirm this -- it is the point of the baseline."""
    client = chroma_client(settings)
    if collection_count(client, SIMPLE_COLLECTION) == 0:
        pytest.skip(
            "baseline index not built (uv run rights-ingest-simple); "
            "it is a separate, optional pipeline"
        )
    collection = client.get_collection(SIMPLE_COLLECTION)
    sample = collection.get(limit=50, include=["metadatas"])
    for metadata in sample["metadatas"]:
        assert set(metadata) <= {"offset", "chars", "pipeline", "index_version"}
        assert "citation" not in metadata


# --------------------------------------------------------------------------- #
# Agent behaviour over the golden set
# --------------------------------------------------------------------------- #
def test_out_of_scope_questions_are_refused(golden_results: Sequence[GoldenResult]) -> None:
    failures: list[str] = []
    for result in golden_results:
        if not result.should_refuse:
            continue
        if not result.answer.refused:
            failures.append(
                f"{result.id}: answered {result.row['question']!r} "
                f"(sufficiency {result.answer.sufficiency:.3f})"
            )
    assert not failures, "out-of-scope questions were answered:\n  " + "\n  ".join(failures)


def test_refusals_state_their_score_and_threshold(
    golden_results: Sequence[GoldenResult], settings: Settings
) -> None:
    for result in golden_results:
        if not result.answer.refused:
            continue
        answer = result.answer.answer
        assert f"{result.answer.sufficiency:.2f}" in answer, result.id
        assert f"{settings.sufficiency_threshold:.2f}" in answer, result.id
        assert not result.answer.citations, "a refusal must not carry citations"


def test_expected_citations_are_retrieved(answerable_results: Sequence[GoldenResult]) -> None:
    failures: list[str] = []
    for result in answerable_results:
        provisions = result.retrieved_provisions()
        for expected in result.must_cite:
            if expected.split("(")[0] not in provisions:
                failures.append(
                    f"{result.id}: expected {expected} for {result.row['question']!r}, "
                    f"retrieved {sorted(provisions)[:5]}"
                )
    assert not failures, "expected citations were not retrieved:\n  " + "\n  ".join(failures)


def test_every_answer_cites_or_refuses(golden_results: Sequence[GoldenResult]) -> None:
    """Every answer carries a citation, or is a refusal that says why."""
    failures: list[str] = []
    for result in golden_results:
        answer = result.answer
        if answer.refused:
            continue
        if not answer.citations:
            failures.append(f"{result.id}: answered with no citation")
            continue
        available = [block.citation for block in parse_context(answer.context)]
        unresolvable = [
            c for c in answer.citations if not citation_resolves(c, available)
        ]
        if unresolvable:
            failures.append(f"{result.id}: cited material not in its context: {unresolvable}")
    assert not failures, "\n  ".join(["uncited or misattributed answers:"] + failures)


def test_known_failures_have_not_grown(
    golden_rows: Sequence[dict[str, Any]], baseline: dict[str, Any]
) -> None:
    recorded = {str(row["id"]) for row in golden_rows if row.get("known_failure")}
    expected = set(baseline["known_failures"])
    new = sorted(recorded - expected)
    assert not new, (
        f"new known failures {new}: fix them, or record them deliberately with "
        "`uv run python -m rights_agent goldens --write-baseline`"
    )


def test_known_failures_have_not_silently_started_passing(
    golden_results: Sequence[GoldenResult], baseline: dict[str, Any]
) -> None:
    """A known failure that starts passing must have its marker removed."""
    expected = set(baseline["known_failures"])
    now_passing: list[str] = []
    for result in golden_results:
        if result.id not in expected or not result.must_cite:
            continue
        provisions = result.retrieved_provisions()
        if all(citation.split("(")[0] in provisions for citation in result.must_cite):
            now_passing.append(result.id)
    assert not now_passing, (
        f"{now_passing} now pass: remove the known_failure marker so the gate keeps "
        "protecting them"
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_every_request_records_the_required_fields(
    golden_results: Sequence[GoldenResult],
) -> None:
    for result in golden_results:
        metrics = result.answer.metrics
        assert metrics.request_id
        assert metrics.session_id
        assert metrics.ts
        assert metrics.index_version, "index_version must be on every row"
        assert metrics.embedding_model
        assert metrics.requested_model, "what was asked for is always known"
        assert metrics.prompt_version
        assert metrics.pricing_as_of
        if not metrics.refused:
            # A refusal never reaches generate, so no model served it -- and
            # naming one would be a guess.
            assert metrics.model, "the model that answered must be recorded"
            assert metrics.prompt_tokens > 0
            assert metrics.completion_tokens > 0
            assert metrics.cost_usd > 0
            assert metrics.ttft_ms > 0, "TTFT must be measured, not estimated"


def test_e2e_is_at_least_the_sum_of_timed_stages(
    golden_results: Sequence[GoldenResult],
) -> None:
    for result in golden_results:
        metrics = result.answer.metrics
        assert metrics.stage_ms, "stage timings must be recorded"
        assert metrics.e2e_ms + 1e-6 >= metrics.stage_total_ms(), (
            f"{result.id}: e2e {metrics.e2e_ms} < stages {metrics.stage_total_ms()}"
        )
        assert metrics.orchestration_ms() >= 0


def test_metrics_rows_round_trip_through_the_sink(eval_agent, golden_results) -> None:
    rows = eval_agent.sink.read()
    assert len(rows) >= len(golden_results)
    for row in rows:
        json.dumps(row)  # must stay JSON-serialisable
        assert row["index_version"]
        assert "context" not in row, "the retrieved context belongs in the trace, not here"


def test_context_is_not_duplicated_into_metrics(golden_results: Sequence[GoldenResult]) -> None:
    for result in golden_results:
        payload = json.loads(result.answer.metrics.to_json())
        assert "context" not in payload
        assert "docs" not in payload


# --------------------------------------------------------------------------- #
# The invocation trap
# --------------------------------------------------------------------------- #
def test_each_request_gets_a_fresh_thread(eval_agent) -> None:
    """Pitfall 2: a reused ``thread_id`` makes everything refuse after a few queries."""
    question = "What does the document say about bereavement leave?"
    seen: list[int] = []
    for _ in range(4):
        answer = eval_agent.ask(question, session_id="thread-trap")
        seen.append(answer.attempts)
        assert not answer.refused, "state leaked between requests in the same session"
    assert seen == [0, 0, 0, 0], f"attempts accumulated across requests: {seen}"


# --------------------------------------------------------------------------- #
# The dashboard imports the same modules as these tests
# --------------------------------------------------------------------------- #
def _quiet(settings: Settings) -> Settings:
    """Same settings, no span export: these assertions are about the panels."""
    return settings.with_overrides(tracing_enabled=False)


def _service(settings: Settings, tmp_path: Path):
    """A dashboard whose metrics and audit records land in ``tmp_path``.

    The index is shared -- that is what is under test -- but the writes are not:
    asserting on the dashboard must not append to the operator's audit chain.
    """
    from rights_agent.agent import Agent
    from rights_agent.audit import AuditLog
    from rights_agent.demo.app import DemoService
    from rights_agent.metrics import MetricsSink

    sink = MetricsSink(tmp_path / "metrics.jsonl")
    agent = Agent(
        _quiet(settings),
        sink=sink,
        audit=AuditLog(tmp_path / "audit.jsonl"),
        init_tracing=False,
    )
    return DemoService(_quiet(settings), agent=agent, sink=sink)


def test_the_index_panel_reads_the_index_it_claims_to(
    settings: Settings, tmp_path: Path, manifest: IndexManifest
) -> None:
    """The panel exists to make "we embedded the document" inspectable, so the
    thing it shows has to be the index actually being queried."""
    service = _service(settings, tmp_path)
    payload = service.chunks(limit=5)

    assert payload["total"] == manifest.collections[LEAF_COLLECTION]
    assert payload["embedder"] == manifest.embedding_model
    assert len(payload["chunks"]) == 5
    for chunk in payload["chunks"]:
        assert chunk["index_version"] == manifest.index_version, (
            "a chunk in the index disagrees with the manifest about which index it is"
        )
        assert chunk["citation"], chunk["id"]


def test_the_embedded_text_starts_with_the_breadcrumb(
    settings: Settings, tmp_path: Path
) -> None:
    """The claim the panel is there to make visible: the citation is *inside*
    the vector, not stored beside it."""
    service = _service(settings, tmp_path)
    for chunk in service.chunks(limit=10)["chunks"]:
        assert chunk["embedded_text"].startswith(chunk["breadcrumb"]), chunk["id"]
        assert chunk["raw_text"] in chunk["embedded_text"], chunk["id"]


def test_searching_the_panel_runs_the_agent_s_own_retrieval(
    settings: Settings, tmp_path: Path
) -> None:
    """A panel with its own private search would be a demo prop. Same retriever,
    same scores, descending."""
    service = _service(settings, tmp_path)
    payload = service.chunks(query="bereavement leave", limit=5)

    scores = [chunk["score"] for chunk in payload["chunks"]]
    assert all(score is not None for score in scores), "search results carry no score"
    assert scores == sorted(scores, reverse=True), "results are not ranked"
    assert payload["query"] == "bereavement leave"


def test_a_vector_is_returned_whole_with_its_summary(
    settings: Settings, tmp_path: Path, manifest: IndexManifest
) -> None:
    """A demo that shows eight of 1,536 numbers and calls it "the embedding" is
    doing the hand-waving this panel exists to stop."""
    service = _service(settings, tmp_path)
    chunk = service.chunks(limit=1)["chunks"][0]

    vector = service.chunks(limit=1, with_vector=chunk["id"])["vector"]

    assert vector["dimensions"] == len(vector["values"]) > 0
    assert len(vector["preview"]) == 16
    assert vector["preview"] == vector["values"][:16]
    # Cosine distance assumes unit vectors. Reported rather than asserted in the
    # panel; asserted here, because an embedder that stops normalising is a
    # silent change in what every distance means.
    assert vector["norm"] == pytest.approx(1.0, abs=0.01), vector["norm"]


def test_an_unknown_chunk_id_reports_rather_than_raises(
    settings: Settings, tmp_path: Path
) -> None:
    service = _service(settings, tmp_path)
    assert "error" in service.chunks(limit=1, with_vector="no-such-chunk")["vector"]


def test_the_warm_up_question_reaches_the_model(settings: Settings, tmp_path: Path) -> None:
    """A refused warm-up warms nothing.

    The first version asked "What does this document cover?", which scores 0.218
    on the sufficiency gate against the real Act and is refused -- so no model
    call was made, no connection was opened, and the startup banner reported
    "0 ms ttft" as though that were a result.
    """
    from rights_agent.demo.app import WARMUP_QUESTION

    service = _service(settings, tmp_path)
    answer = service.agent.ask(WARMUP_QUESTION, session_id="warmup-probe", record=False)

    assert answer.metrics.route == "generate", (
        f"the warm-up question is {answer.metrics.route}d against this index "
        f"(sufficiency {answer.metrics.sufficiency:.3f}); pick one the corpus answers"
    )
    assert answer.citations, "a warm-up that produces no citations did not exercise the path"


def test_the_warm_up_is_not_recorded_anywhere(settings: Settings, tmp_path: Path) -> None:
    """A request nobody made must not appear in a record of what was asked.

    The warm-up exists so the audience does not watch the first request pay for
    TLS setup and an empty prompt cache. If it landed in the metrics rows it
    would also drag the latency panel's first p50, and if it landed in the audit
    chain it would put a question no human asked into the compliance record.
    """
    from rights_agent.demo.app import WARMUP_QUESTION, warm_up

    service = _service(settings, tmp_path)
    result = warm_up(service)

    assert "failed" not in result, result
    assert service.transcript("warmup")["turns"] == []
    listed = [row["session_id"] for row in service.sessions()["sessions"]]
    assert "warmup" not in listed, "the warm-up appeared in the history list"
    rows = service.agent.sink.read()
    assert not [row for row in rows if row.get("question") == WARMUP_QUESTION], (
        "the warm-up wrote a metrics row"
    )
    assert service.agent.audit.read() == [], "the warm-up wrote an audit record"


def test_dashboard_state_exposes_every_panel(settings: Settings, tmp_path: Path) -> None:
    """No demo-only code path: the dashboard's snapshot is built from the same
    metrics rows, judges and manifest the suites assert on."""
    service = _service(settings, tmp_path)
    state = service.state()
    assert set(state) >= {"index", "runtime", "summary", "job", "output", "jobs_available"}
    assert state["index"]["index_version"] == service.agent.index_version
    assert set(state["summary"]) == {"latency", "quality", "cost", "traffic", "window"}
    assert state["summary"]["window"]["size"] == settings.panel_window
    # Percentiles for latency, never a mean.
    assert set(state["summary"]["latency"]["e2e_ms"]) == {"p50", "p95", "p99", "n"}
    # Quality reports the p10 next to the mean.
    assert set(state["summary"]["quality"]["groundedness"]) == {"mean", "p10", "n"}
    assert "monthly_projection_usd" in state["summary"]["cost"]
    assert "projection_assumes_requests_per_day" in state["summary"]["cost"]
    # The five-component cost model, not just the tokens.
    assert set(state["summary"]["cost"]["modelled_components_usd"]) == {
        "generation_input",
        "generation_output",
        "judge",
        "trace_storage",
        "infrastructure",
    }
    # The chat surface and the audit panel.
    assert state["suggestions"], "the chat box offers no suggestions"
    assert set(state["audit"]) == {"enabled", "chain", "retention"}
    assert state["audit"]["retention"]["floor_days"] >= 183


def test_dashboard_exposes_every_documented_control(
    settings: Settings, tmp_path: Path
) -> None:
    service = _service(settings, tmp_path)
    assert set(service.runner.jobs) == {
        "baseline_traffic",
        "incident_traffic",
        "ci_gate",
        "calibrate_judge",
        "shift_intents",
        "drift_report",
        "reprice",
        "verify_audit",
        "tamper_audit",
        "reset",
    }


def test_a_failing_job_reports_into_the_output_log(
    settings: Settings, tmp_path: Path
) -> None:
    """A traceback the room cannot see is a traceback that did not happen."""
    service = _service(settings, tmp_path)
    service.runner.jobs["exploding"] = lambda params: (_ for _ in ()).throw(
        RuntimeError("job blew up")
    )
    accepted, _ = service.runner.start("exploding")
    assert accepted
    thread = service.runner._thread
    assert thread is not None
    thread.join(timeout=10)
    entry = service.runner.snapshot()["output"][0]
    assert entry["job"] == "exploding" and entry["ok"] is False
    assert "job blew up" in entry["text"]


def test_only_one_job_runs_at_a_time(settings: Settings, tmp_path: Path) -> None:
    """The UI polls a snapshot; it must never be able to queue work in parallel
    against a single-writer SQLite index."""
    import threading

    service = _service(settings, tmp_path)
    release = threading.Event()
    service.runner.jobs["waiting"] = lambda params: (release.wait(5), "done")[1]
    assert service.runner.start("waiting")[0] is True
    accepted, message = service.runner.start("waiting")
    assert accepted is False and "still running" in message
    release.set()
    thread = service.runner._thread
    if thread is not None:
        thread.join(timeout=10)


def test_health_endpoint_reports_readiness(settings: Settings, tmp_path: Path) -> None:
    health = _service(settings, tmp_path).health()
    assert health["status"] == "ok"
    assert health["index_version"]


# --------------------------------------------------------------------------- #
# Integrity: the audit chain
# --------------------------------------------------------------------------- #
def test_every_request_appends_one_audit_record(
    eval_agent, golden_results: Sequence[GoldenResult]
) -> None:
    rows = eval_agent.audit.read()
    assert len(rows) >= len(golden_results)
    by_request = {str(row["request_id"]): row for row in rows}
    for result in golden_results:
        record = by_request.get(result.answer.request_id)
        assert record is not None, f"{result.id} has no audit record"
        assert record["index_version"]
        assert record["refused"] == result.answer.refused
        assert record["sequence"] == result.answer.metrics.audit_sequence


def test_the_audit_chain_verifies_from_genesis(eval_agent, golden_results) -> None:
    """Testing your own observability is not paranoia: an audit log nobody
    verifies is a file, not a control."""
    verification = eval_agent.audit.verify()
    assert verification.ok, verification.render()
    assert verification.records == verification.verified


def test_an_edited_record_is_detected_and_located(eval_agent, golden_results) -> None:
    """Demo 7, as an assertion. Restores the store afterwards so the ordering of
    tests cannot matter."""
    original = eval_agent.audit.path.read_text(encoding="utf-8")
    try:
        eval_agent.audit.tamper(0, "question")
        verification = eval_agent.audit.verify()
        assert not verification.ok
        assert verification.broken_at == 0
        assert "CHAIN BROKEN at record 0" in verification.render()
    finally:
        eval_agent.audit.path.write_text(original, encoding="utf-8")
        eval_agent.audit.reload()
    assert eval_agent.audit.verify().ok, "the store was not restored"


def test_audit_records_never_store_source_text(eval_agent, golden_results) -> None:
    for row in eval_agent.audit.read():
        for source in row.get("sources") or []:
            assert set(source) == {"id", "citation", "version", "sha256"}
            assert len(source["sha256"]) == 64


def test_audit_records_are_redacted_at_capture(eval_agent) -> None:
    answer = eval_agent.ask(
        "Does my contract with jo@example.com cover bereavement leave?",
        session_id="redaction-check",
    )
    record = next(
        row
        for row in eval_agent.audit.read()
        if row["request_id"] == answer.request_id
    )
    assert "jo@example.com" not in json.dumps(record)
    assert "[email]" in record["question"]
    assert record["question_sha256"], "identity must survive redaction"


def test_retention_is_configured_at_or_above_the_floor(settings: Settings) -> None:
    from rights_agent.audit import RETENTION_FLOOR_DAYS, RetentionPolicy

    policy = RetentionPolicy(configured_days=settings.retention_days)
    policy.validate()
    assert policy.configured_days >= RETENTION_FLOOR_DAYS


# --------------------------------------------------------------------------- #
# The chat surface
# --------------------------------------------------------------------------- #
def test_streaming_emits_tokens_then_one_answer(
    settings: Settings, tmp_path: Path
) -> None:
    """The first token event is what makes TTFT visible in the UI rather than
    only reported after the fact."""
    service = _service(settings, tmp_path)
    events = list(
        service.ask_streaming(
            "What does the document say about bereavement leave?", "stream-test"
        )
    )
    kinds = [event["type"] for event in events]
    assert kinds[0] == "start"
    assert kinds[-1] == "answer"
    assert kinds.count("answer") == 1
    tokens = [event["text"] for event in events if event["type"] == "token"]
    assert tokens, "nothing was streamed"
    final = events[-1]["answer"]
    assert "".join(tokens).strip() == final["answer"]
    assert final["citations"]


def test_a_follow_up_is_answered_rather_than_refused(
    settings: Settings, tmp_path: Path
) -> None:
    """A chat bot whose every follow-up refuses is not a chat bot.

    The resolved question is what the gate scores here, and that is deliberate:
    those words came from the *user*, one turn earlier. A rewrite the system
    invented for itself is still never scored -- see ``test_refinement_cannot_
    talk_its_way_past_the_gate``.
    """
    service = _service(settings, tmp_path)
    first = service.ask("What does the document say about bereavement leave?", "follow-up")
    assert not first["refused"]
    second = service.ask("How long is it?", "follow-up")
    assert second["history_used"], "the follow-up was not resolved against the session"
    assert not second["refused"], "a resolved follow-up must not refuse"
    assert second["citations"]


def test_refinement_cannot_talk_its_way_past_the_gate(settings: Settings) -> None:
    """The rule that still holds: a query the *system* invented is never scored."""
    from rights_agent.graph import AgentDeps, build_graph, initial_state

    graph = build_graph(AgentDeps(settings=_quiet(settings)))
    state = initial_state("How do I mine cryptocurrency on company laptops?")
    final = graph.invoke(state, config={"configurable": {"thread_id": "refine-gate"}})
    assert final["refused"], "refinement rescued an out-of-scope question"
    assert final["attempts"] == settings.max_attempts
    assert final["rewritten_query"] != final["question"], "refinement did run"
    assert final["scored_question"] == final["question"]


def test_a_follow_up_with_no_session_history_is_unchanged(
    settings: Settings, tmp_path: Path
) -> None:
    service = _service(settings, tmp_path)
    answer = service.ask("How long is it?", "no-history")
    assert not answer["history_used"]


def test_the_transcript_records_both_sides(settings: Settings, tmp_path: Path) -> None:
    service = _service(settings, tmp_path)
    service.ask("What does the document say about penalty notices?", "transcript-test")
    history = service.transcript("transcript-test")
    assert history["source"] == "live"
    assert [turn["role"] for turn in history["turns"]] == ["user", "agent"]
    assert history["turns"][1]["citations"]
    assert not history["turns"][1]["reconstructed"]


def test_cost_components_are_recorded_on_every_row(
    golden_results: Sequence[GoldenResult],
) -> None:
    from rights_agent.costs import COMPONENTS

    for result in golden_results:
        metrics = result.answer.metrics
        assert set(metrics.cost_components) == set(COMPONENTS), result.id
        assert metrics.cost_total_usd >= metrics.cost_usd
        assert metrics.trace_bytes > 0


# --------------------------------------------------------------------------- #
# Chat history, across both tiers
# --------------------------------------------------------------------------- #
def test_the_history_list_shows_chat_conversations(
    settings: Settings, tmp_path: Path
) -> None:
    service = _service(settings, tmp_path)
    service.ask("What does the document say about bereavement leave?", "hist-a")
    service.ask("How long is it?", "hist-a")
    service.ask("How are tips allocated between workers?", "hist-b")

    listed = {s["session_id"]: s for s in service.sessions()["sessions"]}
    assert set(listed) == {"hist-a", "hist-b"}
    assert listed["hist-a"]["requests"] == 2
    assert listed["hist-a"]["turns"] == 4
    assert listed["hist-a"]["source"] == "live"
    assert listed["hist-a"]["cost_usd"] > 0
    assert listed["hist-a"]["title"].startswith("What does the document say")


def test_synthetic_traffic_is_audited_but_is_not_a_conversation(
    settings: Settings, tmp_path: Path
) -> None:
    """A history list swamped by generated load is a list nobody reads -- and a
    bounded transcript store filled with it evicts the real conversation."""
    service = _service(settings, tmp_path)
    service.ask("What does the document say about penalty notices?", "mine")
    accepted, _ = service.runner.start("baseline_traffic", {"count": 4})
    assert accepted
    thread = service.runner._thread
    assert thread is not None
    thread.join(timeout=120)

    listed = [s["session_id"] for s in service.sessions()["sessions"]]
    assert listed == ["mine"], f"synthetic traffic leaked into the history: {listed}"
    # It is still audited: the record is of every decision, not every conversation.
    actors = {str(row.get("actor")) for row in service.agent.audit.read()}
    assert {"dashboard", "demo"} <= actors


def test_the_audit_tier_is_not_a_side_door_around_the_token(
    settings: Settings, tmp_path: Path
) -> None:
    """Protecting /api/audit while /api/chat/history reads the same records is
    not a control, it is a side door. Found by a nuclei template that kept
    matching after the live transcript had been cleared."""
    service = _service(settings.with_overrides(demo_token="s3cret"), tmp_path)
    service.ask("What does the document say about penalty notices?", "side-door")
    service.agent.conversations.clear()

    refused = service.transcript("side-door", may_read_audit=False)
    assert refused["source"] == "empty" and refused["turns"] == []
    assert "X-Demo-Token" in refused["note"]

    allowed = service.transcript("side-door", may_read_audit=True)
    assert allowed["source"] == "audit" and allowed["turns"]


def test_history_survives_losing_the_in_memory_transcript(
    settings: Settings, tmp_path: Path
) -> None:
    """Transcripts are in memory on purpose. What survives is the audit record --
    the questions, the citations and the numbers, but not the answer prose."""
    service = _service(settings, tmp_path)
    answered = service.ask("What does the document say about bereavement leave?", "durable")
    service.agent.conversations.clear()  # stand in for a restart

    listed = service.sessions()["sessions"]
    assert [s["source"] for s in listed] == ["audit"]

    history = service.transcript("durable")
    assert history["source"] == "audit"
    assert "not the answers" in history["note"]
    roles = [turn["role"] for turn in history["turns"]]
    assert roles == ["user", "agent"]
    assert all(turn["reconstructed"] for turn in history["turns"])
    assert history["turns"][1]["citations"] == answered["citations"]
    assert answered["answer"] not in history["turns"][1]["content"], (
        "answer prose must not be reconstructible from the audit record"
    )


def test_a_reconstructed_question_is_the_redacted_one(
    settings: Settings, tmp_path: Path
) -> None:
    """History reads the audit record, so it inherits capture-time redaction."""
    service = _service(settings, tmp_path)
    service.ask("Does my contract with jo@example.com cover leave?", "redacted-history")
    service.agent.conversations.clear()
    turns = service.transcript("redacted-history")["turns"]
    assert "jo@example.com" not in turns[0]["content"]
    assert "[email]" in turns[0]["content"]


def test_an_unknown_session_reports_empty_rather_than_failing(
    settings: Settings, tmp_path: Path
) -> None:
    history = _service(settings, tmp_path).transcript("never-existed")
    assert history["source"] == "empty"
    assert history["turns"] == []


def test_history_is_unavailable_not_fatal_when_the_audit_store_is_corrupt(
    settings: Settings, tmp_path: Path
) -> None:
    """The chat must not go down because the audit file is unreadable; the audit
    panel is where that failure is reported."""
    service = _service(settings, tmp_path)
    service.ask("What does the document say about penalty notices?", "corrupt-check")
    service.agent.audit.path.write_text("{not json\n", encoding="utf-8")
    service.agent.audit.reload()

    assert service.sessions()["sessions"] != [], "the live transcript should still list"
    assert service.state()["audit"]["chain"]["ok"] is False
    assert service.ask("And bereavement leave?", "corrupt-check")["answer"]


def test_reopening_a_conversation_restores_follow_up_context(
    settings: Settings, tmp_path: Path
) -> None:
    """After a restart the working transcript is empty, so a follow-up would have
    nothing to borrow. Reopening the conversation puts the questions back."""
    service = _service(settings, tmp_path)
    service.ask("What does the document say about bereavement leave?", "reopen")
    service.agent.conversations.clear()  # stand in for a restart

    orphaned = service.ask("How long is it?", "reopen")
    assert not orphaned["history_used"], "nothing should be borrowable yet"

    service.agent.conversations.clear()
    service.transcript("reopen")  # the UI reopening the conversation
    resumed = service.ask("How long is it?", "reopen")
    assert resumed["history_used"], "reopening did not restore the follow-up context"
    assert not resumed["refused"]


def test_reopening_twice_does_not_duplicate_the_transcript(
    settings: Settings, tmp_path: Path
) -> None:
    service = _service(settings, tmp_path)
    service.ask("What does the document say about penalty notices?", "twice")
    service.agent.conversations.clear()
    first = service.transcript("twice")
    second = service.transcript("twice")
    assert len(second["turns"]) == len(first["turns"])
    assert second["source"] == "live", "the second read is served from memory"


# --------------------------------------------------------------------------- #
# The recorded model is the one that answered
# --------------------------------------------------------------------------- #
def test_the_recorded_model_is_the_one_that_served(
    golden_results: Sequence[GoldenResult], settings: Settings
) -> None:
    for result in golden_results:
        metrics = result.answer.metrics
        if metrics.refused:
            assert not metrics.model, "no model was called for a refusal"
            continue
        assert metrics.model == settings.model, (
            "the row must name the model that answered, not the one configured"
        )
        assert not metrics.fallback


def test_a_fallback_is_recorded_rather_than_papered_over(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Configure a hosted model with no key: the stub answers, and the row, the
    API payload and the audit record must all say so."""
    from rights_agent.agent import Agent
    from rights_agent.audit import AuditLog
    from rights_agent.metrics import MetricsSink

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    agent = Agent(
        settings.with_overrides(model="deepseek-v4-flash", tracing_enabled=False),
        sink=MetricsSink(tmp_path / "metrics.jsonl"),
        audit=AuditLog(tmp_path / "audit.jsonl"),
        init_tracing=False,
    )
    answer = agent.ask("What does the document say about bereavement leave?", session_id="fb")

    assert answer.metrics.model == "stub-local", "the stub answered"
    assert answer.metrics.requested_model == "deepseek-v4-flash"
    assert answer.metrics.fallback is True
    payload = answer.to_dict()
    assert payload["model"] == "stub-local" and payload["fallback"] is True

    record = agent.audit.read()[-1]
    assert record["model"] == "stub-local"
    assert record["requested_model"] == "deepseek-v4-flash"
    assert record["fallback"] is True


def test_both_cost_figures_price_the_same_model(
    golden_results: Sequence[GoldenResult],
) -> None:
    """``cost_usd`` and ``cost_components`` disagreeing about which model they
    are about is worse than either being absent."""
    from rights_agent.config import cost_usd

    for result in golden_results:
        metrics = result.answer.metrics
        if metrics.refused:
            continue
        expected, _ = cost_usd(
            metrics.model, metrics.prompt_tokens, metrics.completion_tokens, metrics.cached_tokens
        )
        components = metrics.cost_components
        assert components["generation_input"] + components["generation_output"] == pytest.approx(
            expected, rel=1e-6
        ), f"{result.id}: the component model and cost_usd priced different models"


def test_the_dashboard_says_whether_the_model_can_actually_serve(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dashboard showing a model name the process cannot reach is a dashboard
    that lies."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    unreachable = _service(settings.with_overrides(model="deepseek-v4-flash"), tmp_path)
    assert unreachable.state()["runtime"]["model_available"] is False

    offline = _service(settings, tmp_path / "b")
    assert offline.state()["runtime"]["model_available"] is True


def test_the_gate_evaluates_the_offline_stub(settings: Settings) -> None:
    """A merge gate whose result depends on whether an API key happened to be in
    the environment is not a gate."""
    from conftest import GATE_MODEL

    assert settings.model == GATE_MODEL


def test_every_suggestion_behaves_as_advertised(
    settings: Settings, tmp_path: Path
) -> None:
    """A suggested question that refuses in front of a room is worse than no
    suggestion at all.

    Run against the gate's offline stub, so this checks *retrieval and the gate*
    -- the parts that decide whether a chip works -- without depending on a
    hosted provider.
    """
    from rights_agent.demo.app import FOLLOW_UP_SUGGESTION, REFUSAL_SUGGESTION, SUGGESTIONS

    service = _service(settings, tmp_path)
    assert REFUSAL_SUGGESTION in SUGGESTIONS
    assert FOLLOW_UP_SUGGESTION in SUGGESTIONS

    failures: list[str] = []
    for question in SUGGESTIONS:
        answer = service.ask(question, "suggestions")
        if question == REFUSAL_SUGGESTION:
            if not answer["refused"]:
                failures.append(f"{question!r} was meant to be refused and was answered")
            continue
        if answer["refused"]:
            failures.append(
                f"{question!r} refused at sufficiency {answer['sufficiency']:.3f} "
                f"(threshold {settings.sufficiency_threshold:.2f})"
            )
        elif not answer["citations"]:
            failures.append(f"{question!r} answered without a citation")
    assert not failures, "suggestion chips that misbehave:\n  " + "\n  ".join(failures)


# --------------------------------------------------------------------------- #
# Security posture of the dashboard
#
# Each of these corresponds to a finding from security/nuclei/. They live in the
# blocking suite because a control that is not asserted is a control that comes
# back off in the next refactor.
# --------------------------------------------------------------------------- #
def test_the_server_refuses_to_run_wide_open(settings: Settings) -> None:
    """/api/job can run `reset` and `tamper_audit`, which delete and corrupt the
    audit record. Reachable from a network with no token, that hands the control
    this project argues for to anyone who can route to the port."""
    from rights_agent.config import ConfigError
    from rights_agent.demo.app import serve

    exposed = settings.with_overrides(
        demo_host="0.0.0.0",  # noqa: S104 - the point of the test
        demo_token="",
        demo_allow_insecure=False,
        tracing_enabled=False,
    )
    with pytest.raises(ConfigError) as excinfo:
        serve(exposed)
    message = str(excinfo.value)
    assert "RIGHTS_DEMO_TOKEN" in message
    assert "127.0.0.1" in message, "the error must name both ways out"


def test_loopback_needs_no_token(settings: Settings) -> None:
    """A password prompt in front of a local demo is friction with no benefit."""
    assert settings.with_overrides(demo_host="127.0.0.1").demo_is_loopback
    assert not settings.with_overrides(demo_host="0.0.0.0").demo_is_loopback  # noqa: S104


def test_the_token_gates_exactly_the_dangerous_endpoints(
    settings: Settings, tmp_path: Path
) -> None:
    from rights_agent.demo.app import PROTECTED_PATHS

    service = _service(settings.with_overrides(demo_token="s3cret"), tmp_path)
    assert PROTECTED_PATHS == {"/api/job", "/api/degraded", "/api/chat/reset", "/api/audit"}
    for path in PROTECTED_PATHS:
        assert not service.authorised(path, ""), f"{path} accepted an empty token"
        assert not service.authorised(path, "wrong"), f"{path} accepted a wrong token"
        assert service.authorised(path, "s3cret")
    # Reading the dashboard and asking a question stay open: the token exists to
    # stop the audit record being erased, not to put a login on a demo.
    for path in ("/", "/api/state", "/api/chat", "/api/chat/history"):
        assert service.authorised(path, "")


def test_no_token_configured_means_no_gate(settings: Settings, tmp_path: Path) -> None:
    service = _service(settings, tmp_path)
    assert service.settings.demo_token == ""
    assert service.authorised("/api/job", "")


def test_every_response_carries_security_headers() -> None:
    from rights_agent.demo.app import SECURITY_HEADERS

    names = {name for name, _ in SECURITY_HEADERS}
    assert {
        "Content-Security-Policy",
        "X-Content-Type-Options",
        "X-Frame-Options",
        "Referrer-Policy",
    } <= names
    csp = dict(SECURITY_HEADERS)["Content-Security-Policy"]
    assert "frame-ancestors 'none'" in csp
    assert "default-src 'self'" in csp


def test_concurrent_chats_are_bounded(settings: Settings, tmp_path: Path) -> None:
    """Each chat is a thread and a billable model call, so an unbounded endpoint
    is an unbounded invoice."""
    service = _service(settings.with_overrides(demo_max_concurrent_chats=1), tmp_path)
    streams = [service.ask_streaming("What does the document say about tips?", "cap-1")]
    first = next(streams[0])
    assert first["type"] == "start"
    refused = list(service.ask_streaming("Another question entirely?", "cap-2"))
    assert refused[0]["type"] == "error"
    assert "at capacity" in refused[0]["error"]
    for _ in streams[0]:
        pass


def test_the_audit_endpoint_is_protected_not_merely_redacted(
    settings: Settings, tmp_path: Path
) -> None:
    """Redaction at capture limits the damage; it does not make the store public.
    The records still say who asked what, when, and about which provision."""
    from rights_agent.demo.app import PROTECTED_PATHS

    assert "/api/audit" in PROTECTED_PATHS
    service = _service(settings.with_overrides(demo_token="s3cret"), tmp_path)
    assert not service.authorised("/api/audit", "")
