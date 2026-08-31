"""Span annotations: the judge's verdict attached to the trace it judged."""

from __future__ import annotations

import json
import urllib.error
from typing import Any

import pytest

from rights_agent.annotations import (
    Annotation,
    annotate_span,
    band,
    judge_annotations,
)


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None


@pytest.fixture
def captured(monkeypatch):
    """Capture the request instead of sending it."""
    sent: dict[str, Any] = {}

    def fake_urlopen(request, timeout=None):
        sent["url"] = request.full_url
        sent["body"] = json.loads(request.data)
        sent["headers"] = dict(request.headers)
        return _Response({"data": [{"id": "x"} for _ in sent["body"]["data"]]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return sent


def test_bands_read_the_number_for_someone_who_does_not_know_the_metric() -> None:
    assert band(0.95) == "good"
    assert band(0.80) == "fair"
    assert band(0.20) == "poor"


def test_scores_become_one_annotation_each(captured, isolated_settings) -> None:
    annotations = judge_annotations(
        {"groundedness": 0.95, "citation_coverage": 0.6},
        judge="heuristic-lexical",
        sufficiency=0.83,
        route="generate",
        index_version="parser-6+x+y",
    )
    accepted = annotate_span("abc123", annotations, isolated_settings)

    assert accepted == 3
    names = [item["name"] for item in captured["body"]["data"]]
    assert names == ["citation_coverage", "groundedness", "sufficiency"]
    assert all(item["span_id"] == "abc123" for item in captured["body"]["data"])
    assert captured["url"].endswith("/v1/span_annotations?sync=true")


def test_the_annotator_kind_distinguishes_the_instrument(
    captured, isolated_settings
) -> None:
    """A lexical overlap score and a model's opinion fail in completely
    different ways; one bucket for both would hide that."""
    annotate_span(
        "abc",
        judge_annotations({"groundedness": 0.9}, judge="heuristic-lexical"),
        isolated_settings,
    )
    assert captured["body"]["data"][0]["annotator_kind"] == "CODE"

    annotate_span(
        "abc",
        judge_annotations({"groundedness": 0.9}, judge="llm-judge-deepseek"),
        isolated_settings,
    )
    assert captured["body"]["data"][0]["annotator_kind"] == "LLM"


def test_re_annotating_corrects_rather_than_duplicates(
    captured, isolated_settings
) -> None:
    """A stable identifier per (span, metric): re-running the judge should
    correct the record, not append a second opinion to it."""
    annotate_span(
        "abc", judge_annotations({"groundedness": 0.9}, judge="heuristic"), isolated_settings
    )
    first = captured["body"]["data"][0]["identifier"]
    annotate_span(
        "abc", judge_annotations({"groundedness": 0.4}, judge="heuristic"), isolated_settings
    )
    assert captured["body"]["data"][0]["identifier"] == first


def test_the_routing_decision_is_annotated_next_to_the_score(
    captured, isolated_settings
) -> None:
    """The useful question is "answers that scored badly *and* were generated",
    and a filter can only ask that if the decision is on the span too."""
    annotate_span(
        "abc",
        judge_annotations({"groundedness": 0.5}, judge="h", sufficiency=0.2, route="refuse"),
        isolated_settings,
    )
    sufficiency = next(
        item for item in captured["body"]["data"] if item["name"] == "sufficiency"
    )
    assert sufficiency["result"]["label"] == "refuse"
    assert sufficiency["result"]["score"] == 0.2


def test_a_rejected_annotation_is_logged_not_raised(monkeypatch, isolated_settings) -> None:
    """Observability must never take down the thing it observes."""

    def reject(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 422, "Unprocessable", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", reject)
    assert annotate_span("abc", [Annotation("groundedness", 0.5)], isolated_settings) == 0


def test_an_unreachable_collector_is_not_an_error(monkeypatch, isolated_settings) -> None:
    def explode(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", explode)
    assert annotate_span("abc", [Annotation("groundedness", 0.5)], isolated_settings) == 0


def test_nothing_is_sent_without_a_span_or_scores(captured, isolated_settings) -> None:
    assert annotate_span("", [Annotation("groundedness", 0.5)], isolated_settings) == 0
    assert annotate_span("abc", [], isolated_settings) == 0
    assert "body" not in captured


def test_a_span_not_yet_exported_is_retried_not_dropped(monkeypatch, isolated_settings) -> None:
    """The batch processor holds spans for seconds. Posting the instant a
    request finishes reliably 404s, and that 404 reads like a bad span id."""
    from rights_agent import annotations as module

    calls = {"n": 0}

    def not_yet(request, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(
                request.full_url, 404, "Not Found", {},
                __import__("io").BytesIO(b'{"detail":"Spans with IDs abc do not exist."}'),
            )
        return _Response({"data": [{"id": "x"}]})

    monkeypatch.setattr("urllib.request.urlopen", not_yet)
    monkeypatch.setattr(module, "flush_spans", lambda *a, **k: None)
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)

    module._annotate_with_retry("abc", [Annotation("groundedness", 0.9)], isolated_settings)

    assert calls["n"] == 3, "gave up before the span arrived"


def test_a_real_rejection_is_not_retried(monkeypatch, isolated_settings) -> None:
    """422 means Phoenix will never accept this. Retrying it is four times the
    latency for the same failure."""
    from rights_agent import annotations as module

    calls = {"n": 0}

    def rejected(request, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(request.full_url, 422, "Bad", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", rejected)
    monkeypatch.setattr(module, "flush_spans", lambda *a, **k: None)
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)

    module._annotate_with_retry("abc", [Annotation("groundedness", 0.9)], isolated_settings)

    assert calls["n"] == 1


def test_annotating_never_blocks_the_answer(monkeypatch, isolated_settings) -> None:
    """It is a demo about latency: an HTTP round trip per answer would inflate
    the number on the panel with the cost of reporting it."""
    import time as real_time

    from rights_agent import annotations as module

    def slow(request, timeout=None):
        real_time.sleep(0.4)
        return _Response({"data": [{"id": "x"}]})

    monkeypatch.setattr("urllib.request.urlopen", slow)
    monkeypatch.setattr(module, "flush_spans", lambda *a, **k: None)

    started = real_time.perf_counter()
    module.annotate_span_later("abc", [Annotation("groundedness", 0.9)], isolated_settings)
    elapsed = real_time.perf_counter() - started

    assert elapsed < 0.1, f"annotation blocked the caller for {elapsed * 1000:.0f}ms"
