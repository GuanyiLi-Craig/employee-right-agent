"""Phoenix span annotations: the judge's verdict, attached to the trace.

The scores already ride on the ``rag.judge`` span as attributes, which is enough
to read one request. Annotations are a different thing and worth both: Phoenix
treats them as first-class feedback on a span, so they aggregate across a
project, filter (``groundedness < 0.7``), and sit next to human labels in the
same list. An attribute is something you read; an annotation is something you
can ask a question of.

Three design decisions worth stating, because each one is a claim about
provenance:

* **``annotator_kind`` distinguishes the instrument.** The offline judge is
  deterministic code and is annotated ``CODE``; a model-graded judge is ``LLM``.
  Collapsing the two would put a lexical overlap score and a model's opinion in
  one bucket, and they fail in completely different ways.
* **``identifier`` makes a re-annotation an update, not a duplicate.** Re-running
  the judge over the same span should correct the record, not append a second
  opinion to it.
* **Failure here is never fatal.** Observability must not be able to take down
  the thing it observes, so every call is wrapped and every error is logged and
  swallowed -- the same rule the rest of the Phoenix integration follows.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rights_agent.config import Settings
from rights_agent.log import get_logger
from rights_agent.telemetry import flush_spans

log = get_logger("annotations")

#: Phoenix's REST path for span-level feedback.
ANNOTATIONS_PATH = "/v1/span_annotations"

#: Kept short: this runs on the request path, and a slow collector must not
#: become slow answers.
TIMEOUT_S = 3.0

#: Bands used for the annotation label. A score carries the number; a label
#: carries the reading of it, which is what makes a project-wide filter useful
#: to someone who does not know what 0.62 means for this metric.
GOOD = 0.90
FAIR = 0.70


def band(score: float) -> str:
    """``0.94 -> "good"``. The reading of a number, not the number."""
    if score >= GOOD:
        return "good"
    if score >= FAIR:
        return "fair"
    return "poor"


@dataclass(frozen=True, slots=True)
class Annotation:
    """One scored judgement about one span."""

    name: str
    score: float
    explanation: str = ""
    annotator_kind: str = "CODE"
    label: str = ""
    metadata: Mapping[str, Any] | None = None

    def payload(self, span_id: str) -> dict[str, Any]:
        return {
            "span_id": span_id,
            "name": self.name,
            "annotator_kind": self.annotator_kind,
            # Stable per (span, metric), so re-running the judge corrects the
            # record rather than appending a second opinion to it.
            "identifier": f"rights-agent:{self.name}",
            "result": {
                "label": self.label or band(self.score),
                "score": round(float(self.score), 6),
                "explanation": self.explanation or None,
            },
            "metadata": dict(self.metadata or {}),
        }


#: How many times to retry a span Phoenix has not received yet, and how long to
#: wait between tries. The batch span processor holds spans for a few seconds, so
#: an annotation posted the instant a request finishes reliably 404s.
NOT_FOUND_RETRIES = 4
NOT_FOUND_DELAY_S = 1.0


def annotate_span_later(
    span_id: str, annotations: Sequence[Annotation], settings: Settings
) -> None:
    """Annotate off the request path, once the span has actually been exported.

    Two reasons this is not done inline. The span does not exist yet -- the batch
    processor holds it for seconds, and Phoenix answers ``404 Spans with IDs ...
    do not exist``, which reads like a bad id rather than a race. And this is a
    demo whose whole subject is latency: adding an HTTP round trip to every
    answer would inflate the number on the panel with the cost of reporting it.
    """
    if not span_id or not annotations:
        return
    thread = threading.Thread(
        target=_annotate_with_retry,
        args=(span_id, annotations, settings),
        name=f"annotate-{span_id[:8]}",
        daemon=True,
    )
    thread.start()


def _annotate_with_retry(
    span_id: str, annotations: Sequence[Annotation], settings: Settings
) -> None:
    flush_spans()
    for attempt in range(1, NOT_FOUND_RETRIES + 1):
        accepted, missing = _post(span_id, annotations, settings)
        if accepted or not missing:
            return
        if attempt < NOT_FOUND_RETRIES:
            time.sleep(NOT_FOUND_DELAY_S)
            flush_spans()
    log.warning(
        "span %s never appeared in Phoenix; %d annotations dropped",
        span_id,
        len(annotations),
    )


def annotate_span(
    span_id: str, annotations: Sequence[Annotation], settings: Settings
) -> int:
    """POST ``annotations`` against ``span_id``. Returns how many were accepted.

    Never raises: a collector that is down, slow or of a different version must
    not turn a working answer into an error.
    """
    return _post(span_id, annotations, settings)[0]


def _post(
    span_id: str, annotations: Sequence[Annotation], settings: Settings
) -> tuple[int, bool]:
    """``(accepted, span_not_found)``.

    The second value separates "Phoenix has not received the span yet", which is
    worth retrying, from "Phoenix rejected this", which is not.
    """
    if not span_id or not annotations:
        return 0, False
    url = settings.phoenix_endpoint.rstrip("/") + ANNOTATIONS_PATH + "?sync=true"
    body = json.dumps(
        {"data": [annotation.payload(span_id) for annotation in annotations]}
    ).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 - the endpoint is operator-configured
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:  # noqa: S310
            payload = json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        # Read the body: Phoenix explains rejections there, and "HTTP 422" alone
        # has sent people to the wrong file more than once.
        detail = exc.read()[:400].decode("utf-8", "replace")
        if exc.code == 404 and "do not exist" in detail:
            log.debug("span %s not exported yet", span_id)
            return 0, True
        log.warning("span annotation rejected (%s): %s", exc.code, detail)
        return 0, False
    except Exception as exc:  # noqa: BLE001 - unreachable collector, DNS, timeout
        log.warning("span annotation failed: %s", exc)
        return 0, False
    accepted = len(payload.get("data") or [])
    log.debug("annotated span %s with %d scores", span_id, accepted)
    return accepted, False


def judge_annotations(
    scores: Mapping[str, float],
    *,
    judge: str,
    sufficiency: float | None = None,
    route: str = "",
    index_version: str = "",
) -> list[Annotation]:
    """Turn one request's scores into annotations Phoenix can aggregate.

    ``sufficiency`` and ``route`` are included because the interesting question
    in Phoenix is rarely "what did it score" on its own -- it is "show me the
    answers that scored badly *and* were generated rather than refused", and a
    filter can only ask that if the decision is annotated alongside the score.
    """
    kind = "LLM" if "llm" in judge.lower() else "CODE"
    metadata = {"judge": judge, "index_version": index_version, "route": route}
    out = [
        Annotation(
            name=name,
            score=float(value),
            annotator_kind=kind,
            explanation=f"{judge} scored {name} at {float(value):.3f}",
            metadata=metadata,
        )
        for name, value in sorted(scores.items())
    ]
    if sufficiency is not None:
        out.append(
            Annotation(
                name="sufficiency",
                score=float(sufficiency),
                # Not a judgement about the answer: the retrieval gate's own
                # number, annotated so the routing decision is filterable next to
                # the quality it produced.
                annotator_kind="CODE",
                label=route or band(float(sufficiency)),
                explanation=f"retrieval sufficiency {float(sufficiency):.3f} → {route or 'n/a'}",
                metadata=metadata,
            )
        )
    return out
