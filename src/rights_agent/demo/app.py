"""The demo runner: one command, one browser tab.

    uv run rights-demo        # → http://127.0.0.1:8000

**Standard library only** -- ``http.server`` and one static HTML page.  No
Flask, no FastAPI, no build step, no npm.  The worst moment in a live demo is a
dependency that resolved differently that morning.

The server is threaded, so a long job never blocks a poll, and the page never
waits on anything: it posts a job, then polls ``/api/state``.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import secrets
import signal
import threading
import time
import uuid
from functools import partial
from hmac import compare_digest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qs, urlparse

from rights_agent.agent import Agent
from rights_agent.audit import RETENTION_FLOOR_DAYS, RetentionPolicy
from rights_agent.config import (
    ConfigError,
    PRICING,
    PRICING_AS_OF,
    Settings,
    price_for,
    settings as load_settings,
)
from rights_agent.conversation import reconstruct_turns, summarise_sessions
from rights_agent.demo.jobs import JobRunner
from rights_agent.embedding import EmbedderError
from rights_agent.entrypoints import operator_error_exit
from rights_agent.log import configure_logging, get_logger
from rights_agent.metrics import MetricsSink, summarise
from rights_agent.store import IndexNotBuiltError, StoreError
from rights_agent.telemetry import shutdown_telemetry, telemetry_status

log = get_logger("demo.app")

STATIC_DIR = Path(__file__).parent / "static"
MAX_BODY_BYTES = 64 * 1024
RECENT_ROWS = 500

#: How long a streaming reader waits for the next chunk before giving up. The
#: generator runs on its own thread; a hung model must not hold a connection
#: open forever.
STREAM_TIMEOUT_S = 120.0

#: Questions offered as chips in the chat box. Chosen to walk the demo: an
#: in-scope question, a follow-up that only works with conversation context, and
#: an out-of-scope one that must be refused.
SUGGESTIONS: tuple[str, ...] = (
    # The demo arc: a question the corpus answers well, a follow-up that only
    # works with conversation context, two questions with a concrete figure in
    # the provision, and one that must be refused.
    #
    # Every chip is checked against the live index in
    # ``evals/test_deterministic.py::test_every_suggestion_behaves_as_advertised``:
    # a suggested question that refuses in front of a room is worse than no
    # suggestion at all.
    "What does the document say about bereavement leave?",
    "How long is it?",
    "When must an employer give notice of a shift?",
    "What is the threshold for a penalty notice?",
    "What are the rules for cryptocurrency mining?",
)

#: The chip that is *meant* to be refused, so the test can tell the difference
#: between a demo of the gate and a broken suggestion.
REFUSAL_SUGGESTION = "What are the rules for cryptocurrency mining?"

#: The chip that only works as a follow-up to the one before it.
FOLLOW_UP_SUGGESTION = "How long is it?"


#: Sent on every response. The page loads no third-party script and has no
#: session cookie to steal, so the practical exposure is small -- but framing and
#: MIME sniffing are available for free, and each header costs one line. Reported
#: by ``security/nuclei/missing-security-headers.yaml``.
SECURITY_HEADERS: tuple[tuple[str, str], ...] = (
    # Self only: no third-party origin should be able to load or frame this.
    (
        "Content-Security-Policy",
        "default-src 'self'; style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; connect-src 'self'; "
        "img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
    ),
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "no-referrer"),
    ("Cross-Origin-Opener-Policy", "same-origin"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
    ("Permissions-Policy", "geolocation=(), camera=(), microphone=()"),
)

#: Paths that change state or read the audit record. Gated by
#: ``RIGHTS_DEMO_TOKEN`` when one is set: ``/api/job`` can run ``reset`` and
#: ``tamper_audit``, which delete and corrupt the compliance record this project
#: spends a whole slide arguing for.
PROTECTED_PATHS: frozenset[str] = frozenset(
    {"/api/job", "/api/degraded", "/api/chat/reset", "/api/audit"}
)

#: ``/api/chunks`` is deliberately *not* on that list, and the reason is worth
#: writing down because the last audit here found a protected read reachable
#: through an unprotected path. It serves the indexed corpus -- public UK
#: legislation, already returned verbatim in every answer's "retrieved
#: provisions" detail -- and it reads the index, not the audit record, so it is
#: not a side door around ``/api/audit``. It changes no state.

#: Header the token travels in.
TOKEN_HEADER = "X-Demo-Token"

#: Whose requests appear in the chat's history list. The traffic controls use a
#: different actor and the eval suite uses another again, so filtering on this
#: keeps a conversation list a list of conversations.
CHAT_ACTOR = "dashboard"


class DemoService:
    """Everything the handler needs, constructed once.

    ``agent`` and ``sink`` are injectable so a test can point the metrics log and
    the audit chain at a temporary directory. Without that, asserting on the
    dashboard writes rows into the operator's ``runs/`` -- and an audit chain
    whose contents depend on whether the test suite ran this morning is not an
    audit chain.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        agent: Agent | None = None,
        sink: MetricsSink | None = None,
    ) -> None:
        self.settings = settings
        self.agent = agent or Agent(settings)
        self.sink = sink or self.agent.sink
        self.runner = JobRunner(self.agent, settings)
        self.last_answer: dict[str, Any] | None = None
        self._lock = threading.Lock()
        # Bounded: each chat is a thread and a billable model call, so an
        # unbounded endpoint is an unbounded invoice as well as unbounded threads.
        self._chat_slots = threading.BoundedSemaphore(settings.demo_max_concurrent_chats)

    # ---- actions ----------------------------------------------------------
    def ask(self, question: str, session_id: str | None) -> dict[str, Any]:
        answer = self.agent.ask(question, session_id=session_id, user_id="dashboard")
        payload = answer.to_dict()
        with self._lock:
            self.last_answer = payload
        return payload

    def authorised(self, path: str, token: str) -> bool:
        """Whether ``token`` may call ``path``.

        No token configured means no gate -- the demo has to stay a demo on a
        laptop. :func:`serve` is what refuses to run wide open on an address the
        network can reach.
        """
        if not self.settings.demo_token or path not in PROTECTED_PATHS:
            return True
        return compare_digest(token, self.settings.demo_token)

    def ask_streaming(self, question: str, session_id: str) -> Iterator[dict[str, Any]]:
        """Yield ``{"type": ...}`` events as the answer is produced.

        The generation runs on its own thread and pushes chunks onto a queue,
        so the HTTP writer never has to wait on the graph and the graph never
        has to know it is being streamed. The first ``token`` event is what
        makes TTFT visible in the UI rather than just reported after the fact.
        """
        if not self._chat_slots.acquire(blocking=False):
            yield {
                "type": "error",
                "error": (
                    f"at capacity: {self.settings.demo_max_concurrent_chats} concurrent "
                    "requests already in flight. Raise RIGHTS_DEMO_MAX_CONCURRENT_CHATS "
                    "if that is genuinely what you want to pay for."
                ),
            }
            return
        events: queue.Queue[Any] = queue.Queue()
        sentinel = object()
        holder: dict[str, Any] = {}

        def run() -> None:
            try:
                answer = self.agent.ask(
                    question,
                    session_id=session_id,
                    user_id="dashboard",
                    on_token=lambda chunk: events.put({"type": "token", "text": chunk}),
                )
                holder["answer"] = answer.to_dict()
            except Exception as exc:  # noqa: BLE001 - reported to the client
                log.exception("streamed ask failed")
                holder["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                self._chat_slots.release()
                events.put(sentinel)

        worker = threading.Thread(target=run, name="chat-stream", daemon=True)
        worker.start()

        yield {"type": "start", "session_id": session_id, "question": question}
        while True:
            try:
                item = events.get(timeout=STREAM_TIMEOUT_S)
            except queue.Empty:
                yield {"type": "error", "error": f"no output for {STREAM_TIMEOUT_S:.0f}s"}
                return
            if item is sentinel:
                break
            yield item

        worker.join(timeout=5)
        if "error" in holder:
            yield {"type": "error", "error": holder["error"]}
            return
        payload = holder.get("answer", {})
        with self._lock:
            self.last_answer = payload
        yield {"type": "answer", "answer": payload}

    def transcript(self, session_id: str, *, may_read_audit: bool = True) -> dict[str, Any]:
        """One conversation, from memory if it is still there and from the audit
        record if it is not.

        The two tiers are labelled rather than blended: a reconstructed turn has
        the citations and the numbers but not the answer prose, because the audit
        record deliberately does not store it.

        ``may_read_audit`` is the caller's authorisation for the audit store, and
        it has to be honoured here too. Protecting ``/api/audit`` while this
        endpoint reads the same records is not a control, it is a side door --
        found by ``security/nuclei/guessable-session-id.yaml`` still matching
        after the live transcript had been cleared.
        """
        live = self.agent.conversations.transcript(session_id)
        if live:
            return {"session_id": session_id, "source": "live", "turns": live}
        if not may_read_audit:
            return {
                "session_id": session_id,
                "source": "empty",
                "turns": [],
                "note": (
                    "This conversation is no longer in memory. Reconstructing it means "
                    f"reading the audit record, which needs {TOKEN_HEADER}."
                ),
            }
        rows = [
            row
            for row in self._audit_rows()
            if str(row.get("session_id") or "") == session_id
            and str(row.get("actor") or "") == CHAT_ACTOR
        ]
        turns = reconstruct_turns(rows)
        if turns:
            # Reopening a conversation puts its questions back into the working
            # transcript, so the next follow-up has something to resolve against.
            # Done here rather than in the agent: the agent must not read an
            # append-only compliance artefact to make a retrieval decision, and
            # "reopen this conversation" is a UI action.
            for turn in turns:
                self.agent.conversations.add(session_id, turn)
        return {
            "session_id": session_id,
            "source": "audit" if rows else "empty",
            "turns": [turn.to_dict() for turn in turns],
            "note": (
                "Rebuilt from the audit record. Transcripts are held in memory on purpose, "
                "so a restart leaves the questions, the citations and the numbers — but not "
                "the answers, which are not stored durably anywhere. Follow-ups work again "
                "from here: reopening restored the questions."
            )
            if rows
            else "",
        }

    def sessions(self) -> dict[str, Any]:
        """The history list, newest first."""
        summaries = summarise_sessions(
            self.agent.conversations.live(),
            self._audit_rows(),
            self.sink.read(limit=RECENT_ROWS),
            actors=[CHAT_ACTOR],
        )
        return {
            "sessions": [summary.to_dict() for summary in summaries],
            "note": (
                "Live conversations keep their answers; audit-only ones keep the questions, "
                "citations and numbers. Cost is grouped by session, the same way the cost "
                "panel groups it."
            ),
        }

    def _audit_rows(self) -> list[dict[str, Any]]:
        """Audit records, or an empty list if the store cannot be read.

        History is a convenience; a corrupt or absent audit store must not take
        the chat interface down with it. The audit panel is where that failure is
        reported.
        """
        if not self.settings.audit_enabled:
            return []
        try:
            return self.agent.audit.read()
        except Exception as exc:  # noqa: BLE001 - reported by the audit panel
            log.warning("history unavailable from the audit record: %s", exc)
            return []

    def clear_session(self, session_id: str) -> None:
        self.agent.conversations.clear(session_id)

    def state(self) -> dict[str, Any]:
        rows = self.sink.read(limit=RECENT_ROWS)
        summary = summarise(
            rows,
            requests_per_day=self.settings.projection_requests_per_day,
            window=self.settings.panel_window,
        )
        with self._lock:
            last = self.last_answer
        manifest = self.agent.deps.retriever.manifest if self.agent.deps.retriever else None
        return {
            "index": {
                "index_version": self.agent.index_version,
                "embedding_model": manifest.embedding_model if manifest else "",
                "parser_version": manifest.parser_version if manifest else "",
                "corpus": Path(manifest.corpus_path).name if manifest else "",
                "collections": manifest.collections if manifest else {},
                "built_at": manifest.built_at if manifest else "",
            },
            "audit": self._audit_state(),
            "suggestions": list(SUGGESTIONS),
            "runtime": {
                "model": self.settings.model,
                "model_label": price_for(self.settings.model).label(self.settings.model),
                # Whether the configured model can actually serve, checked once
                # rather than discovered per request. A dashboard that shows a
                # model name the process cannot reach is a dashboard that lies.
                "model_available": self._model_available(),
                "degraded": self.agent.degraded,
                "top_k": self.settings.top_k,
                "sufficiency_threshold": self.settings.sufficiency_threshold,
                "max_attempts": self.settings.max_attempts,
                "pricing_as_of": PRICING_AS_OF,
                "pricing_models": sorted(PRICING),
                "tracing": telemetry_status().status,
                "tracing_enabled": telemetry_status().enabled,
                "phoenix_endpoint": self.settings.phoenix_endpoint,
                "projection_requests_per_day": self.settings.projection_requests_per_day,
                "judge_sample_rate": self.settings.judge_sample_rate,
            },
            "summary": summary,
            "last_answer": last,
            **self.runner.snapshot(),
        }

    def _model_available(self) -> bool:
        """Can the configured model serve, or will every request fall back?"""
        from rights_agent.llm import STUB_MODEL, make_client

        if self.settings.model == STUB_MODEL:
            return True
        try:
            return make_client(self.settings).model == self.settings.model
        except Exception:  # noqa: BLE001 - the panel must render regardless
            return False

    def _audit_state(self) -> dict[str, Any]:
        """Chain status and the retention policy, both named.

        The retention number is reported with its two boundaries attached,
        because "we keep it for N days" is only defensible with the reason.
        """
        policy = RetentionPolicy(
            floor_days=RETENTION_FLOOR_DAYS, configured_days=self.settings.retention_days
        )
        try:
            verification = self.agent.audit.verify()
            chain = verification.to_dict()
        except Exception as exc:  # noqa: BLE001 - a corrupt store must still render
            chain = {"ok": False, "records": 0, "verified": 0, "reason": str(exc)}
        return {
            "enabled": self.settings.audit_enabled,
            "chain": chain,
            "retention": policy.to_dict(),
        }

    # ---- the index itself -------------------------------------------------
    def chunks(
        self, query: str = "", limit: int = 25, offset: int = 0, with_vector: str = ""
    ) -> dict[str, Any]:
        """What is actually in the vector store, one chunk at a time.

        The panel this feeds exists because "we embedded the document" is the
        step everyone nods along to and nobody looks at. Reading a row makes
        three otherwise abstract claims concrete: the embedded text *starts with
        the breadcrumb*, so the citation is in the vector rather than beside it;
        every row carries the ``index_version`` that produced it; and the vector
        is 1,536 floats that no amount of staring will explain, which is the
        honest reason retrieval is judged by what it returns rather than by
        inspection.

        ``query`` runs a real similarity search, so the panel doubles as a way to
        show *why* a provision was retrieved. Without one it pages through the
        collection in insertion order.
        """
        retriever = self.agent.deps.retriever
        if retriever is None:
            return {"error": "no index", "chunks": [], "total": 0}

        limit = max(1, min(limit, 100))
        collection = retriever.leaves
        total = retriever.leaf_count
        rows: list[dict[str, Any]] = []

        if query.strip():
            docs = retriever.search(query, k=limit, expand=False)
            ids = [doc.id for doc in docs]
            scores = {doc.id: doc.score for doc in docs}
            raw = collection.get(ids=ids, include=["documents", "metadatas"]) if ids else {}
        else:
            raw = collection.get(
                limit=limit, offset=max(0, offset), include=["documents", "metadatas"]
            )
            scores = {}

        for index, identifier in enumerate(raw.get("ids") or []):
            metadata = dict((raw.get("metadatas") or [{}])[index] or {})
            embedded = str(((raw.get("documents") or [""])[index]) or "")
            rows.append(
                {
                    "id": str(identifier),
                    "citation": str(metadata.get("citation") or ""),
                    "breadcrumb": str(metadata.get("breadcrumb") or ""),
                    "embedded_text": embedded,
                    "raw_text": str(metadata.get("raw_text") or ""),
                    "index_version": str(metadata.get("index_version") or ""),
                    "page": metadata.get("page"),
                    "kind": str(metadata.get("kind") or ""),
                    "host_document": str(metadata.get("host_document") or ""),
                    "inserted_by": str(metadata.get("inserted_by") or ""),
                    "score": round(float(scores[identifier]), 4) if identifier in scores else None,
                }
            )
        if query.strip():
            rows.sort(key=lambda row: -(row["score"] or 0.0))

        payload: dict[str, Any] = {
            "chunks": rows,
            "total": total,
            "offset": 0 if query.strip() else max(0, offset),
            "limit": limit,
            "query": query,
            "index_version": self.agent.index_version,
            "embedder": retriever.manifest.embedding_model,
        }
        if with_vector:
            payload["vector"] = self._vector(with_vector)
        return payload

    def _vector(self, chunk_id: str) -> dict[str, Any]:
        """One chunk's embedding, with the summary that makes it readable.

        The full vector is returned as well as the preview: a demo that shows
        eight of 1,536 numbers and calls it "the embedding" is doing the same
        hand-waving this panel exists to stop.
        """
        retriever = self.agent.deps.retriever
        if retriever is None:
            return {"error": "no index"}
        raw = retriever.leaves.get(ids=[chunk_id], include=["embeddings"])
        embeddings = raw.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return {"error": f"no embedding for {chunk_id}"}
        vector = [float(value) for value in embeddings[0]]
        magnitude = math.sqrt(sum(value * value for value in vector))
        return {
            "id": chunk_id,
            "dimensions": len(vector),
            # Normalised vectors are what cosine distance assumes. Reported
            # rather than asserted: an embedder that returns un-normalised
            # vectors is a thing you want to see, not a thing you want hidden.
            "norm": round(magnitude, 6),
            "min": round(min(vector), 6),
            "max": round(max(vector), 6),
            "preview": [round(value, 6) for value in vector[:16]],
            "values": [round(value, 6) for value in vector],
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "index_version": self.agent.index_version,
            "degraded": self.agent.degraded,
            "tracing": telemetry_status().enabled,
        }


class DemoHandler(BaseHTTPRequestHandler):
    """Minimal JSON + static file handler."""

    server_version = "rights-agent-demo"
    protocol_version = "HTTP/1.1"

    def __init__(self, *args: Any, service: DemoService, **kwargs: Any) -> None:
        self.service = service
        super().__init__(*args, **kwargs)

    # ---- plumbing ---------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        log.debug("%s - %s", self.address_string(), fmt % args)

    def _security_headers(self) -> None:
        for name, value in SECURITY_HEADERS:
            self.send_header(name, value)

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._security_headers()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _stream_ndjson(self, events: Iterator[dict[str, Any]]) -> None:
        """Write newline-delimited JSON using HTTP chunked transfer.

        Chunked framing by hand because ``http.server`` will not do it for us,
        and a streamed body has no Content-Length to send. NDJSON rather than
        SSE framing: the client is a ``fetch`` reader, and one JSON object per
        line is less to get wrong on both sides.
        """
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self._security_headers()
        # Deliberately *not* ``Connection: close``: closing the socket from the
        # handler can emit an RST while the terminating chunk is still in the
        # send buffer, which a browser reports as an aborted request. Chunked
        # framing already delimits the end; keep-alive is the graceful path.
        self.end_headers()
        try:
            for event in events:
                payload = (json.dumps(event, ensure_ascii=False, default=str) + "\n").encode("utf-8")
                self.wfile.write(f"{len(payload):X}\r\n".encode("ascii"))
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # The reader navigated away mid-answer. Not an error worth a stack
            # trace: the generation thread finishes and records its row anyway.
            log.debug("client disconnected during stream")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ValueError(f"request body too large ({length} bytes)")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"body is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("body must be a JSON object")
        return payload

    # ---- routes -----------------------------------------------------------
    def _refuse_unauthorised(self, path: str) -> bool:
        """Reject and report; returns True when the request was refused."""
        if self.service.authorised(path, self.headers.get(TOKEN_HEADER, "") or ""):
            return False
        log.warning("unauthorised %s %s from %s", self.command, path, self.address_string())
        self._json(
            {"error": f"{TOKEN_HEADER} required for {path}"}, HTTPStatus.UNAUTHORIZED
        )
        return True

    def do_GET(self) -> None:  # noqa: N802 - http.server's naming
        path = urlparse(self.path).path
        try:
            if self._refuse_unauthorised(path):
                return
            if path in {"/", "/index.html"}:
                self._static("index.html", "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json(self.service.state())
            elif path == "/api/health":
                self._json(self.service.health())
            elif path == "/api/chat/history":
                params = parse_qs(urlparse(self.path).query)
                session_id = (params.get("session_id") or [""])[0]
                if not session_id:
                    self._json({"error": "session_id is required"}, HTTPStatus.BAD_REQUEST)
                    return
                self._json(
                    self.service.transcript(
                        session_id,
                        may_read_audit=self.service.authorised(
                            "/api/audit", self.headers.get(TOKEN_HEADER, "") or ""
                        ),
                    )
                )
            elif path == "/api/chat/sessions":
                self._json(self.service.sessions())
            elif path == "/api/chunks":
                params = parse_qs(urlparse(self.path).query)
                self._json(
                    self.service.chunks(
                        query=(params.get("q") or [""])[0],
                        limit=int((params.get("limit") or ["25"])[0]),
                        offset=int((params.get("offset") or ["0"])[0]),
                        with_vector=(params.get("vector") or [""])[0],
                    )
                )
            elif path == "/api/audit":
                params = parse_qs(urlparse(self.path).query)
                limit = int((params.get("limit") or ["20"])[0])
                rows = self.service.agent.audit.read()
                self._json(
                    {
                        "records": rows[-max(1, min(limit, 200)):],
                        "total": len(rows),
                        **self.service._audit_state(),
                    }
                )
            else:
                self._json({"error": f"no route for {path}"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001 - a 500 must still be JSON
            log.exception("GET %s failed", path)
            self._json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if self._refuse_unauthorised(path):
                return
            payload = self._read_json()
            if path == "/api/ask":
                question = str(payload.get("question", "")).strip()
                if not question:
                    self._json({"error": "question is required"}, HTTPStatus.BAD_REQUEST)
                    return
                self._json(self.service.ask(question, payload.get("session_id")))
            elif path == "/api/chat":
                question = str(payload.get("question", "")).strip()
                if not question:
                    self._json({"error": "question is required"}, HTTPStatus.BAD_REQUEST)
                    return
                session_id = str(payload.get("session_id") or "") or f"chat-{uuid.uuid4().hex[:8]}"
                self._stream_ndjson(self.service.ask_streaming(question, session_id))
            elif path == "/api/chat/reset":
                session_id = str(payload.get("session_id") or "")
                self.service.clear_session(session_id)
                self._json(
                    {
                        "session_id": session_id,
                        "turns": [],
                        "note": (
                            "The working transcript was dropped. The audit records for "
                            "this session remain: an append-only compliance log that a "
                            "user request can delete is not one."
                        ),
                    }
                )
            elif path == "/api/job":
                name = str(payload.get("job", ""))
                accepted, message = self.service.runner.start(name, payload.get("params") or {})
                self._json(
                    {"accepted": accepted, "message": message},
                    HTTPStatus.OK if accepted else HTTPStatus.CONFLICT,
                )
            elif path == "/api/degraded":
                self.service.agent.set_degraded(bool(payload.get("degraded")))
                self._json({"degraded": self.service.agent.degraded})
            else:
                self._json({"error": f"no route for {path}"}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001
            log.exception("POST %s failed", path)
            self._json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _static(self, name: str, content_type: str) -> None:
        path = STATIC_DIR / name
        if not path.is_file():
            self._json({"error": f"{name} is missing from the package"}, HTTPStatus.NOT_FOUND)
            return
        self._send(HTTPStatus.OK, path.read_bytes(), content_type)


#: The warm-up question.
#:
#: It has to be one the corpus actually answers. The obvious phrasing -- "What
#: does this document cover?" -- scores 0.218 on the sufficiency gate and is
#: refused, and a refusal never reaches the model, so the warm-up warmed
#: nothing while reporting success. Deliberately not one of the SUGGESTIONS
#: either: a chip's first click should show its own prompt-cache state rather
#: than one inherited from a request nobody saw.
#:
#: ``evals/test_deterministic.py::test_the_warm_up_question_reaches_the_model``
#: asserts it still generates against the live index.
WARMUP_QUESTION = "What does the document say about trade union recognition?"


def warm_up(service: "DemoService") -> str:
    """Pay the first-request costs before anyone is watching.

    A hosted model's first call pays TLS setup and an empty prompt cache: 10.5s
    TTFT against 0.8s warm, on identical settings. The ONNX embedder pays its
    first inference the same way. Both land on whoever asks the first question,
    which at a demo is the audience.

    Recorded nowhere -- ``record=False`` keeps it out of the metrics panels, the
    history list and the audit chain, because a request nobody made should not
    appear in a record of what the system was asked.
    """
    started = time.perf_counter()
    try:
        answer = service.agent.ask(
            WARMUP_QUESTION,
            session_id="warmup",
            record=False,
            use_history=False,
            remember=False,
        )
    except Exception as exc:  # noqa: BLE001 - a cold start is not a reason not to serve
        # Loud, and not fatal: the server is still perfectly able to answer, the
        # first real question is just going to be slow.
        log.warning("warm-up failed (%s); the first request will pay for it", exc)
        return f"failed ({type(exc).__name__})"
    elapsed = (time.perf_counter() - started) * 1000
    if answer.metrics.route != "generate":
        # The expensive thing to warm is the model connection, and a refusal
        # never opens one. Said out loud rather than reported as a time, because
        # "0 ms ttft" reads like a success.
        log.warning(
            "warm-up was %sd (sufficiency %.3f), so the model was never called",
            answer.metrics.route,
            answer.metrics.sufficiency,
        )
        return (
            f"{answer.metrics.route}d after {elapsed:.0f} ms — the model was NOT "
            f"warmed; the first question will pay for it"
        )
    return f"{answer.metrics.ttft_ms:.0f} ms ttft, {elapsed:.0f} ms total"


def serve(settings: Settings | None = None) -> int:
    settings = settings or load_settings()

    # An unauthenticated /api/job can run `reset` and `tamper_audit`, which
    # delete and corrupt the audit record. On loopback that is nobody's problem.
    # Reachable from a network it is the control this project argues for, handed
    # to anyone on the wifi -- so refuse, and say which of the two fixes to pick.
    if not settings.demo_is_loopback and not settings.demo_token:
        if not settings.demo_allow_insecure:
            raise ConfigError(
                f"refusing to serve {settings.demo_host}:{settings.demo_port} with no "
                "RIGHTS_DEMO_TOKEN. /api/job can run `reset` and `tamper_audit`, which "
                "delete and corrupt the audit record, and /api/audit reads it. Either "
                "bind to 127.0.0.1 (and publish the container port as "
                "127.0.0.1:8000:8000), or set RIGHTS_DEMO_TOKEN. "
                "RIGHTS_DEMO_ALLOW_INSECURE=true overrides this on a network you trust."
            )
        log.warning(
            "serving %s with no token because RIGHTS_DEMO_ALLOW_INSECURE is set: "
            "anyone who can reach this port can erase the audit chain",
            settings.demo_host,
        )

    try:
        service = DemoService(settings)
    except (IndexNotBuiltError, StoreError, EmbedderError) as exc:
        # An operator error with a known fix. Print the fix, not a traceback.
        print(f"error: {exc}")
        return 2

    # Before the port opens, so the first request cannot race the warm-up.
    warmed = warm_up(service) if settings.warmup else "skipped (RIGHTS_WARMUP=false)"

    handler = partial(DemoHandler, service=service)
    server = ThreadingHTTPServer((settings.demo_host, settings.demo_port), handler)
    server.daemon_threads = True

    def shutdown(signum: int, _frame: Any) -> None:
        log.info("signal %s received, shutting down", signum)
        threading.Thread(target=server.shutdown, daemon=True).start()

    for received in (signal.SIGINT, signal.SIGTERM):
        signal.signal(received, shutdown)

    host = settings.demo_host if settings.demo_host != "0.0.0.0" else "127.0.0.1"  # noqa: S104
    print(f"dashboard   http://{host}:{settings.demo_port}")
    print(f"index       {service.agent.index_version}")
    print(f"model       {price_for(settings.model).label(settings.model)}")
    print(f"tracing     {telemetry_status().status}")
    print(f"warm-up     {warmed}")
    print(
        "auth        "
        + (
            f"{TOKEN_HEADER} required for {', '.join(sorted(PROTECTED_PATHS))}"
            if settings.demo_token
            else "none (loopback only)" if settings.demo_is_loopback
            else "NONE — this port is reachable from the network"
        )
    )
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        shutdown_telemetry()
    return 0


@operator_error_exit
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the observability dashboard.")
    parser.add_argument("--host", default=None, help="bind address (default RIGHTS_DEMO_HOST)")
    parser.add_argument("--port", type=int, default=None, help="port (default RIGHTS_DEMO_PORT)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    configure_logging("WARNING" if args.quiet else None)

    settings = load_settings()
    overrides: dict[str, object] = {}
    if args.host:
        overrides["demo_host"] = args.host
    if args.port:
        overrides["demo_port"] = args.port
    if overrides:
        settings = settings.with_overrides(**overrides)
    return serve(settings)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
