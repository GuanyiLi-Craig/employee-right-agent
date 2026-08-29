# Session 5 — Demo Project Implementation Document

**Project:** an observable, evaluable RAG agent
**Stack:** `uv` · LangGraph · Arize Phoenix · ChromaDB
**Audience:** the engineer (or coding agent) building the demo
**Status:** specification — build to this, then verify against §12

---

## 1. Purpose and scope

Build a small Python project that demonstrates **how you know an LLM system works, and what it costs**. The RAG pipeline is the vehicle; the observability and evaluation layers are the subject.

The project must be demonstrable live, in front of a room, on a laptop, with the network switched off.

### 1.1 In scope

| # | Capability | Why it is in the demo |
|---|---|---|
| 1 | Simple embedding pipeline | The baseline everyone writes first. Needed for contrast. |
| 2 | Hierarchical embedding pipeline | The technique being taught. |
| 3 | Retrieval with a sufficiency gate | Gives the agent something to *decide*, which gives tracing something to show. |
| 4 | LangGraph workflow | Node boundaries become span boundaries for free. |
| 5 | Phoenix tracing | Answers "why was *this* request wrong". |
| 6 | Phoenix evaluation / experiments | Answers "is it getting worse". |
| 7 | Deterministic eval suite + CI gate | The part that can block a merge. |
| 8 | Demo runner | One command, one screen, no terminal juggling. |

### 1.2 Out of scope

Multi-tenant auth, a production web front end, model fine-tuning, distributed serving, agent tool-calling (that is Session 6), and any cloud dependency.

### 1.3 Non-negotiable constraints

1. **Runs with no API key and no network.** Both the model and the embedder must have offline fallbacks. A demo that needs conference wifi is a demo you cannot give.
2. **Deterministic enough to gate CI.** The offline path must produce the same output for the same input.
3. **No demo-only code path.** The demo runner and the test suite import the *same* modules. What the room sees is what the tests assert.
4. **Observability cannot break the app.** Every tracing import is optional; every export failure is swallowed.

---

## 2. Choosing the corpus

The hierarchical pipeline only earns its keep on a document with **real, machine-detectable structure**. Pick a corpus with numbered nesting.

**Good candidates:** legislation or regulation (Part → Chapter → Section → subsection), an internal policy manual with numbered clauses, an RFC or standard, API reference documentation, a technical handbook with numbered headings.

**Poor candidates:** blog posts, chat transcripts, support tickets, meeting notes. These are flat; the simple pipeline is the right tool and the demo has no contrast to show.

**Reference choice for this document:** a UK Public General Act as published on legislation.gov.uk. Roughly 300 pages, five to six Parts, ~160 sections, ~1,000 subsections. Substitute freely — every rule below is expressed in terms of *the tree*, not the specific document.

> **Requirement:** the corpus must be committed to the repo under `data/`. Downloading it at demo time violates constraint 1.

---

## 3. Technology decisions

| Concern | Choice | Rationale |
|---|---|---|
| Dependency management | **uv** | Single tool for venv, resolution, lockfile and script running. Reproducible across machines via a committed `uv.lock`. |
| Workflow | **LangGraph** | Explicit state machine. Each node is a natural span boundary, so the trace shape falls out of the design instead of being hand-instrumented. |
| Vector store | **ChromaDB** | Embedded, file-backed, no server to run. Supports multiple collections and metadata filtering, which the hierarchical design needs. |
| Tracing + evals | **Arize Phoenix** | Self-hostable, runs locally, and covers both halves — span viewing *and* dataset/experiment evaluation — so the demo needs one tool rather than two. |
| Semantic conventions | **OpenInference** | Makes spans render as Retriever / LLM / Evaluator and makes the trace queryable. |

### 3.1 Version compatibility

Phoenix and LangGraph both move quickly. **Pin exact versions and commit the lockfile.** Treat any version in this document as a starting point to be re-resolved, not gospel.

Two constraints discovered in practice, worth checking at bootstrap:

- **Python ≥ 3.11.** ChromaDB pulls `onnxruntime`, whose recent wheels drop 3.10.
- **Chroma custom embedding functions** must implement `name()`, `embed_documents()` and `embed_query()` in Chroma 1.x. A bare `__call__` is no longer sufficient.

---

## 4. Prerequisites

```bash
# uv — https://docs.astral.sh/uv/
curl -LsSf https://astral.sh/uv/install.sh | sh      # macOS / Linux
# or: brew install uv   |   winget install astral-sh.uv

# poppler, for PDF text extraction with layout preserved
brew install poppler                                 # macOS
sudo apt-get install -y poppler-utils                # Debian / Ubuntu
```

Verify: `uv --version` and `pdftotext -v`.

---

## 5. Bootstrap

```bash
uv init coderco-session5 --python 3.11
cd coderco-session5

uv add "chromadb==1.5.9" "langgraph==1.2.11"
uv add --optional trace "arize-phoenix" "openinference-instrumentation-langchain" \
                        "openinference-semantic-conventions" \
                        "opentelemetry-sdk" "opentelemetry-exporter-otlp"
uv add --optional models "openai" "anthropic" "tiktoken"
uv add --dev pytest

uv lock          # commit uv.lock
uv sync --extra trace --group dev
```

### 5.1 `pyproject.toml` requirements

```toml
[project]
name = "coderco-session5"
requires-python = ">=3.11"
dependencies = ["chromadb==1.5.9", "langgraph==1.2.11"]

[project.optional-dependencies]
trace  = ["arize-phoenix", "openinference-instrumentation-langchain",
          "openinference-semantic-conventions",
          "opentelemetry-sdk", "opentelemetry-exporter-otlp"]
models = ["openai", "anthropic", "tiktoken"]

[dependency-groups]
dev = ["pytest"]

[project.scripts]
s5-ingest        = "s5.pipelines.hierarchical:main"
s5-ingest-simple = "s5.pipelines.simple:main"
s5-compare       = "s5.pipelines.compare:main"
s5-ask           = "s5.cli:ask"
s5-demo          = "s5.demo.app:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/s5"]
```

**Rule:** every command in the README is `uv run …`. Never instruct a reader to activate a venv — `uv run` is the reproducible path and removes a whole class of "works on my machine".

---

## 6. Repository layout

```
coderco-session5/
├── pyproject.toml
├── uv.lock                      # committed
├── README.md
├── data/
│   └── corpus.pdf               # committed
├── src/s5/
│   ├── config.py                # settings, paths, pricing table
│   ├── telemetry.py             # Phoenix bootstrap, span helpers, metrics sink
│   ├── embedding.py             # embedder selection + offline fallback
│   ├── document/
│   │   ├── parser.py            # PDF → Node tree
│   │   └── nodes.py             # Node dataclass, breadcrumb, citation
│   ├── pipelines/
│   │   ├── simple.py            # fixed-window baseline
│   │   ├── hierarchical.py      # tree → two collections
│   │   └── compare.py           # side-by-side harness
│   ├── retrieval.py             # search, small-to-big, sufficiency
│   ├── llm.py                   # streaming clients + offline stub + TTFT/ITL
│   ├── judges.py                # RAG triad + calibration
│   ├── graph.py                 # LangGraph workflow
│   └── cli.py
├── evals/
│   ├── golden.jsonl
│   ├── calibration.jsonl
│   ├── conftest.py
│   ├── test_deterministic.py
│   └── test_quality.py
├── demo/
│   ├── app.py                   # stdlib HTTP server
│   └── index.html
└── .github/workflows/eval.yml
```

---

## 7. Core data contracts

Define these first. Everything else is written against them.

### 7.1 `Node` — the document tree

```python
@dataclass
class Node:
    kind: str                 # document | part | heading | section | subsection
    number: str = ""
    title: str = ""
    text: str = ""            # this node's own text, excluding children
    page: int = 0
    parent: "Node | None" = None
    children: list["Node"] = field(default_factory=list)

    def label(self) -> str: ...        # "Part 1 Employment Rights", "s.25 …", "(3)"
    def path(self) -> list["Node"]: ... # root → self
    def breadcrumb(self) -> str: ...    # " > ".join(label for each ancestor)
    def citation(self) -> str: ...      # short quotable form, e.g. "s.25(1)"
    def full_text(self) -> str: ...     # self + all descendants
    def walk(self): ...                 # depth-first iterator
```

**`breadcrumb()` is the single most important method in the project.** It is what turns an orphaned clause back into a findable one.

### 7.2 `Doc` — a retrieval result

```python
@dataclass
class Doc:
    id: str
    citation: str
    breadcrumb: str
    text: str
    score: float           # cosine similarity, 0..1
    parent_id: str = ""
    expanded: bool = False # was this widened to its parent provision?
    metadata: dict = field(default_factory=dict)
```

### 7.3 `RequestMetrics` — one row per request

One JSON line per request, appended to `runs/metrics.jsonl`. This file *is* the dashboard.

```
request_id · session_id · user_id · tenant · ts
question · rewritten_query · route · intent
ttft_ms · itl_ms_mean · itl_ms_p95 · e2e_ms · stage_ms{}
index_version · embedding_model · retrieved_ids[] · retrieval_scores[]
citations[] · attempts · sufficiency · refused
model · prompt_tokens · completion_tokens · cached_tokens
cost_usd · cost_breakdown{}
scores{} · trace_span_id · error
```

**Design rules.**
- `index_version` on **every** row. Without it you cannot answer "which index produced this answer" six months later.
- Keep the retrieved *context* out of this file. It belongs in the trace, under its own retention rules — not duplicated into a third store.
- One row per request, appended, never mutated.

---

## 8. Milestone 1 — the simple embedding pipeline

Build this first. It is forty lines, it is the correct default for flat prose, and it is the control in the experiment.

### 8.1 Specification

```python
CHUNK_CHARS   = 1_000     # ~250 tokens
OVERLAP_CHARS = 150       # ~15%

def extract_text(pdf: Path) -> str:
    """All pages concatenated. Structure is discarded."""

def fixed_window_chunks(text, size=CHUNK_CHARS, overlap=OVERLAP_CHARS) -> list[str]:
    """Slide a fixed window. Overlap reduces — does not remove — boundary loss."""

def ingest_simple(pdf: Path, reset: bool = True) -> dict:
    """Write chunks to collection `corpus_simple`. Returns a manifest."""
```

**Collection:** `corpus_simple`, cosine space.
**Metadata per chunk:** `{offset, chars, pipeline: "simple"}` — nothing else is available, because nothing in this pipeline ever knew what a section was.

### 8.2 Acceptance

- [ ] Builds in under 5 seconds on the reference corpus.
- [ ] `collection.count()` is within ±10% of `len(text) / (CHUNK_CHARS - OVERLAP_CHARS)`.
- [ ] Querying returns chunks. **No chunk carries a citation** — confirm this, it is the point.

---

## 9. Milestone 2 — the hierarchical embedding pipeline

### 9.1 The idea in one paragraph

A subsection lifted out of a structured document is unfindable on its own. *"The threshold is £500"* retrieves for nothing: the words that make it findable — the topic, the jurisdiction, the section it belongs to — live in its **ancestors**. So rebuild the tree the authors wrote, and **prepend each node's breadcrumb to the text you embed**. Not store beside — *embed with*.

```
Document
 └── Part 1  Employment Rights
      └── Zero hours workers, etc                 ← cross-heading
           └── s.1  Right to guaranteed hours
                └── (1) An employer must make …   ← leaf: what gets embedded
```

The embedded string for that leaf is:

```
Document > Part 1 Employment Rights > Zero hours workers, etc > s.1 Right to guaranteed hours > (1)
An employer must make a guaranteed hours offer …
```

### 9.2 Parser requirements (`document/parser.py`)

Input: PDF. Use `pdftotext -layout` — indentation is a structural signal and you need it preserved.

Detect, in this precedence order:

| Marker | Typical form | Rule |
|---|---|---|
| Part | `PART 1` alone on a line, ALL CAPS | Next caps line is its title |
| Schedule | `SCHEDULE 1` | May contain its own internal Parts — nest them |
| Section | `25   Right not to be unfairly dismissed` | Number in **column 0**, ≥2 spaces, then a short heading |
| Subsection | `    (1)  An employer must …` | Indented, parenthesised digit |
| Cross-heading | `        Dismissal` | Centred, title case, **immediately followed by a section line** |

**Four traps that will bite you.** Budget time for these; they are most of the parser's difficulty.

1. **The table of contents looks exactly like the body.** Skip leading pages until the enacting text, or you will build a second, empty copy of every section.
2. **Running headers repeat on every page** (`Employment Rights Act 2025 (c. 36)`, `Part 1—Employment rights`). Filter them — but filter *case-sensitively*, or your header pattern will also swallow the real `PART 1` markers.
3. **Quoted/inserted text mimics structure.** An amending document contains blocks that look like headings but belong to a *different* document. Disambiguate by **position, not content**: a real cross-heading is immediately followed by a column-0 section line; an inserted one is followed by indented text.
4. **Inserted provisions must not be attributed to their host section.** If s.1 inserts new sections into another Act, those subsections belong to the *inserted* provision. Attributing them to s.1 makes every citation point at the wrong document. Detect them (a number with a letter suffix — `27BA` — at non-zero indent) and nest them, then render the citation to show both: `HostAct s.27BA(1) (as inserted by s.1)`.

**Validation gate — assert these before proceeding:**

```python
counts = stats(tree)
assert counts["section"]     >= 150
assert counts["subsection"]  >= 900
assert counts["part"]        >= 5
assert all(">" in n.breadcrumb() for n in tree.walk() if n.kind == "subsection")
```

### 9.3 Two collections

| Collection | One row per | Searched? | Purpose |
|---|---|---|---|
| `corpus_leaves` | subsection (or section if it has none) | **yes** | Precise matching |
| `corpus_parents` | section / inserted provision | **no** | Widening a hit into its surrounding provision |

**Leaf row:**
```python
{
  "id": f"l{ordinal:05d}::{node.citation()}",     # ordinal = stable document order
  "document": f"{node.breadcrumb()}\n{node.text}", # ← breadcrumb IS the embedded text
  "metadata": {
      "citation", "breadcrumb", "parent_id", "part", "heading",
      "section_number", "section_title", "kind", "page", "chars",
      "raw_text",          # leaf text without the breadcrumb, for prompt assembly
      "index_version",
  },
}
```

> **ID uniqueness.** Citation + page is *not* unique — a section's `(3)` and an inserted provision's `(3)` can share a page. Include a monotonic ordinal. It is stable across runs because tree order is deterministic, so ids remain diffable between builds.

> **Chroma metadata values must be primitives** (`str | int | float | bool`). No lists, no dicts. Serialise or flatten.

### 9.4 Index versioning

```python
index_version = f"{PARSER_VERSION}+{embedder_name}+{sha256(corpus_bytes)[:8]}"
```

Write it into every row's metadata, the collection metadata, and a `runs/index_manifest.json`. It then appears on every metrics row and every audit record.

### 9.5 Acceptance

- [ ] Tree validation gate passes.
- [ ] Leaf count is within 10% of the subsection count.
- [ ] Every leaf's embedded document starts with its breadcrumb.
- [ ] `index_manifest.json` written with version, embedder, counts, build time.
- [ ] Rebuilding twice produces **identical ids** (determinism check).

---

## 10. Milestone 3 — the embedder, with an offline fallback

### 10.1 Requirement

Two implementations behind one interface:

1. **Preferred:** Chroma's default ONNX MiniLM (`all-MiniLM-L6-v2`). Downloads once, no API key.
2. **Fallback:** a deterministic hashed bag-of-words embedder, pure Python, no network.

```python
class HashingEmbedder:
    DIM = 512
    @staticmethod
    def name() -> str: return "hashing-bow-512"
    def get_config(self) -> dict: ...
    @staticmethod
    def build_from_config(cfg) -> "HashingEmbedder": ...
    def embed_documents(self, input) -> list[list[float]]: ...
    def embed_query(self, input) -> list[list[float]]: ...
    def __call__(self, input) -> list[list[float]]: ...
```

Implementation: tokenise, drop stopwords, hash each token **and each bigram** into a bucket, weight `1 + log(count)`, L2-normalise. Bigrams matter — they keep "guaranteed hours" distinguishable from "hours" and "guaranteed" separately.

Be honest in the docstring about what it is: **lexical, not semantic**. It works on statutes because they repeat their own vocabulary and because breadcrumbs are full of exact terms. It would be a poor choice for paraphrase-heavy support tickets.

### 10.2 The pinning rule — do not skip this

> **The retriever must use the embedder recorded in the manifest, and refuse to start if it cannot.**

```python
recorded = load_manifest().get("embedding_model")
embedder, name = get_embedder(require=recorded)
if recorded and recorded != name:
    raise RuntimeError(f"Index built with {recorded!r}, process resolved {name!r}")
```

Querying a hashed index with MiniLM vectors does not raise. It returns confident nonsense. This is one of the nastiest silent failures in RAG.

---

## 11. Milestone 4 — retrieval

### 11.1 Search

```python
def search(query: str, k: int = 6, where: dict | None = None,
           expand: bool = True) -> list[Doc]:
```

1. Query `corpus_leaves`, `n_results=k`, `include=["documents","metadatas","distances"]`.
2. Convert distance → similarity: `score = 1 - distance` (cosine space).
3. If `expand`, widen the top ~3 hits to their parent provision.
4. Emit a RETRIEVER span (§13).

### 11.2 Small-to-big expansion — with a size cap

```python
MAX_PARENT_CHARS = 4_000
```

If the parent is larger than the cap, **keep the leaf** and record `expand_skipped` in metadata.

> **Why this cap exists.** In an amending document a single "section" can contain an entire inserted chapter — tens of thousands of characters. Expanding into it silently consumes the whole context budget. Expansion is an optimisation, not an obligation.

### 11.3 Context assembly — must degrade gracefully

```python
def format_context(docs, budget_chars=6_000, min_block_chars=400) -> str:
```

De-duplicate by citation, prefix each block with `[citation] breadcrumb`, and **truncate an oversized block rather than skipping it**.

> **Failure to avoid:** a naive implementation `break`s on the first block that does not fit. If the top hit happens to be a 36,000-character provision, you produce an **empty context** and the model answers from nothing — with no error anywhere.

### 11.4 The sufficiency gate — hybrid, on purpose

```python
def sufficiency(docs: list[Doc], question: str, top: int = 3) -> float:
    similarity = mean(top-3 scores)
    terms      = distinctive_words(question)      # len ≥ 5, minus a generic stoplist
    coverage   = |terms ∩ words(retrieved text + breadcrumbs)| / |terms|
    return 0.35 * similarity + 0.65 * coverage
```

Two signals because neither is trustworthy alone:

- **Similarity** is poorly calibrated across queries — a 0.31 on one question means something different from a 0.31 on another — so it can never be the whole gate.
- **Coverage** is crude and lexical and *decisive*: "cryptocurrency" is simply not in an employment statute, and no embedding score should hide that.

Weighted toward coverage deliberately. Default threshold ≈ `0.45`, expressed as configuration, not a constant buried in code.

**Score the ORIGINAL question, never a rewritten one.** A refined query can retrieve beautifully for itself and still fail the user.

### 11.5 Acceptance

- [ ] In-scope questions score above threshold; out-of-scope questions score below.
- [ ] Expansion never produces a context longer than the budget.
- [ ] `format_context` never returns `""` when `docs` is non-empty.

---

## 12. Milestone 5 — the LangGraph workflow

### 12.1 Shape

```
classify → retrieve → assess ─┬─ sufficient ──────────→ generate → score → END
              ↑               ├─ thin, attempts left ─→ refine ──┘
              └───────────────┘
                              └─ thin, none left ─────→ refuse → END
```

### 12.2 State

```python
class AgentState(TypedDict, total=False):
    question: str
    session_id: str
    rewritten_query: str
    intent: str
    docs: list[dict]          # PLAIN dicts
    context: str
    answer: str
    attempts: int
    sufficiency: float
    refused: bool
    scores: dict[str, float]
    stage_ms: dict[str, float]
    llm_stats: dict[str, float]
```

> **Only plain types in state.** The checkpointer serialises it. Dataclasses produce deprecation warnings now and hard failures later — and a checkpoint you cannot read from another process is a checkpoint you cannot debug.

### 12.3 Nodes

| Node | LLM call? | Responsibility |
|---|---|---|
| `classify` | **no** | Keyword-based intent label |
| `retrieve` | no | Search + assemble context + score sufficiency |
| `assess` | **no** | Record the gate decision as a span |
| `refine` | no | Vocabulary expansion using the corpus's own terms |
| `generate` | yes | Answer from context, with citations |
| `refuse` | no | Decline, stating the score and the threshold |
| `score` | optional | Inline quality scoring |

> **Deliberately not LLM calls.** Intent classification is keyword-based; the sufficiency gate is arithmetic. Putting a model in every box is the standard way to make an agent expensive and non-deterministic for no measurable gain. Say this in the code comments — students will otherwise assume the opposite.

### 12.4 Assembly

```python
builder = StateGraph(AgentState)
# add_node for each …
builder.add_edge(START, "classify")
builder.add_edge("classify", "retrieve")
builder.add_edge("retrieve", "assess")
builder.add_conditional_edges("assess", route_after_assess,
                              {"generate": "generate", "refine": "refine",
                               "refuse": "refuse"})
builder.add_edge("refine", "retrieve")
builder.add_edge("generate", "score")
builder.add_edge("score", END)
builder.add_edge("refuse", END)
graph = builder.compile(checkpointer=MemorySaver())
```

### 12.5 The invocation trap

> **Use a unique `thread_id` per request, and initialise every state field explicitly.**

```python
initial: AgentState = {
    "question": q, "rewritten_query": q, "attempts": 0, "docs": [],
    "context": "", "answer": "", "sufficiency": 0.0, "refused": False,
    "scores": {}, "stage_ms": {}, "llm_stats": {},
}
state = graph.invoke(initial, config={
    "configurable": {"thread_id": f"{session_id}:{request_id}"},
    "run_name": "rag-agent",
    "metadata": {"session_id": session_id, "prompt_version": PROMPT_VERSION},
})
```

Reusing one `thread_id` across questions makes the checkpointer replay prior state: `attempts` accumulates, the previous rewrite leaks into the next question, and everything refuses after the third query. It is invisible in logs and instantly obvious in a trace — which makes it a *great* thing to demo deliberately, and a miserable thing to hit by accident.

---

## 13. Milestone 6 — Phoenix tracing

### 13.1 Bootstrap — must never raise

```python
def init_telemetry(project_name: str, endpoint: str) -> bool:
    try:
        from phoenix.otel import register
        provider = register(project_name=project_name,
                            endpoint=f"{endpoint}/v1/traces",
                            auto_instrument=True, batch=True)
        ...
        return True
    except Exception as exc:          # deliberate blanket catch
        _status = f"disabled ({exc})"
        return False
```

Run the UI with `uv run phoenix serve` (default `http://localhost:6006`), or set `PHOENIX_COLLECTOR_ENDPOINT`.

### 13.2 The span helper

Provide a context manager that yields a **no-op shim** when tracing is off, so call sites never branch:

```python
@contextmanager
def span(name: str, kind: str = "chain", **attributes): ...
```

### 13.3 Semantic conventions

Import from `openinference.semconv.trace`, with literal-string fallbacks if the package is absent:

| Attribute | Set on | Why it matters |
|---|---|---|
| `OPENINFERENCE_SPAN_KIND` | every span | Makes it render as RETRIEVER / LLM / EVALUATOR |
| `INPUT_VALUE` / `OUTPUT_VALUE` | every span | The trace is readable without opening code |
| `LLM_TOKEN_COUNT_PROMPT` / `_COMPLETION` | LLM span | The only honest basis for a cost number |
| `SESSION_ID` / `USER_ID` | root span | Group a conversation; find one customer's bad day |
| `retrieval.documents.N.document.{id,score,content,metadata}` | retriever span | Renders a documents table |
| `metadata.index_version` | root span | Which index produced this, months later |

> **Naming precision:** these are `openinference.span.kind` values (CHAIN, RETRIEVER, LLM, EVALUATOR). That is a *different concept* from OpenTelemetry's own `SpanKind` (SERVER, CLIENT, INTERNAL). Conflating them is a common and easily-corrected error.

OTel attributes must be primitives or homogeneous sequences — JSON-encode anything else and truncate to a few KB.

### 13.4 Latency instrumentation

Measure rather than estimate:

```
e2e ≈ TTFT + ITL × (output_tokens − 1)
```

Wrap the streaming loop: timestamp the first yielded token (**TTFT**), record the gap between every subsequent token (**ITL**), keep mean and p95. Store per-stage durations in `stage_ms`.

> The gap between measured `e2e` and that formula is **non-generation overhead** — orchestration, network, post-processing, client. On a thin pipeline it is small; on a system with three sequential model calls it is frequently the largest single term, and almost nobody measures it.

### 13.5 Acceptance

- [ ] A request produces a root span with child spans for each node.
- [ ] The retriever span renders a documents table with scores in the Phoenix UI.
- [ ] **Stop Phoenix, re-run: the agent still answers.** Non-negotiable.

---

## 14. Milestone 7 — evaluation

### 14.1 Golden dataset (`evals/golden.jsonl`)

~30 rows. Design rules:

```json
{"id":"g13","question":"What does section 18 say about bereavement leave?",
 "intent":"leave","must_cite":["s.18"],"should_refuse":false}
```

- **Assert citations, not prose.** Wording changes when models change; the source does not. Asserting exact text produces a suite that breaks on every improvement, and a suite that cries wolf gets deleted.
- **Stratify by intent** so a fix to one topic cannot mask a break in another.
- **Include out-of-scope cases** with `should_refuse: true`. Refusing correctly is a behaviour worth testing.
- **Phrase questions as "what does the document say"**, not "am I entitled to". The system reports what a source says; it does not advise. Keeping the golden set in that register stops the suite quietly asserting a claim about the world.
- **Keep a `known_failure: true` flag.** The gate asserts the list does not *grow* — and fails if a known failure starts passing, so the marker gets removed.

### 14.2 The RAG triad (`judges.py`)

| Metric | Question | Blames |
|---|---|---|
| Context relevance | Did we retrieve material capable of answering? | Retrieval |
| Groundedness | Is every claim supported by that material? | Generation |
| Answer relevance | Does it address what was asked? | Question ↔ answer alignment |
| Citation coverage | Is each claim attributable? | Attribution |

Implement **two** judges behind one interface:

- `HeuristicJudge` — lexical overlap. Deterministic, offline, cheap, weak. Runs in CI on every commit and never flakes.
- `LLMJudge` — a model reading a rubric. Stronger, slower, biased, non-deterministic. Runs on a sample.

The rubric must explicitly instruct the judge to **ignore length and formatting**.

### 14.3 Calibration — the part most projects skip

```python
def cohens_kappa(human: Sequence[int], machine: Sequence[int]) -> float: ...
def calibrate(judge, labelled_rows, scorer, threshold=0.7) -> Calibration: ...
```

`evals/calibration.jsonl` holds ~16 hand-labelled `(question, context, answer, human_label)` rows. **Deliberately include ~4 hard cases:**

1. A correct **paraphrase** using none of the context's vocabulary → a lexical judge scores it 0. A false negative, and a structural one.
2. **Partially** grounded — one supported claim, one invented.
3. Accurate but **uncited**.
4. Grounded first sentence, plausible-sounding unsupported second sentence.

> **Expected demo behaviour:** clean examples alone yield a near-perfect kappa. Adding the hard cases drops it materially. **The judge did not get worse — the test got honest.** Kappa corrects for chance: if 90% of answers are good, a judge that says "good" every time scores 90% agreement and is worthless.

### 14.4 Phoenix datasets and experiments

```python
import phoenix as px
from phoenix.experiments import run_experiment

client  = px.Client()
dataset = client.upload_dataset(dataframe=df, dataset_name=f"golden-{index_version}",
                                input_keys=["question"], output_keys=["expected"])
run_experiment(dataset=dataset, task=run_agent, evaluators=[...])
```

Tag each experiment with `index_version` + `prompt_version` so two runs are comparable.

> **Wrap this in try/except and fall back to a local summary.** The Phoenix experiments API surface moves between releases; the lesson survives without the UI, and a broken import must not break the demo.

---

## 15. Milestone 8 — the CI gate

Two suites, deliberately separated.

### 15.1 `test_deterministic.py` — this blocks the merge

Structural assertions only. **Nothing here asks a model for an opinion — which is exactly why it is allowed to fail a build.**

- Index manifest complete; leaf count sane.
- Retriever embedder matches the manifest.
- Tree shape gate (§9.2).
- Every leaf has a resolvable breadcrumb and citation.
- Out-of-scope questions refuse.
- Known-failure list has not grown, and none has silently started passing.
- Expected citations retrieved.
- Every answer carries a citation **or** is a refusal that states its score and threshold.
- Every request records `index_version`, `model`, tokens, cost, TTFT.
- `e2e` ≥ sum of timed stages.

### 15.2 `test_quality.py` — aggregate thresholds

Model output is a distribution. Asserting every answer clears a bar produces a flaky suite that gets deleted.

- Mean groundedness ≥ threshold.
- **p10** groundedness ≥ floor — the mean stays green while one answer in ten is unsupported.
- Mean citation coverage ≥ threshold.
- Mean context relevance ≥ threshold.
- **Judge kappa ≥ 0.6.** Gate the instrument before you gate with it; if this fails, every other threshold is measuring nothing.

**Setting thresholds:** observe a green build, then set gates *below* the observed values — enough room for ordinary variance, none for a regression. Ratchet upward as the system improves; **never** downward to fix a red build.

### 15.3 CI workflow

```yaml
- run: sudo apt-get install -y poppler-utils
- run: uv sync --group dev
- run: uv run s5-ingest --no-onnx        # rebuild the index — do not commit it
- run: uv run pytest evals/test_deterministic.py -q
- run: uv run pytest evals/test_quality.py -q
```

> Rebuilding the index in CI means a change to the chunker is actually exercised. Committing a prebuilt index means your eval tests a stale artefact and passes while the thing you changed is broken.

---

## 16. Milestone 9 — the demo runner

One command, one browser tab.

```bash
uv run s5-demo        # → http://localhost:8000
```

**Implementation constraint: standard library only** (`http.server` + one static HTML page). No Flask, no FastAPI, no build step, no npm. The worst moment in a live demo is a dependency that resolved differently that morning.

| Region | Contents |
|---|---|
| Ask box | Question input; answer with citations; chips for TTFT, ITL, e2e, cost, tokens, sufficiency, scores; a stage bar |
| Latency panel | p50 / p95 / p99 for TTFT, ITL, e2e — **and no mean anywhere** |
| Quality panel | Groundedness and citation coverage, each with its **p10** |
| Cost panel | Per request, per conversation, monthly projection, component breakdown |
| Controls | Baseline traffic · degraded-fallback traffic · CI gate · calibrate judge · shift intents · drift report · reprice · reset |
| Output console | Result of the last job |

Requirements:

- Long jobs run on a **background thread**; the UI polls a `/api/state` snapshot. It must never block.
- Job exceptions surface **in the Output panel**, not in a terminal nobody is projecting.
- The dashboard imports the same modules as the tests. No demo-only code path.

### 16.1 The degraded mode

Provide a flag that makes the model client answer **worse and slower**: pick lower-ranked evidence and drop citations. This reproduces the signature of a primary model failing over to a weaker fallback — latency up and citation coverage down in the same window.

> **Be honest about what it demonstrates.** Two correlated series tell you *when* something changed and that the symptoms share a cause. They do **not** prove which model answered — the trace does. Say so.

---

## 17. Configuration

All defaults must work with no `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `S5_MODEL` | `stub-local` | `stub-local`, or a hosted model id |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | unset | Optional |
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://localhost:6006` | Tracing target |
| `S5_TOP_K` | `6` | Leaves retrieved |
| `S5_SUFFICIENCY` | `0.45` | Refusal gate |
| `S5_MAX_ATTEMPTS` | `2` | Refine retries |
| `S5_RUNS_DIR` | `./runs` | Artefacts |

> **Provide `S5_RUNS_DIR`.** Chroma stores its index in SQLite, which fails with `disk I/O error` on some network and virtualised mounts. Pointing at local disk is the fix, and you want that documented *before* someone hits it live.

### 17.1 Pricing table

Keep model prices in **one dated table** with a `PRICING_AS_OF` constant, and derive every cost figure from it. Change the table and the whole model moves — that property is the difference between a cost model and a spreadsheet.

Two ratios worth naming because they survive any price change: **output is typically several times input**, and **cached input is roughly a tenth**. Those are what make brevity and prompt layout cost levers.

---

## 18. Acceptance criteria

The project is done when, on a clean machine with the network disabled:

```bash
git clone … && cd coderco-session5
uv sync --group dev
uv run s5-ingest --no-onnx
uv run s5-ask "…a question your corpus answers…"
uv run s5-compare
uv run pytest evals/ -q
uv run s5-demo
```

- [ ] Every command succeeds with **no API key and no network**.
- [ ] Ingest completes in < 30s and prints an `index_version`.
- [ ] `s5-ask` prints an answer **with citations**, plus TTFT, ITL, e2e, stage breakdown and cost.
- [ ] An out-of-scope question is **refused**, stating score and threshold.
- [ ] `s5-compare` shows the hierarchical index returning citable provisions where the simple index returns uncitable windows.
- [ ] `pytest evals/` passes; every assertion is deterministic.
- [ ] The dashboard opens, all buttons work, no button blocks the UI.
- [ ] Killing Phoenix mid-demo changes nothing except that spans stop appearing.
- [ ] Two consecutive ingests produce identical chunk ids.

---

## 19. Known pitfalls — checklist

Ordered by how much time they cost when missed.

| # | Pitfall | Symptom | Fix |
|---|---|---|---|
| 1 | Embedder mismatch between ingest and query | Confident nonsense, no error | Pin to the manifest; refuse to start on mismatch (§10.2) |
| 2 | Reused LangGraph `thread_id` | Everything refuses after a few queries | Unique thread per request; initialise all state (§12.5) |
| 3 | Context budget `break`s on an oversized block | Empty context, model answers from nothing | Truncate, don't skip (§11.3) |
| 4 | Unbounded parent expansion | One provision eats the whole budget | `MAX_PARENT_CHARS` (§11.2) |
| 5 | Table of contents parsed as body | Duplicate empty sections | Skip front matter (§9.2) |
| 6 | Case-insensitive header filter | Parts silently vanish | Match case-sensitively (§9.2) |
| 7 | Inserted provisions attributed to host | Every citation points at the wrong document | Nest and render both (§9.2) |
| 8 | Non-unique chunk ids | `DuplicateIDError` on ingest | Add a document-order ordinal (§9.3) |
| 9 | Dataclasses in graph state | Serialisation warnings, later failures | Plain types only (§12.2) |
| 10 | Chroma custom embedder missing `embed_query` | `AttributeError` at query time | Implement the full protocol (§10.1) |
| 11 | Sufficiency scored on the rewritten query | Refinement hides its own failure | Always score the original (§11.4) |
| 12 | Python 3.10 | `onnxruntime` has no wheel | Require ≥ 3.11 (§3.1) |

---

## 20. Suggested build order

| Day | Milestone | Verify by |
|---|---|---|
| 1 | §7 contracts, §8 simple pipeline | Simple index builds and queries |
| 1 | §10 embedder + fallback | Works offline; pinning enforced |
| 2 | §9 parser + hierarchical pipeline | Tree validation gate passes |
| 2 | §11 retrieval + sufficiency | In-scope pass, out-of-scope refuse |
| 3 | §12 LangGraph workflow | End-to-end answer with citations |
| 3 | §13 Phoenix tracing | Span tree visible; app survives Phoenix dying |
| 4 | §14 golden set + judges + calibration | Kappa reported; hard cases drop it |
| 4 | §15 CI gate | Both suites green, no flakes over 10 runs |
| 5 | §16 demo runner | Full run-through with the network off |
| 5 | §18 acceptance | Clean-machine rehearsal |

Build the simple pipeline first even though it is not the final design. It gives you a working end-to-end path within an hour, and it is the control that makes the hierarchical result meaningful.

---

## 21. Extension exercises

For attendees who finish early, or as follow-up homework.

1. **Add BM25** alongside the vector search and fuse with Reciprocal Rank Fusion. Measure the change on the golden set before keeping it.
2. **Swap the embedder** to MiniLM and re-run the comparison harness. Which questions change rank, and why?
3. **Break the parser on purpose** — remove the inserted-provision rule — and watch which eval assertions catch it. Some will; note which do not.
4. **Add a second corpus** and a routing node that picks between indexes. Now `index_version` on every row starts earning its keep.
5. **Run the judge with a hosted model** and compare its kappa against the heuristic judge on the same calibration set.
6. **Add a cost budget** to the agent: refuse or downgrade the model when a conversation exceeds a threshold.
