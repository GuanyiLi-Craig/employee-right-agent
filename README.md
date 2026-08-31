# Employment rights agent

[![eval](https://github.com/GuanyiLi-Craig/employee-right-agent/actions/workflows/eval.yml/badge.svg)](https://github.com/GuanyiLi-Craig/employee-right-agent/actions/workflows/eval.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A **chat agent** over a hierarchically structured legal document, instrumented
end to end. The retrieval pipeline is the vehicle; **the observability,
evaluation and audit layers are the subject**. The question it answers is not
"can an LLM answer this" but *how do you know it works, what does it cost, and
can you prove what it did.*

Four questions, and most teams instrument only the first:

| | | |
|---|---|---|
| **Is it up?** | latency, errors, throughput | solved — and it stays green through every failure here |
| **Is it any good?** | groundedness, citations, refusals | percentiles and a p10, not a mean |
| **What does it cost?** | five components, one dated table | tokens are not the whole bill |
| **Can you prove it?** | who asked, what it read, how the answer was produced | a hash-chained audit record |

Runs with **no API key and no network**. Both the model and the embedder have
offline, deterministic fallbacks, because a demo that needs conference wifi is a
demo you cannot give.

```
                     ┌─────────────┐
   the Act (a PDF or │  ingest     │  ← a separate job. The only writer.
   layout text file) │  pipeline   │
                     └──────┬──────┘
                            │ chroma + index_manifest.json  (named volume)
                            ▼
  ┌──────────┐   ┌──────────────────────────────────────────┐   ┌──────────┐
  │  chat +  │──▶│ classify → retrieve → assess ─┬─ generate│──▶│ Phoenix  │
  │dashboard │   │              ▲                ├─ refine ─┘   │  :6006   │
  │  :8000   │◀──│              └────────────────┴─ refuse      │  traces  │
  └──────────┘   └──────────────────────────────────────────┘   └──────────┘
                            │                    │
       one JSON line ◀──────┘                    └──────▶ one hash-chained
    runs/metrics.jsonl                              record  runs/audit.jsonl
```

---

## Quick start — Docker Compose

The embedding pipeline is a **job, not a service**. It is the only writer of the
Chroma index, it runs to completion and exits, and the query service never
starts one implicitly. Re-indexing is therefore always a deliberate act with a
recorded `index_version`, and a bad index can never be half-published by a
restart loop.

```bash
docker compose build                    # build the image
docker compose run --rm ingest          # 1. the embedding pipeline (required)
docker compose run --rm ingest-simple   # 2. the baseline index    (optional)
docker compose up -d                    # 3. phoenix + dashboard

open http://localhost:8000              # dashboard
open http://localhost:6006              # traces
```

**Ports already in use?** Every published port is overridable. Put them in
`.env` so every `docker compose` command in the project agrees:

```bash
printf 'DASHBOARD_PORT=8010\nPHOENIX_PORT=6016\nPHOENIX_GRPC_PORT=4327\n' >> .env
```

Other jobs, all read-only with respect to the index:

```bash
docker compose run --rm ask "What does the document say about bereavement leave?"
docker compose run --rm compare         # fixed windows vs. the hierarchical index
docker compose run --rm evaluate        # the CI gate's numbers
docker compose run --rm evals           # the CI gate itself (pytest)
docker compose run --rm corpus          # regenerate data/corpus.layout.txt
docker compose down                     # stop; the index volume survives
docker compose down -v                  # stop and discard the index
```

If you start the dashboard before the index exists it waits
(`RIGHTS_WAIT_FOR_INDEX`, default 90s in compose) and then exits with the exact
command that fixes it.

## Quick start — local

```bash
uv sync --extra trace --extra models --group dev   # models: only needed for a hosted provider
uv run rights-ingest --no-onnx                       # the embedding pipeline
uv run rights-ask "What does the document say about bereavement leave?"
uv run rights-ask "How do I mine cryptocurrency on company laptops?"   # refused
uv run rights-compare
uv run pytest -q
uv run rights-demo                                   # http://127.0.0.1:8000
```

Every command is `uv run …`. Never activate a venv by hand — that is where
"works on my machine" comes from. `make help` lists shortcuts.

Optional, for tracing locally: `uv run phoenix serve` (or the compose service),
then `PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006`.

---

## What each piece is for

| Module | Responsibility | The thing worth reading it for |
|---|---|---|
| [`document/parser.py`](src/rights_agent/document/parser.py) | layout text → `Node` tree | the four traps that silently produce a plausible, wrong tree |
| [`document/nodes.py`](src/rights_agent/document/nodes.py) | the tree, `breadcrumb()`, `citation()` | why an inserted provision must never be cited as one of ours |
| [`pipelines/simple.py`](src/rights_agent/pipelines/simple.py) | fixed-window baseline | what it *cannot* do: no chunk can carry a citation |
| [`pipelines/hierarchical.py`](src/rights_agent/pipelines/hierarchical.py) | tree → two collections | the breadcrumb is *embedded*, not stored beside |
| [`embedding.py`](src/rights_agent/embedding.py) | MiniLM, and an offline hashed fallback | the pinning rule: a cross-embedder query returns confident nonsense |
| [`store.py`](src/rights_agent/store.py) | Chroma access, the manifest | `get_collection()` silently substitutes a default embedder |
| [`retrieval.py`](src/rights_agent/retrieval.py) | search, small-to-big, sufficiency | truncate, never skip; score the *original* question |
| [`llm.py`](src/rights_agent/llm.py) | streaming clients (stub, DeepSeek, OpenAI, Anthropic), measured TTFT/ITL | why `e2e ≈ TTFT + ITL×(n−1)` is an approximation on both sides; how a provider default buys you reasoning you did not ask for |
| [`graph.py`](src/rights_agent/graph.py) | the LangGraph workflow | why most nodes deliberately make no model call |
| [`agent.py`](src/rights_agent/agent.py) | one request, one row, one root span | the reused-`thread_id` trap; recording which model *actually* served |
| [`telemetry.py`](src/rights_agent/telemetry.py) | Phoenix bootstrap, span helpers | observability that cannot break the app |
| [`metrics.py`](src/rights_agent/metrics.py) | `runs/metrics.jsonl` | percentiles for latency, p10 for quality, never a mean for latency |
| [`judges.py`](src/rights_agent/judges.py) | the RAG triad, Cohen's kappa | gate the instrument before you gate with it |
| [`analysis.py`](src/rights_agent/analysis.py) | drift, PSI, three kinds of drift | why one PSI figure quotes your smoothing constant |
| [`costs.py`](src/rights_agent/costs.py) | the five-component cost model | tokens are not the whole bill, and the judge is on the list |
| [`audit.py`](src/rights_agent/audit.py) | hash-chained audit record | tamper-**evident**, not tamper-proof — and what that costs you |
| [`conversation.py`](src/rights_agent/conversation.py) | transcripts, history, follow-up resolution | resolving a follow-up without a model call; two tiers of history |
| [`demo/`](src/rights_agent/demo/) | the chat UI and the dashboard | standard library only; no build step |

## The corpus

`data/ukpga_20250036_en.pdf` is the real **Employment Rights Act 2025 (c. 36)**,
335 pages: 24 Parts, 12 Schedules, 459 sections and schedule paragraphs, 87
provisions inserted into *other* Acts, 1,971 subsections, 2,078 citable leaves.
It is what the demo serves, what the committed eval datasets were generated
against, and what CI gates. `RIGHTS_CORPUS` selects it (see `.env.example`).

`data/corpus.layout.txt` is a **generated** Act that stays as the offline
fallback and as the fixture the parser unit tests are written against — 6 Parts,
167 sections, ~1,100 subsections, 109 pages, containing all four parser traps on
purpose, and reproducible byte-for-byte so `index_version` means something.

> The generated Act is not a lesser copy of the real one; it is the control. A
> parser written entirely against it passed its own tests and then failed
> silently on real legislation in seven distinct ways — a smaller tree, never an
> error. [`data/README.md`](data/README.md) lists each one, what it cost, and the
> measurements before and after. That table is the most useful thing in this
> repository if you are about to point a parser at a document you did not write.

Switching corpus invalidates the eval datasets, because an expected citation
names a provision and a provision exists in one document. `evals/datasets/<embedder>/baseline.json`
records which corpus it was built for and the gate refuses to run against
another — one legible failure instead of twenty that look like a retrieval
regression.

## The embedder

Three, behind one interface, measured on this corpus — recall of the golden
set's 30 expected citations:

| embedder | recall | query p50 | ingest | needs |
|---|---|---|---|---|
| `openai-text-embedding-3-small` | **100.0%** | 143 ms | ~17 s | a key, ~1¢ |
| `onnx-all-MiniLM-L6-v2` | 90.0% | 53 ms | 60 s | an 80 MB model (baked into the image) |
| `hashing-bow-512` | 86.7% | 2 ms | 3 s | nothing |

`RIGHTS_EMBEDDER` selects one. The demo runs `openai`; **CI runs `hashing`**,
because it is the only one of the three that is bit-deterministic — MiniLM's
floats move with thread count, and the API does not promise stable vectors
across calls — and a gate whose vectors shift under it fails for reasons
unrelated to the change under test.

That split is why `evals/datasets/` has **one directory per embedder**. Which
golden rows are `known_failure` is a property of the retrieval config, not of
the corpus: six rows fail on the bag-of-words that pass on the API embedder.
A single shared golden set would be wrong for two of the three, and wrong in the
direction that looks like a regression in whichever one did not generate it.
`evals/datasets/<embedder>/baseline.json` stamps the full `index_version` it was
generated against, and the gate refuses to run against a different one.

The 17-point spread is the argument for measuring retrieval separately from
generation. None of it is visible in the answer — which reads fluently either
way — and all of it is visible in whether the right provision was in the context
at all.

## The chat interface

The dashboard leads with a chat, because the numbers only mean something next to
the answer they describe. Each turn streams (so TTFT is something you *watch*,
not a figure reported afterwards) and carries its own chips: TTFT, ITL, e2e,
tokens, all-in cost, sufficiency, the judged scores, its intent, and its audit
record number. Expand *how this answer was produced* for the stage bar, the
retrieved provisions with scores, and the hash of the audit record.

Follow-ups work:

```
you    What does the document say about bereavement leave?
agent  [s.19] provides: … [s.20] provides: …
you    How long is it?
agent  [ERA 1996 s.80EB (as inserted by s.23)] provides: …        follow-up resolved
```

A follow-up borrows the distinctive terms from your previous question —
deterministically, with **no model call**, for the same reason intent
classification has none: a rewrite that changes when you re-run it makes every
downstream measurement unreproducible.

Chains work too — a follow-up borrows from the last question that *was not
itself* a follow-up, so "how long is it?" then "and for agency workers?" both
stay on bereavement leave rather than the second one borrowing "long" from the
first.

### History

The **Conversations** panel lists past chats, newest first, and clicking one
reopens it. Your active session is remembered in `localStorage`, so a page
reload puts you back in the conversation you were having.

History has two tiers, and the panel labels which one a conversation came from:

| badge | source | what you get |
|---|---|---|
| `live` | the in-memory transcript | the full conversation, answers included |
| `audit` | the hash-chained audit record | the questions, citations, refusals, cost — **not the answer prose** |

That split is the design, not a shortcoming. Transcripts are held in memory on
purpose: the durable record of a request is the audit log, with redaction applied
at capture, and adding a second durable copy of every model output would mean a
larger personal-data store under weaker retention rules than the trace already
has. So after a restart you can still see *what was asked, what it cited and what
it cost* — and a reconstructed answer bubble says plainly that the prose was never
kept, rather than rendering empty and letting you assume the model said nothing.

Reopening a conversation restores its questions to the working transcript, so
follow-ups resolve again from there. Synthetic traffic from the controls and the
golden set are audited like everything else but never listed as conversations —
a history swamped by `eval-g029` is a history nobody reads.

Two honest limits, both visible in the trace:

- It resolves **topic**, not reference. It carries "bereavement leave" forward so
  the retriever has something to match; it does not know that "that" meant the
  notice period.
- The gate scores the **resolved** question for a follow-up, and the *original*
  for everything else. Those borrowed words came from the user one turn earlier,
  so they are part of the question as asked. A query the *system* invented during
  refinement is still never scored — that rule is what stops refinement talking
  its way past the gate, and there is a test for each half.

## The audit record

One hash-chained record per decision, in [`runs/audit.jsonl`](src/rights_agent/audit.py):
**who** asked (actor, tenant, role, lawful basis), **what** happened (the
question, answered or refused, the citations), and **how** the answer was
produced (index and document versions, model, prompt version, the sufficiency
score against its threshold).

```bash
docker compose exec dashboard python -m rights_agent evaluate --gate  # includes chain integrity
curl -s localhost:8000/api/audit?limit=1 | jq          # or read it over the API
```

Four design points worth the code read:

- **Sources are id + version + hash, never a copy of the text.** An id and a
  hash prove a source *changed*; the version is what lets you reconstruct what it
  said. If your corpus is not versioned, hashing gives you detection without
  recovery — worth knowing before an audit rather than during one.
- **Redaction happens at capture.** A redaction applied when the UI renders has
  already been exported by anyone with an API key. The record keeps a SHA-256 of
  the *original* question, so identity survives redaction.
- **Tamper-evident, not tamper-proof.** Each record carries the hash of the one
  before it, so an edit cannot be made *silently* between the decision and the
  audit — press **Tamper with the audit log** and watch it name record 0. Someone
  who can rewrite the whole store can recompute every subsequent hash; there is a
  test asserting exactly that. The fix is to anchor outside the store, which is
  what `write_checkpoint` demonstrates.
- **Retention is two boundaries, and neither is Article 12.** Article 12 is a
  *logging-capability* duty. The floor is AI Act Articles 19 and 26(6) —
  appropriate to purpose, generally at least six months. The ceiling is GDPR's
  storage-limitation principle, which sets no universal maximum but requires you
  to justify the period. `RIGHTS_RETENTION_DAYS` below the floor is refused at
  startup, with the basis in the error message.

> The field list is **an engineering design, not a legal schema**, and this
> repository is not legal advice. Certain employment uses are *listed* in Annex
> III; where such a system is classified high-risk the obligations apply, and
> Article 6(3) allows a documented assessment that a listed system does not pose
> a significant risk. "All employment AI is high-risk" is wrong.

## Running a real model

The default is `stub-local`: offline, deterministic, extractive, and what CI
uses. Point `RIGHTS_MODEL` at a hosted model to swap it in — the graph, the
gate, the audit record and the cost model are unchanged.

```bash
printf 'RIGHTS_MODEL=deepseek-v4-flash\nDEEPSEEK_API_KEY=sk-…\n' >> .env
docker compose up -d --force-recreate dashboard
```

DeepSeek speaks the OpenAI protocol, so `DeepSeekClient` subclasses the OpenAI
one with a different `base_url`. Two provider details it has to get right,
because both fail quietly:

- **`thinking` defaults to `enabled`.** Omitting it buys reasoning tokens on
  every request and a far slower first token, without asking. The mode is always
  sent explicitly, in both directions. `RIGHTS_THINKING=true` turns it on
  deliberately.
- **Reasoning arrives in its own field** (`reasoning_content` on the streaming
  delta). It is excluded from the answer and **counted**, so an answer can never
  quietly contain chain-of-thought — and if any arrives while thinking is off,
  the count says so rather than the text appearing.

Cache accounting differs too: DeepSeek reports `prompt_cache_hit_tokens`
alongside the OpenAI-compatible field, and cache-hit input is about **1/31** of
cache-miss rather than the usual tenth. On this model prompt layout is *the*
cost lever rather than one of several — visible in the cost panel, where a
well-cached prompt moves the dominant component from input to output.

A missing key falls back to the stub rather than failing the request — that
keeps the offline guarantee. But a fallback is never silent:

- the log says `falling back to stub-local: DEEPSEEK_API_KEY is not set`
- the metrics row and the audit record carry **both** `model` (what served) and
  `requested_model` (what you asked for), plus a `fallback` flag
- the chat shows a red `deepseek-v4-flash → stub-local` chip on the message
- the header greys the model name out, because `/api/state` reports
  `model_available: false` — checked once rather than discovered per request

That matters more than it sounds. Recording only the configured model is exactly
how a silent failover becomes undetectable: the graphs move, the trace is the
only place the truth survives, and the row you would check agrees with the lie.

| model | input | cached input | output |
|---|---|---|---|
| `deepseek-v4-flash` | $0.44 | $0.014 | $1.32 |
| `deepseek-v4-pro` | $1.32 | $0.044 | $3.96 |

Per 1M tokens, peak rate, from DeepSeek's published list as of the
`PRICING_AS_OF` in [`config.py`](src/rights_agent/config.py). **Off-peak is half
of peak** (peak is 01:00–04:00 and 06:00–10:00 UTC on weekdays), so
`deepseek-v4-flash-offpeak` exists as a pricing row you can reprice against —
the same model, the same traffic, and the only variable is *when you ran it*.
It is not a model id the API accepts, and `make_client` says so rather than
letting you find out from a 400.

## Cost: five components, one table

A cost model that stops at model tokens sends teams optimising the wrong term.

| component | what moves it |
|---|---|
| `generation_input` | the retrieved context you chose to send — every call, forever. **Dominant in this workload**; top-k and chunk size are usually a bigger lever than the model. |
| `generation_output` | dearer per token than input on many models, which makes brevity a cost control. |
| `judge` | an LLM call like any other. Sampling is about money, not statistical elegance. |
| `trace_storage` | cheap per request, large in aggregate; the multiplier is retention. |
| `infrastructure` | compute, orchestration, the vector store. |

**Reprice** puts the same recorded traffic at two prices with nothing re-run —
here the same DeepSeek model at peak and off-peak:

```
component             deepseek-v4-flash-offpeak deepseek-v4-flash
generation_input                0.001377          0.002753
generation_output               0.003696          0.007392
judge                           0.000918          0.001836
monthly projection                346.71            571.36

dominant component: generation_output — the answer you generate. The lever is
brevity. Worth checking the prompt cache is working first: output only leads
once input is discounted.
```

Which component leads is a property of the **price list** as much as the
workload, so the report looks it up rather than asserting one: with 30,720 of
36,000 prompt tokens served from cache at 1/31 the rate, input stops leading
even though the prompt is six times the answer. Uncached, the same traffic is
input-dominated.

The judge line is priced against the serving model by default, so a DeepSeek run
is not costed with a competitor's judge — and it is labelled *modelled, not
incurred*, because the offline judge makes no model call.

## The degraded fallback

`DegradedClient` wraps **any** client, so the incident demo works whichever
provider is configured — it was a no-op for hosted models until that was caught
by driving the UI. It leads with lower-ranked evidence (the context is reversed),
removes citations from the output, and adds a delay before and between chunks.

Two things worth knowing:

- **Citations are removed, not discouraged.** Asking DeepSeek not to cite did not
  work, and neither did stripping the citation from the context header — the
  breadcrumb still names the section. A capable model is hard to make careless,
  so the behaviour being simulated is simulated directly.
- **The delay is sized against a hosted baseline.** The stub answers in
  milliseconds, so any penalty looks dramatic; a real model already takes ~0.8 s,
  where a +0.6 s penalty moved TTFT p50 from 1.01 s to 1.37 s — true, and far too
  subtle to point at. It is now +1.5 s, which reproduces the doubling a real
  failover causes.

`model` still reports the real model, with a `degraded` flag beside it. Inventing
a different model name here would be the exact silent-failover confusion the
metrics row exists to prevent.

## Drift: three kinds, and the PSI gotcha

**Corpus drift** (the documents changed), **intent drift** (the questions moved),
**performance drift** (the scores fell) — kept apart because the investigations
are completely different. Re-indexing does not fix a prompt regression, and
prompt engineering does not fix a corpus that moved.

**Drift report** gives two PSI figures and a list, deliberately:

```
PSI over intents in BOTH windows     1.4640  (significant)
PSI including unseen categories      2.1331  (significant, epsilon=0.0001)
NEW intents                        harassment, unions
```

PSI divides by the baseline probability, so a category absent from the baseline
sends it to infinity; every implementation smooths with an epsilon, which keeps
the number finite and makes it **depend on that constant**. Quoting one figure
for a window containing new categories quotes your smoothing constant as much as
your data. So: the first figure for the shift among known intents
(epsilon-free, renormalised over the shared support), and the **new-intent list**
for everything else — it needs no threshold and is usually the more actionable
finding. The bands (<0.1 stable, >0.25 significant) come from credit-risk
modelling: a useful convention, not a law of nature.

Intent is classified by keyword, deliberately. An embedding classifier has better
coverage, but if you group metrics by a dimension, that dimension has to be
stable — a label that changes when you re-run it makes every time series
meaningless.

## Phoenix: traces and the golden set

Traces flow automatically once Phoenix is up — every request produces a root span
with `rag.retrieve` (RETRIEVER), `rag.generate` (LLM), `rag.judge` (EVALUATOR),
`rag.assess`, and LangGraph's own node spans beside them. Kill Phoenix and the
agent keeps answering; spans simply stop being exported.

**The judged scores are also pushed as span annotations** — `groundedness`,
`citation_coverage`, `context_relevance`, `answer_relevance` and `sufficiency`,
each with a score, a band (`good` / `fair` / `poor`) and the routing decision.
They are attributes *and* annotations on purpose: an attribute is something you
read on one span, an annotation is something a project can aggregate and filter,
so "every generated answer that scored under 0.7 for groundedness" becomes a
query rather than a script. Phoenix shows their running means in the project
sidebar.

`annotator_kind` separates the instruments: the offline judge is `CODE`, a
model-graded judge is `LLM`. A lexical overlap score and a model's opinion fail
in completely different ways, and one bucket for both would hide that.

Two things this got wrong first, both worth knowing if you copy it. Annotating
inline 404s — the batch span processor holds spans for seconds, and Phoenix
rejects feedback for a span it has not received, with an error that reads like a
bad id. And it belongs off the request path: this is a demo about latency, and an
HTTP round trip per answer would inflate the number on the panel with the cost of
reporting it. So it runs in a background thread that flushes spans first and
retries a bounded number of times. `RIGHTS_ANNOTATE_TRACES=false` turns it off;
it follows `RIGHTS_TRACING` by default.

Uploading the golden set as a Phoenix **dataset** and running it as an
**experiment** is a separate, deliberate act, because it costs money and is
non-deterministic:

```bash
make phoenix          # or: docker compose --profile tools run --rm evaluate --phoenix
```

```
uploaded 37 rows to dataset 'golden-parser-6+openai-text-embedding-3-small+bc461767' ·
experiment ran 37 tasks with 74 evaluations ·
open <your Phoenix UI>/datasets/RGF0YXNldDox/compare?experimentId=RXhwZXJpbWVudDoy
```

Every example carries `index_version` and `prompt_version` in its metadata, and
so does the experiment, because two runs are only comparable if you can tell
which index and which prompt produced them. `known_failure` travels too —
without it a Phoenix experiment silently counts the known failures as
regressions.

The whole upload is wrapped and falls back to a local summary: **the lesson
survives without the UI**, and a broken import must not break the demo. That
proved its worth — pointing it at a real server surfaced two API mismatches (flat
examples are rejected in favour of nested `input`/`output`/`metadata`, and the
URL helper needs the dataset id as well as the experiment id), and the fallback
meant the numbers still printed while both were fixed.

## Security

YAML-template scanning with Nuclei: the community library (10,689 templates,
13 `info` matches) plus five templates written for this application's own risk
surface. Findings, fixes and accepted risk are in
[`security/README.md`](security/README.md).

```bash
make pentest
```

The short version: the published ports are **loopback only**, the server refuses
to bind a network address without `RIGHTS_DEMO_TOKEN`, that token gates the
endpoints that can erase or read the audit record, session ids come from a
CSPRNG, and concurrent chats are bounded. Every control has an assertion in the
blocking suite — a control that is not asserted comes back off in the next
refactor.

The best finding came from a template catching a hole in the fix for the one
above it: `/api/audit` was protected while `/api/chat/history` reconstructed the
same records through a side door.

## Testing it yourself

[`TESTING.md`](TESTING.md) is a run-through with expected output for every step:
the Docker path, the local path, both eval suites, the dashboard demo in the
order to show it, and a **break it on purpose** section that makes each defence
fire so you can see it working.

## Evaluation

Two suites, deliberately separated.

```bash
uv run pytest evals/test_deterministic.py -q   # blocks the merge
uv run pytest evals/test_quality.py -q         # aggregate thresholds
uv run python -m rights_agent evaluate --gate         # the same numbers, reported
uv run python -m rights_agent evaluate --calibration  # judge kappa
```

`test_deterministic.py` contains **structural assertions only** — nothing in it
asks a model for an opinion, which is exactly why it is allowed to fail a build.
`test_quality.py` gates aggregates, because model output is a distribution and
asserting every answer clears a bar produces a flaky suite that gets deleted.

Thresholds live in [`evals/datasets/<embedder>/baseline.json`](evals/datasets),
set *below*
observed values on a green build. **Ratchet them upward as the system improves;
never downward to fix a red build.** The same file records which golden rows are
known to fail: the gate asserts that list does not grow, and fails if a known
failure starts passing so the marker gets removed.

Regenerate the datasets against the live index (a deliberate act):

```bash
uv run python -m rights_agent goldens --write-baseline
```

### The judge, and why kappa

`HeuristicJudge` is lexical, deterministic and honestly weak; it runs in CI on
every commit and never flakes. `LLMJudge` is stronger and worse behaved and runs
on a sample. Neither is believed before its agreement with human labels is
measured:

```
set                        n    kappa  agreement
clean examples only       12    1.000      1.000
plus hard cases           17    0.764      0.882
groundedness alone        17    0.643      0.824
```

The hard cases did not make the judge worse — they made the measurement honest.
Kappa corrects for chance: if 90% of answers are good, a judge that says "good"
every time scores 90% raw agreement and is worthless. The two remaining
disagreements are the instructive ones: a correct paraphrase using none of the
context's vocabulary (a structural false negative for any lexical judge), and
correctly-cited boilerplate that does not answer the question (perfect on
support, useless as an answer).

## Configuration

Every default works with no `.env`. See [`.env.example`](.env.example).

| Variable | Default | Purpose |
|---|---|---|
| `RIGHTS_MODEL` | `stub-local` | `stub-local`, or a hosted model id (e.g. `deepseek-v4-flash`) |
| `DEEPSEEK_API_KEY` | unset | required for a `deepseek-*` model |
| `RIGHTS_THINKING` | `false` | let a model that supports it think first |
| `RIGHTS_DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | proxy or self-hosted gateway |
| `RIGHTS_JUDGE_MODEL` | serving model | what the judge cost line is priced against |
| `RIGHTS_EMBEDDER` | `auto` (`hashing` in Docker) | `auto`, `onnx`, `hashing` — ingest only |
| `RIGHTS_TOP_K` | `6` | leaves retrieved |
| `RIGHTS_SUFFICIENCY` | `0.45` | the refusal gate |
| `RIGHTS_MAX_ATTEMPTS` | `2` | refinement retries |
| `RIGHTS_MAX_PARENT_CHARS` | `4000` | cap on small-to-big expansion |
| `RIGHTS_CONTEXT_BUDGET` | `6000` | prompt context budget |
| `RIGHTS_RUNS_DIR` | `./runs` | index, manifests, metrics |
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://localhost:6006` | tracing target |
| `RIGHTS_TRACING` | `true` | set `false` to export nothing |
| `RIGHTS_DEGRADED` | `false` | simulate a weaker fallback model (works for any provider) |
| `RIGHTS_AUDIT` | `true` | write the hash-chained audit record |
| `RIGHTS_RETENTION_DAYS` | `183` | below the Articles 19/26(6) floor is refused |
| `RIGHTS_PROJECTION_RPD` | `50000` | the monthly projection's volume assumption |
| `RIGHTS_JUDGE_SAMPLE_RATE` | `0.10` | what the cost panel's judge line is modelled at |
| `RIGHTS_PANEL_WINDOW` | `20` | requests the latency and quality panels cover |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | unset | optional |

`RIGHTS_RUNS_DIR` matters: Chroma keeps its index in SQLite, which fails with
`disk I/O error` on some network and virtualised mounts. Point it at local disk.

Costs come from one dated table with a `PRICING_AS_OF` constant
([`config.py`](src/rights_agent/config.py)); change the table and every figure,
component and projection moves with it. The offline stub costs nothing to run,
so it is priced *as if* it were a small hosted model and every surface that
reports that number says so.

## Pitfalls this codebase defends against

Each has a test naming it.

| # | Pitfall | Symptom | Defence |
|---|---|---|---|
| 1 | Embedder mismatch between ingest and query | confident nonsense, no error | readers are pinned to the manifest and refuse to start on a mismatch |
| 2 | Reused LangGraph `thread_id` | everything refuses after a few queries | unique thread per request, every state field initialised |
| 3 | Context budget `break`s on an oversized block | empty context, model answers from nothing | truncate, never skip |
| 4 | Unbounded parent expansion | one provision eats the whole budget | `RIGHTS_MAX_PARENT_CHARS`, and `expand_skipped` recorded |
| 5 | Table of contents parsed as body | duplicate empty sections | front matter skipped at the enacting formula |
| 6 | Case-insensitive header filter | Parts silently vanish | headers matched case-sensitively |
| 7 | Inserted provisions attributed to their host | every citation points at the wrong document | nested, and cited as `ERA 1996 s.27BA(1) (as inserted by s.1)` |
| 8 | Non-unique chunk ids | rows silently missing | document-order ordinal in every id, uniqueness asserted |
| 9 | Dataclasses in graph state | serialisation warnings, later failures | plain types only |
| 9b | A node writes a state key the schema does not declare | LangGraph drops the update **silently**; consumers fall back to a default and look correct | every declared key initialised, and a test scans the node bodies against the schema |
| 10 | Chroma custom embedder missing `embed_query` | `AttributeError` at query time | full 1.x protocol implemented |
| 11 | Sufficiency scored on the rewritten query | refinement hides its own failure | always score the original |
| 12 | Python 3.10 | `onnxruntime` has no wheel | `requires-python = ">=3.11"` |

Two more found while building this, both specific to chromadb 1.5.x:

- `get_collection()` **without** an explicit embedding function returns Chroma's
  *default* embedder even when the persisted configuration names another one.
  Every read goes through [`store.open_collection`](src/rights_agent/store.py),
  which passes the embedder explicitly and cross-checks the recorded name.
- Re-adding an existing id is **silently ignored** rather than raising, so a
  collision shows up as a quietly incomplete index. Ids are checked before the
  first `add`.

## Layout

```
src/rights_agent/       the package: pipelines, retrieval, graph, telemetry, demo
data/                   the committed corpus (generated, reproducible)
evals/                  datasets/<embedder>/{golden,calibration}.jsonl + baseline.json
tests/                  unit tests; no index required
docker/                 Dockerfile (runtime + dev targets) and the entrypoint
docker-compose.yml      phoenix · dashboard · ingest jobs · tools
security/               nuclei templates for this application's own risks
uitest/                 browser tests that assert on what is on screen
runs/                   generated: chroma index, manifests, metrics.jsonl,
                        audit.jsonl, audit_checkpoint.json (all git-ignored)
```

## Out of scope

Multi-tenant auth, a production web front end, fine-tuning, distributed serving,
agent tool-calling, and any cloud dependency.

## Contributing

[`CONTRIBUTING.md`](CONTRIBUTING.md) covers the setup, the two commands CI
runs, and the rules around the eval gate — chiefly that thresholds ratchet
upward and never down. Security issues go to [`SECURITY.md`](SECURITY.md)
rather than a public issue; the threat model and the accepted risks are
written down in [`security/README.md`](security/README.md).

## Licence

MIT — see [`LICENSE`](LICENSE).

`data/` holds two corpora with different provenance.
`corpus.layout.txt` is generated by this repository and carries the same
licence as the code — it is generated rather than downloaded precisely so the
default path has no licensing question attached. `ukpga_20250036_en.pdf` is the
real Employment Rights Act 2025, UK legislation published under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
