# Deck changes

The repository has only `CoderCo-Session5-Evaluation-Observability.pdf`, not the
PPTX it was exported from, so these are the edits to apply by hand. The speaker
notes are already updated; this list is what has to change *on the slides* to
match them.

Ordered by how badly it goes wrong if you skip it.

---

## Must change — the slide currently says something untrue

### Slide 1 · CODERCO · AI / MLOPS SERIES

**No change needed — the slide is now true.** The corpus is the real Employment
Rights Act 2025 (c. 36). Correct the page count if the slide says 334: the PDF
from legislation.gov.uk is **335** pages.

Numbers you can quote, all measured on the shipped index
(`parser-6+openai-text-embedding-3-small+bc461767`):

> 24 Parts · 12 Schedules · 459 sections and schedule paragraphs ·
> 87 provisions inserted into *other* Acts · 1,971 subsections ·
> 2,078 citable leaves · 546 parents · 335 pages

An earlier build of this repo shipped a *generated* Act instead, and an earlier
version of this document told you to soften the slide to match. That is no
longer the case — and the switch is worth two minutes of stage time on its own,
because the generated Act had all four parser traps in it and the parser still
broke on the real one in eight distinct ways, every one of them silently. The
table in `data/README.md` lists them. The one-line version: **a parser that
passes its own tests on a corpus you generated has told you nothing about a
corpus you did not.**

### Slide 4 · One question, fully instrumented

- `make present → type a question, press Ask` → **`click a suggestion chip, watch it stream`**
- Add **`a follow-up`** to the bullet list: *"How long is it?" — resolved against the
  previous turn, with no model call*
- The example question in any screenshot or caption must not be *"the qualifying
  period for unfair dismissal"*. This corpus names the topic but states no figure,
  so the model correctly says so — an honest answer that reads as a failure.
  Use **"What does the document say about bereavement leave?"**

### Slide 6 · The incident you cannot see from one graph

The card that reads **"groundedness barely moves"** is now model-dependent and
must not be stated as a fact on a slide. Replace with:

> citation coverage collapses **far further** than groundedness — read both deltas
> off the Output panel

On the extractive stub groundedness barely moves; on a paraphrasing model it moves
too, because a lexical judge cannot see paraphrase. The job measures both and
prints them, so the slide should point at the panel rather than pre-commit.

Add one line to the same slide, because it is the reason the demo works at all:

> the panels are **windowed** — a cumulative percentile cannot move during an incident

### Slide 12 · Evaluate the evaluator

- `+ 4 realistic ones` → **`+ 5 realistic ones`**
- `case c13` → **`case h001`** (and if there is room, add **`case h005`** — real,
  correctly-cited boilerplate that answers nothing, the failure in the opposite
  direction)
- `gating on both recovers most of it` → **`gating on both beats groundedness
  alone`**, with the three figures in order: **1.000 → 0.883 → 0.764**

  Those numbers moved once for a reason worth a sentence if anyone asks. Clean
  kappa was 0.824 for a while, and the disagreeing case turned out to be the
  judge being *right*: a calibration row whose whole answer was
  `[s.1(2)] provides: (2) of that section.` — a wrapped cross-reference the
  parser had read as a subsection. The judge scored it 0.00 and the generated
  label said "acceptable". Fixing the parser removed the row and clean kappa
  returned to 1.000. **The instrument caught a bug in the corpus, not the other
  way round**, which is the strongest available argument for gating the
  instrument first.

### Slide 14 · Drift, then the same traffic at two prices

- `Reprice · Haiku / Sonnet` → **`Reprice`** (one button plus a model dropdown)
- `Haiku vs Sonnet ~3×` → **`~2.9×`**, and add the reason, which is the better
  beat: *the model lines are exactly 3×; the judge, trace storage and
  infrastructure lines dilute the total*
- **`in THIS workload generation INPUT dominates`** →
  **`which component dominates is a property of the price list as much as the
  workload`**. With a steep prompt-cache discount (DeepSeek's cache-hit input is
  about 1/31 of cache-miss) generation **output** leads instead, even though the
  prompt is six times the answer.

---

## Should change — true but now incomplete

### Slide 8 · Reading one request in Phoenix

Note that a trace contains **two layers of spans**: LangGraph's auto-instrumented
node spans, and this project's own `rag.*` spans carrying the OpenInference
semantics. Without the prefix you get same-named siblings and an unreadable trace.
Span names on the slide should be `rag.retrieve`, `rag.generate`, `rag.assess`,
`rag.judge`.

Add one line, because it is a mistake worth naming: **the numbers on the terminal
and the numbers in Phoenix are now the same run.** They were not. `evaluate
--phoenix` ran the golden set locally to print the table, then ran it *again* as
a Phoenix experiment — so a hosted model produced two different sets of answers,
the two reports disagreed by a couple of points, and it charged twice. It now
records the run that already happened. Verified through the Phoenix REST API:
`groundedness 0.9306` and `citation_coverage 0.9278` in both places, to four
decimal places, over the same 30 answerable rows.

*"The trace is the source of truth" stops being true the moment your report and
your trace are two different executions.*

**Also new on this slide: span annotations.** The judged scores are pushed to
Phoenix as annotations as well as attributes — `groundedness`,
`citation_coverage`, `context_relevance`, `answer_relevance` and `sufficiency`,
each scored and banded, with `sufficiency` labelled by the routing decision. The
project sidebar shows their running means. If the slide has room for one line:
**an attribute is something you read; an annotation is something you can ask a
question of.** `annotator_kind` separates `CODE` (the offline judge) from `LLM`
(a model-graded one), because they fail in completely different ways.

### Slide 10 · The gate

- `the whole suite, offline, under a minute` → **`ten gates`**, and the suite
  behind them is **88 assertions in ~36 seconds** (plus 423 unit tests in 17)
- Add to the categories list: **`integrity — the audit chain verifies from genesis`**
- Add the line that makes the point sharper: **the gate runs against the offline
  stub even when a hosted model is serving the chat.** A merge gate that depends
  on a provider's availability, latency and price list is not a gate.

  Say it with the scar, because the claim was aspirational until recently: the
  pytest suite and the dashboard's gate button both pinned the stub, and the CLI
  had drifted to no pin at all. `make gate` therefore graded a non-deterministic
  hosted model against thresholds measured on the stub, and printed **GATE
  FAILED** on a `groundedness p10` that moves between runs. Three call sites
  defined `GATE_MODEL` independently; there is now one, and the gate report
  prints which model graded.

- If the slide lists the gate's rows, `known failures` is now **2** on the demo
  config and **6** on the CI config — the same suite, different embedder, and
  that difference is the point of the per-embedder datasets.

### Slide 13 · What changes while your code stands still

The right-hand column lists four cost components; the dashboard reports **five**.
Add **infrastructure**, and label the judge line **modelled, not incurred** — the
offline judge makes no model call, so that figure is what a model-graded judge
*would* add at the configured sample rate.

### Slide 15 · Explainability — and making edits detectable

The retention line can be stronger: in this build the Articles 19/26(6) floor is
**enforced**, not documented. A configured retention below 183 days is a startup
error that quotes the basis back at you.

---

## Worth adding — new material, if you have the room

### A new slide · The embedder is a retrieval decision, and it is measurable

This is the strongest new material in the build, and it is one table. Same
corpus, same parser, same chunks, same questions — only the embedder changes.
Recall is of the golden set's 30 expected citations:

| embedder | recall | query p50 | ingest | needs |
|---|---|---|---|---|
| `openai-text-embedding-3-small` | **100.0%** | 143 ms | ~17 s | a key, ~1¢ |
| `onnx-all-MiniLM-L6-v2` | 90.0% | 53 ms | 60 s | an 80 MB model |
| `hashing-bow-512` | 86.7% | 2 ms | 3 s | nothing |

Three things to say over it:

1. **A 13-point spread that the answers do not show you.** All three produce
   fluent, correctly-formatted, confidently-cited answers. The difference is
   only ever visible in whether the right provision was in the context at all —
   which is the whole argument for scoring retrieval separately from generation.
2. **The cheapest one is not embarrassing.** 86.7% from a hashed bag-of-words
   with no model and no network is a reminder that a statute repeats its own
   vocabulary. Reach for the API embedder because you measured, not because
   lexical sounds unfashionable.
3. **The gate runs on the *worst* one, deliberately.** `hashing-bow-512` is the
   only one of the three that is bit-deterministic — MiniLM's floats move with
   thread count, and the API does not promise stable vectors between calls. A
   gate whose vectors can shift under it fails for reasons unrelated to the
   change under test.

The consequence is the part people have not usually thought about: **the eval
datasets are per embedder.** `evals/datasets/<embedder>/` — because which golden
rows are `known_failure` is a property of the retrieval config, not of the
corpus. Six rows fail on the bag-of-words that pass on the API embedder. One
shared golden set would be wrong for two of the three, and wrong in the
direction that reads as a regression in whichever one did not generate it.

### A new slide, or slide 3 · Look inside the vector store

The dashboard now has an **Index** panel: 2,078 chunks, searchable, and clicking
one shows its metadata and its actual embedding. Worth 60 seconds because "and
then we embed it" is the step every RAG talk waves through.

Three things become concrete the moment it is on screen:

- **The breadcrumb is inside the vector, not beside it.** The embedded text
  begins `Employment Rights Act > Part 1 … > s.18 Bereavement leave > (11)` and
  then the provision. That is the hierarchical-chunking claim from slide 3, made
  checkable rather than asserted.
- **Every chunk carries its `index_version`.** Same string as the header, same
  string on every metrics row, same string on the Phoenix dataset.
- **The vector explains nothing.** 1,536 floats, L2 norm 1.000, range −0.079 to
  0.150. You can show it, and it tells you nothing about whether retrieval works
  — which is exactly why the honest instrument is recall against known
  citations, not inspection.

Typing a question into that panel runs **the agent's own retrieval**, so it
doubles as a way to answer "but why did it return *that* provision?" live.



### Slide 4 or a new slide · The chat is the interface

The dashboard now leads with a **chat**: streaming answers, per-message metric
chips, follow-ups resolved against the conversation, and a **Conversations** panel
with two tiers of history — `live` (in memory, with answers) and `audit`
(reconstructed from the compliance record: questions, citations and numbers, but
**not** the answer prose, which is not stored durably anywhere).

That split is a design argument, not a limitation, and it belongs next to slide
15: the durable record is the audit log with redaction applied at capture, and a
second durable copy of every model output would be a larger personal-data store
under weaker retention rules.

### Slide 16 · Tamper — the session 6 bridge

The bridge to session 6 now has evidence behind it. This build was scanned with
**Nuclei** — YAML templates, the same shape of tool session 6 uses:

- community library: 10,689 templates, 18,651 requests, **13 matches, all
  informational**
- five templates for this application: `POST /api/job` ran `reset` and
  `tamper_audit` with **no credential**; `/api/audit` was protected while
  `/api/chat/history` read the same records through a **side door**; session ids
  came from `Math.random()` (~41 bits, not a CSPRNG)

The line to end on: **the audit trail that makes the system explainable is itself
an asset an attacker wants to erase.** A compliance control with no
authentication in front of it is a compliance story rather than a control.

---

## Setup slide / handout

Replace the `pip install` + `make present` block with:

```bash
docker compose run --rm ingest     # the embedding pipeline — a separate job
docker compose up -d               # phoenix + dashboard
open http://localhost:8000         # chat + dashboard
open http://localhost:6006         # traces
```

Both ports publish on **loopback only**. Port overrides go in `.env` so every
compose command sees them. The env prefix is `RIGHTS_`, not `OBS_`.

`.env` also picks the embedder and the model. The demo config is
`RIGHTS_EMBEDDER=openai` and `RIGHTS_MODEL=deepseek-v4-flash`; with neither key
set, everything still runs — the embedder falls back to a local model and the
generator to an offline stub, loudly, in the header.

**Switching embedder means re-indexing and regenerating the datasets**, because
the golden set belongs to the retrieval config:

```bash
docker compose run --rm ingest
docker compose --profile tools run --rm goldens --write-baseline
```

The gate will tell you if you forget: it compares the datasets' stamped
`index_version` against the live index and fails with one line naming the fix,
rather than twenty citation failures that look like a broken retriever.

---

## Before you present

Three things, in order:

1. **Warm the stack.** The dashboard warms itself on startup now — the banner
   prints `warm-up  488 ms ttft` — but if it has been idle for a while, ask one
   throwaway question before the audience is watching. A cold first request is
   10.5 s TTFT against 0.8 s warm, on identical settings, and it is the number
   the latency panel puts on screen first.
2. **Check the header reads what you expect**: `index
   parser-6+openai-text-embedding-3-small+bc461767`, `model deepseek-v4-flash`,
   `audit intact`, and *no* `stub-local` fallback badge.
3. **Run the gate once** — `docker compose --profile tools run --rm evaluate
   --gate`. Ten rows, all PASS, and it names the model it graded with.
