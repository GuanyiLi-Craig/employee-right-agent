# CoderCo · Session 5 — Speaker Notes

**Evaluation & Observability** — how you know it works, and what it costs you.

| | |
|---|---|
| Presenter | Craig Li |
| Runtime | 50 minutes |
| Slides | 16 |
| Live demos | 7 |
| Code | `employee-right-agent/` · `docker compose up -d` |
| Series | Session 5 of 9 · CoderCo AI/MLOps |
| Next | Session 6 — security of LLM-based agents |

---

## How to run this deck

**16 slides, 50 minutes, seven live demos.** Roughly 20 minutes of slides and
30 minutes driving the dashboard. Checkpoints: demo 1 at 4:00, demo 4 (the gate) at
22:00, demo 7 (tamper) at 45:00.

### No measured numbers on the slides

Latency, kappa, PSI and cost all vary per run and per machine, so the slides name
**what to look at** and the dashboard supplies the figure. **Read every number off
the screen.** The values in these notes are what one run produced — use them to know
what to expect and to spot when something is genuinely off, never to say aloud first.

Two intentional exceptions: slide 7's mean-vs-tail contrast is a worked illustration
and is labelled as such, and slide 15's regulatory facts are external.

### Claims are workload-scoped on purpose

Statements that could read as universal laws are scoped: *in this workload* retrieval
is a latency rounding error; TTFT grows with input **and upstream work**; ITL *often*
worsens under saturation and is useful *alongside* CPU/GPU; output is dearer per token
*on many models*; judge sampling *can materially* increase the bill; generation input
dominates *in this workload*. Keep the qualifiers when you paraphrase.

### The five things to get exactly right

1. **EU AI Act scope.** Certain employment uses are *listed* in Annex III (recruitment,
   employment decisions, task allocation, worker monitoring). Where classified
   high-risk, requirements apply from **2 December 2027** — Article 6(3) allows a
   documented assessment that a listed system is not high-risk. Don't say "all EU
   employment AI".
2. **Article 12 is a logging-capability duty.** The retention floor lives in
   **Articles 19 and 26(6)**: logs you control, *appropriate to purpose, generally at
   least six months*, unless other law requires longer. Six months is a floor.
3. **GDPR** sets a storage-limitation principle — no universal maximum period.
4. **The audit record on slide 15 is our engineering design**, not an Article 12
   field list. The slide says so; say it too.
5. **The hash chain is tamper-evident, not tamper-proof.** Anchor or sign externally
   to survive a full-store rewrite.

Also: a trace tells you **how the answer was produced**, not why the model reasoned as
it did. And the span kinds are **OpenInference** `openinference.span.kind` values —
a different concept from OpenTelemetry's own SpanKind.

### The PSI gotcha (slide 14)

PSI divides by the baseline probability, so a category absent from the baseline sends
it to infinity; implementations smooth with an epsilon, which makes the number depend
on that constant. The report therefore shows **PSI over intents present in both
windows**, a separate epsilon-dependent figure including unseen categories, and an
**explicit list of new intents** — usually the more actionable finding, and it needs
no threshold at all.

### Setup before the room fills

```bash
cd employee-right-agent

# the embedding pipeline is a separate job — the only writer of the index
docker compose run --rm ingest
docker compose run --rm ingest-simple        # optional, for the contrast

docker compose up -d                         # phoenix + dashboard
open http://localhost:8000                    # chat + dashboard
open http://localhost:6006                    # traces
# click "Baseline · 24" once so the panels are not empty
```

Both ports publish on **loopback only**. If something already owns 8000 or 6006,
put overrides in `.env` — every compose command reads it, and exporting them in
one shell but not another is how you end up with a half-started stack:

```bash
printf 'DASHBOARD_PORT=8010\nPHOENIX_PORT=6016\nPHOENIX_GRPC_PORT=4327\n' >> .env
```

Only if Chroma reports a disk I/O error: `RIGHTS_RUNS_DIR=/tmp/rights-runs`.
Chroma keeps its index in SQLite, which fails on some network and virtualised
mounts.

**Have the browser on the projector before slide 1.** The deck is the interlude
between demos, not the main event.

**Warm it, then check the header.** The dashboard warms itself at startup — the
banner prints `warm-up  488 ms ttft, 2223 ms total` — because a hosted model's
first call pays TLS setup and an empty prompt cache: 10.5 s TTFT against 0.8 s
warm, on identical settings. If the stack has been idle a while, ask one
throwaway question anyway. Then read the header back to yourself:

    index   parser-6+openai-text-embedding-3-small+bc461767
    model   deepseek-v4-flash (peak rate)
    audit   intact

No `stub-local`, no red model name. Both mean a key did not reach the container,
and the demo still works but the numbers become a story about a stub.

Last check, and it takes nine seconds:

```bash
docker compose --profile tools run --rm evaluate --gate    # ten rows, all PASS
```

### Which model is serving

The default is `stub-local`: offline, deterministic, extractive, and what CI
uses. The offline guarantee is real and worth stating — but if `.env` names a
hosted model, **say which one is answering**, because the header shows it and
somebody will read it out.

Today's build is configured for `deepseek-v4-flash` with thinking disabled. The
header greys the model name out and every message carries a red
`requested → served` chip if the key is missing and it silently fell back to the
stub, so you cannot demo the wrong thing without the screen telling you.

Timings in these notes are DeepSeek's. **Baseline · 24 takes ~35 s and
Incident · 18 ~45 s**, both with a live progress counter — budget for that in
demo 2 rather than talking over silence.

### The three messages to land

1. **Your existing monitoring stays green through every failure mode that matters here.**
2. **Measure the stages separately, or you will optimise the wrong one.**
3. **A judge is an instrument. Calibrate it before you quote it.**

### The seven demos

| # | Slide | Click | The beat |
|---|---|---|---|
| 1 | 4 | a suggestion chip, then a follow-up, then the refusal | streams; TTFT/ITL, stage bar, cost, citations |
| 2 | 6 | Baseline · 24 → Incident · 18 | TTFT p50 roughly doubles, citations collapse, p10 hits zero |
| 3 | 8 | Phoenix trace | two span layers; then kill Phoenix, it still answers |
| 4 | 10 | Run the CI gate | ten gates, ~9 s, offline on the stub even when DeepSeek serves |
| 5 | 12 | Calibrate the judge | 1.000 → 0.883 → 0.764, read in that order |
| 6 | 14 | Shift intents · 20 → Drift report, then Reprice | two PSI figures + new intents; Sonnet ~2.9× Haiku |
| 7 | 16 | Tamper with the audit log | CHAIN BROKEN at record 0 |

### The three beats that must work

**Demo 2** (the incident) · **demo 5** (kappa collapses) · **demo 7** (the tamper).
If running long, shorten demo 3 to 90 seconds and cut the reprice half of demo 6 —
never cut 2, 5 or 7.

### Setting up session 6 (agent security)

Everything built today is the **detection layer** for session 6: the audit record is
what you investigate an injection incident with, the trace is how you find the request
where retrieved content became an instruction, and the refusal gate is a control.

There is now a concrete hook. This build has been scanned with **Nuclei** — YAML
templates, the same shape of tool session 6 will use — in two passes: the community
library (10,689 templates, 18,651 requests, thirteen matches, all informational) and
five templates written for this application's own surface. The interesting findings
were not web vulnerabilities at all:

- `POST /api/job` ran `reset` and `tamper_audit` with **no credential** — the two
  actions that delete and corrupt the audit record this session spends a slide
  arguing for.
- `/api/audit` was protected while `/api/chat/history` reconstructed the same records
  through a side door. A template caught the hole in the fix for the finding above it.
- Session ids came from `Math.random()` — about 41 bits, from a generator that is not
  a CSPRNG — and a transcript is readable by anyone holding its id.

All fixed; `security/README.md` records what was accepted and why. The line worth
saying out loud: **the audit trail that makes the system explainable is itself an
asset an attacker wants to erase**, and a compliance control with no authentication in
front of it is a compliance story rather than a control.

Bridge line: *"session 3 said RAG expands the model's knowledge boundary and must not
expand its permission boundary. Session 6 is about what happens when someone tries to
make it."*

### If a demo misbehaves

Skip it and describe it — the slide already says what it would have shown, without
committing to a figure. The app never blocks: jobs run on a background thread and
errors surface in the Output panel, not in a terminal nobody is projecting.

Three specific recoveries:

- **The model name is greyed out** — the key is missing and every answer is coming
  from the offline stub. The demo still works; say that it is the stub, and that the
  fallback is loud by design rather than silent.
- **The audit chip says BROKEN** — someone already ran the tamper demo. Click
  "Reset": it restarts the chain from genesis.
- **A job button is disabled** — another job is running. One at a time, on purpose:
  the index is a single-writer SQLite store.

### Standing language rules (carried from sessions 3–4)

Numbers are examples — say the label out loud. Avoid absolutes. Nothing is "free" —
say "no extra LLM call" or "an afternoon of work".

---

## At a glance

| # | Slide | Window | Length |
|---:|---|---|---|
| 1 | CODERCO · AI / MLOPS SERIES | 0:00–0:00:30 | 30 sec |
| 2 | Why your monitoring stays green | 0:00:30–0:02:30 | 2 min |
| 3 | Four questions. Most teams instrument one. | 0:02:30–0:04 | 1.5 min |
| 4 | One question, fully instrumented | 0:04–0:08 | 4 min |
| 5 | The two numbers a “latency” chart hides | 0:08–0:10:30 | 2.5 min |
| 6 | The incident you cannot see from one graph | 0:10:30–0:15:30 | 5 min |
| 7 | Two habits that make the panels honest | 0:15:30–0:18 | 2.5 min |
| 8 | Reading one request in Phoenix | 0:18–0:21:30 | 3.5 min |
| 9 | The ruler, and the bug it found | 0:21:30–0:24:30 | 3 min |
| 10 | The gate | 0:24:30–0:28 | 3.5 min |
| 11 | Three questions about one answer | 0:28–0:31 | 3 min |
| 12 | Evaluate the evaluator | 0:31–0:35 | 4 min |
| 13 | What changes while your code stands still | 0:35–0:38 | 3 min |
| 14 | Drift, then the same traffic at two prices | 0:38–0:42 | 4 min |
| 15 | Explainability — and making edits detectable | 0:42–0:45 | 3 min |
| 16 | Tamper with the audit log — then where to start | 0:45–0:50 | 5 min, then Q&A |

---

## 1 · CODERCO · AI / MLOPS SERIES

`0:00–0:00:30 · 30 sec`

> **On screen** — Evaluation & Observability · how you know it works, and what it costs you · Mostly live. 16 slides, 7 demos, one running system. · Craig Li · ~50 minutes · Session 5 of 9

Thirty seconds. Set the promise, then get off the slide.

This session is mostly live. Sixteen slides, seven demos, one running system that you will drive from a browser for about thirty of the fifty minutes.

The system: a RAG agent over the **Employment Rights Act 2025**, 335 pages of real UK legislation — 24 Parts, 12 Schedules, 459 sections and schedule paragraphs, 1,971 subsections, and 87 provisions inserted into *other* Acts, indexed hierarchically into 2,078 citable chunks.

It runs with an API key and without one. Pull the keys and the embedder falls back to a local model and the generator to an offline stub, both announced in the header rather than hidden. That is a design decision worth copying: a demo that depends on conference wifi is a demo you cannot give.

**Be accurate about the corpus.** It is the real thing: the Employment Rights Act 2025 (c. 36) from legislation.gov.uk, 335 pages — 24 Parts, 12 Schedules, 459 sections and schedule paragraphs, 87 provisions inserted into *other* Acts, 2,141 citable leaves. Say "335", not "334".

A *generated* Act is still in `data/`, and it is not a leftover: it is the control. It contains all four structural traps a parser has to survive, it regenerates byte-for-byte so `index_version` means something, and it is what the parser unit tests run against offline.

**And here is the beat worth taking.** The parser passed every one of its tests against the generated Act, with all four traps in it — and then broke on real legislation in seven distinct ways. Not one of them raised an error. Every one produced a *smaller tree*: the enacting formula set with a drop cap (`B     E IT ENACTED`) so the table of contents was parsed as body; a schedule marker sharing its line with its authorising section, so Schedule 9 was silently pruned; schedule paragraphs set one space in rather than at column 0, so all 180 of them dissolved into the previous block's text; inserted-provision headings using one space where the fixture used two, so only 20 of 87 were found. At the worst point, 39% of the leaves in the searched index had no provision above them — and a chunk with no provision above it cannot be cited, which is the entire premise of this system.

Worse, two of the four quality metrics then lied about it in the *reassuring* direction. Real citations abbreviate — `[Sch. 12 para. 4(2)]` — and the judge's sentence splitter split on ". ", tearing the mark into fragments so the sentence carrying the claim appeared to have no citation at all. Citation coverage read 0.78 while every sentence was correctly cited. The recogniser also capped a citation at 80 characters, against a corpus whose longest is 97.

The transferable point, and it is the one to leave them with: **a parser that passes its own tests on a corpus you generated has told you nothing about a corpus you did not — and the metric that should have caught it was broken by the same change.** `data/README.md` has the full table with before-and-after numbers if anyone asks.

Before you start talking, have the stack already up and the browser open on the projector. The first slide should be the LAST thing they look at for a while.

---

## 2 · Why your monitoring stays green

`0:00:30–0:02:30 · 2 min`

> **On screen** — four assumptions your existing observability rests on, and why none of them hold here · A normal service fails loudly · Returns HTTP 200 with a fluent, confident, wrong answer. No exception, no stack trace, nothing to alert on. · A normal service has a deterministic contract · Same input, different output. You cannot assert the result mechanically, so “it worked when I tried it” is not evidence and one bad example…

Two minutes, and this slide earns the session. Most of the room already has monitoring; the argument is that its assumptions are all false here.

A normal service fails loudly. This one returns HTTP 200 with a fluent, confident, wrong answer. No exception, no stack trace, nothing to alert on. Every failure mode we discuss today is invisible to uptime monitoring.

A normal service has a deterministic CONTRACT — the system underneath may be distributed and messy, but you can assert the status code and the payload mechanically. Here you cannot: same input, different output. So "it worked when I tried it" is not evidence — which is why everything later is about distributions and thresholds rather than examples.

A request costs what the last one cost. Here cost varies per request with token counts, and cost and latency are not reliably correlated. A long retrieved context is expensive and fast; a long generation is both expensive and slow. Because the relationship is loose in both directions, a latency dashboard alone will not surface your expensive requests.

Quality is fixed at deploy. It drifts while the code stands still: the corpus moves, the questions move, the provider updates the model underneath you.

Land the takeaway plainly: uptime, latency and error rate stay green through every one of those.

---

## 3 · Four questions. Most teams instrument one.

`0:02:30–0:04 · 1.5 min`

> **On screen** — you already have this · solved · Is it up? · latency, errors, throughput · most teams guess · today · Is it any good? · groundedness, citations, refusals · nobody attributes it · What does it cost? · per request, per conversation, per intent · and a deadline is coming · Can you prove it? · who asked, what it read, how the answer was produced · Production AI needs an auditable decision trail before regulation force…

Ninety seconds. This is the map for the whole session.

Four questions. Is it up — solved, everyone has it. Is it any good — most teams guess, or ask in standup. What does it cost — almost nobody attributes cost to a feature or an intent. Can you prove it — until recently a nice-to-have.

Note the wording on the fourth: who asked, what it read, and HOW THE ANSWER WAS PRODUCED. Not "why it answered". A trace records the mechanism — which documents, which model, which prompt version, which thresholds. It does not tell you why the model reasoned as it did, and claiming otherwise oversells what any of this gives you.

On the fourth: get the regulatory position right, because someone in the room will know it.

The EU AI Act's Annex III high-risk obligations were originally due to apply from 2 August 2026. They were postponed by the Digital Omnibus on AI (Regulation (EU) 2026/1744, in force July 2026) and now apply from 2 December 2027 for standalone Annex III systems. Employment and worker-management AI sits in Annex III point 4, which is why the demo corpus is employment law rather than something whimsical.

Be precise about the scope, because the slide is: this is for ANNEX III HIGH-RISK employment AI. Not every employment-related system is automatically high-risk — Article 6(3) lets a provider conclude that an Annex III system does not pose a significant risk (for example where it performs a narrow procedural task), subject to documenting that assessment and registering it. So "all EU employment AI" would overstate it.

Do NOT hang the argument on the deadline, for two reasons: it has already moved once, and it makes the case sound like compliance theatre. The stronger framing, and the one on the takeaway line: production AI needs an auditable decision trail before regulation forces it. The deadline is a date; the engineering need is permanent.

If asked what did NOT move: the Article 5 prohibited-practices regime and the GPAI provider obligations are already in force, and Article 50 transparency duties kept their original schedule.

Say what the rest of the hour is: questions two, three and four, demonstrated rather than described. Then switch to the browser.

### Optional here, 60 seconds · Look inside the vector store

If the room is the kind that asks "but what does *embedding* actually mean", the dashboard's **Index** panel answers it faster than any slide. 2,078 chunks, searchable; click one and you get its metadata and its actual vector.

Three things become concrete the moment it is on screen, and none of them survive being asserted:

- **The breadcrumb is inside the vector, not stored beside it.** The embedded text begins `Employment Rights Act > Part 1 … > s.18 Bereavement leave > (11)` and *then* the provision. That is the hierarchical-chunking claim, checkable.
- **Every chunk carries its `index_version`** — the same string as the header, the same string on every metrics row, the same string on the Phoenix dataset. That is what makes "which index produced this answer" answerable six months later.
- **The vector explains nothing.** 1,536 floats, L2 norm 1.000, range −0.079 to 0.150. Show it, then say so plainly: this is why the honest instrument is recall against known citations, not inspection. Nobody has ever debugged retrieval by looking at the numbers.

Typing a question into that panel runs **the agent's own retrieval**, not a private copy — so it also answers "but why did it return *that* provision?" live, which is the question that usually derails demo 1.

### Optional here, 90 seconds · The embedder is a retrieval decision, and it is measurable

Same corpus, same parser, same chunks, same 30 questions. Only the embedder changes. Recall is of the golden set's expected citations:

| embedder | recall | query p50 | ingest | needs |
|---|---|---|---|---|
| `openai-text-embedding-3-small` | **100.0%** | 143 ms | ~17 s | a key, ~1¢ |
| `onnx-all-MiniLM-L6-v2` | 90.0% | 53 ms | 60 s | an 80 MB model |
| `hashing-bow-512` | 86.7% | 2 ms | 3 s | nothing |

Three things to say, in this order:

1. **A 13-point spread the answers do not show you.** All three produce fluent, correctly-formatted, confidently-cited answers. The difference is only ever visible in whether the right provision reached the context — which is the entire argument for scoring retrieval separately from generation, and it lands harder here than on any diagram.
2. **The cheapest one is not embarrassing.** 86.7% from a hashed bag-of-words with no model and no network, because a statute repeats its own vocabulary and every chunk starts with a breadcrumb full of exact terms. Reach for the API embedder because you measured, not because "lexical" sounds unfashionable.
3. **The gate runs on the worst one, deliberately.** The bag-of-words is the only one of the three that is bit-deterministic — MiniLM's floats move with thread count, and the API does not promise stable vectors between calls. A gate whose vectors shift under it fails for reasons unrelated to the change under test.

And the consequence most people have not thought through: **the eval datasets are per embedder.** Six golden rows are `known_failure` on the bag-of-words that pass on the API embedder. One shared golden set would be wrong for two of the three configs, and wrong in the direction that reads as a regression in whichever one did not generate it.

---

## 4 · One question, fully instrumented

`0:04–0:08 · 4 min`

> **On screen** — DEMO 1/7 · click a suggestion chip, watch it stream · TTFT · time until the user sees anything — the “is it broken?” number · ITL, mean and p95 · the gap between tokens — the “why is it so slow?” number · the stage bar · classify · retrieve · generate, side by side · cost, tokens, citations · and citations that resolve to provisions, not page numbers · In this workload retrieval is a rounding error on latency — a…

**DEMO 1.** Four minutes. Switch to the dashboard and stay there.

Click the first suggestion chip: **"What does the document say about bereavement leave?"** The five chips are the demo arc — a question the corpus answers well, a follow-up that only works with conversation context, two questions whose provision states a concrete figure, and one that must be refused. Every one is checked against the live index by a test, because a suggested question that refuses in front of a room is worse than no suggestion.

Do **not** ask for "the qualifying period for unfair dismissal". This corpus names the topic but states no figure, and the model will correctly tell you so — an honest answer that looks like a failure. That question needs the real Act.

Watch it stream. TTFT is the number the room should see arriving, not one you read to them afterwards.

Read the numbers off the screen — the slide deliberately does not carry them. TTFT is the time until anything appears, the "is it broken?" number; expect a couple of hundred milliseconds on a laptop. ITL is the gap between tokens, the "why is it so slow?" number. Point out that the ITL p95 sits meaningfully above the mean: that is the stutter a user perceives, and an average hides it.

Then the stage bar, which is the beat that lands: classify is effectively zero, retrieve is tens of milliseconds, generate is the overwhelming majority. Say it as a statement about THIS workload rather than a law: here retrieval is a rounding error on latency, and it still drives most of the quality decisions. That does not make retrieval unimportant — it decides whether the answer is RIGHT. In this shape of system retrieval is a quality lever and generation is the latency lever, and confusing the two wastes months. A workload with a heavy reranker or multi-hop retrieval would look different.

Also point at cost, tokens, and the citations resolving to provisions rather than page numbers. Every request carries the index version that produced it.

Then two things the chips set up.

**A follow-up.** Ask "How long is it?" — the second chip. It gets answered, and the message carries a `follow-up resolved` chip. The mechanism is worth thirty seconds: the follow-up borrows the distinctive terms from the last question that was not *itself* a follow-up, deterministically, with **no model call** — same reason intent classification has none. A rewrite that changes when you re-run it makes every downstream measurement unreproducible. It resolves topic, not reference: it carries "bereavement leave" forward so the retriever has something to match; it does not know that "that" meant the notice period.

There is a wrinkle worth owning if someone asks. The gate scores the *original* question everywhere else, precisely so a refinement cannot talk its way past it. For a conversational follow-up it scores the resolved question — because those words came from the *user*, one turn earlier, and scoring the fragment alone refuses every follow-up. The distinction is who supplied the words, and there is a test for each half.

**A refusal.** Ask the cryptocurrency chip and show it refuse, with the sufficiency score and the threshold on screen. Refusing well is a feature, and it is measurable.

Expand *how this answer was produced* for the stage bar, the retrieved provisions with their scores, and the hash of this decision's audit record — which is where slides 15 and 16 are going.

---

## 5 · The two numbers a “latency” chart hides

`0:08–0:10:30 · 2.5 min`

> **On screen** — request in · first token · last token · retrieve · queue + prefill · decode · ms · TTFT · ITL × (tokens−1) · TTFT grows with input + upstream work · ITL degrades under saturation · Everything before the first token: your retrieval and assembly, queueing, and the model reading the whole prompt. Every extra chunk makes the start slower. · Often steady in isolation, and frequently worsens as the serving fleet saturat…

Two and a half minutes. Back to the slide for the only theory in the metrics section.

Walk the timeline. Request arrives, our retrieval runs — a rounding error. The model queues and prefills: it must read the entire input before emitting anything. First token appears. Everything up to there is TTFT. After that, decode, one token at a time; the gap between them is ITL.

The two cards are the point.

TTFT grows with input AND with upstream work. It is not only prefill: it covers your retrieval and assembly, any queueing, and then the model reading the whole prompt. Every extra retrieved chunk makes the start slower, which is the direct price of "just retrieve more to be safe" — and it lands on the metric users notice most.

ITL degrades under saturation. It is often steady in isolation and frequently worsens as the serving fleet saturates, which makes it a useful model-serving capacity signal ALONGSIDE infrastructure metrics like CPU and GPU utilisation — not a replacement for them.

The formula: end-to-end is approximately TTFT plus ITL times output tokens. Four levers fall out of it — fewer chunks and prompt caching move TTFT, brevity moves the decode term, more capacity or a smaller model moves ITL.

And the line at the right of the formula box, which is the bit nobody measures: the gap between that formula and your observed end-to-end is non-generation overhead — orchestration, network hops, post-processing, and the client itself. On a thin pipeline it is small. On a real system with three sequential model calls it is frequently the largest single term.

If asked about TPOT: usually the same idea as ITL, sometimes defined to include the first token. Definitions vary between vendors; define yours and move on rather than winning the vocabulary argument.

---

## 6 · The incident you cannot see from one graph

`0:10:30–0:15:30 · 5 min`

> **On screen** — DEMO 2/7 · Baseline · 24 then Incident · 18 · TTFT p50 · jumps — read the number off the panel, not off this slide · citation coverage · collapses toward zero, in the same window · separately · each looks like noise a busy team closes as “no repro” · together · they point at one story — and the TRACE is what confirms which model answered · Quality and latency on ONE time axis. On two dashboards owned by two teams,…

**DEMO 2.** Five minutes. The best beat in the session — do not rush it.

Click "Baseline · 24". It takes about thirty-five seconds against a hosted model, with a progress counter — narrate the panels filling in: percentiles, never a mean. Point out that the refusal rate is a quality metric, not an error rate.

Then click "Incident · 18". About forty-five seconds. It asks the **same questions** as the baseline, deliberately: if the question mix moved too, you could not tell whether the model changed or the traffic did.

TTFT p50 jumps sharply — read the before and after off the panel. Citation coverage collapses toward zero. Same window, same panel. Say both numbers as the screen shows them; do not pre-announce values, because they move between runs.

One thing to point at explicitly: the latency and quality panels are **windowed** — "last 20 of N". That is not a detail. A cumulative percentile cannot move during an incident: once a few hundred healthy requests are in the denominator, eighteen bad ones cannot shift the median, and the panel that should have caught it stays green. This was a real bug in this build until it was driven from a browser.

Now make the argument. Separately, each of those is the kind of thing a busy team closes as "no repro". Latency up a bit — traffic is up, probably fine. Citations down — probably a prompt change, someone will look. Together they point at one story.

Be precise about what the graphs actually establish, because this is where a careful engineer will push back. The two series tell you WHEN something changed and that the two symptoms share a cause. They do not, on their own, prove which model answered — that is an inference. The trace and the request-level configuration are what confirm it. The line to say: "the graphs tell us when something changed; the trace tells us the fallback caused it." That is also the argument for the next section.

Then the caveat — and **read it off the Output panel rather than reciting it**, because the job measures both signals and prints the deltas. How far groundedness moves depends on the model: an extractive one lifts sentences verbatim and a lexical judge stays perfectly happy, while a capable one paraphrases and the same judge marks it down. On the offline stub groundedness barely moves; on DeepSeek it moves too, for a reason worth naming — a lexical judge cannot see paraphrase, which is exactly the blind spot demo 5 puts a number on.

The claim that holds either way, and the one the panel shows: **citation coverage collapses far further than groundedness.** A recent run measured citation coverage 1.000 → 0.000 against groundedness 0.975 → 0.896. One headline quality number would have hidden that entirely, which is the argument for a small family of signals instead of a single score.

If you want the mechanism: the degraded fallback wraps whichever client is configured, leads with the lowest-ranked evidence, removes citations from the output, and adds latency. Removing them is deliberate rather than instructed — DeepSeek kept citing through an explicit instruction not to, and again after the citation was stripped from the context header, because the breadcrumb still names the section. A capable model is hard to make careless, so the behaviour being simulated is simulated directly.

Close by pointing forward: correlation found the incident. It cannot tell you which model answered. For that you need the trace.

---

## 7 · Two habits that make the panels honest

`0:15:30–0:18 · 2.5 min`

> **On screen** — 1 · never ONLY a mean (illustrative) · 2 · a trace, not just a log · groundedness mean = 0.92 · CHAIN · agent.request · RETRIEVER · Green. Above threshold. Nobody paged. · retrieve · groundedness p10 = 0.45 · EVALUATOR · assess · The worst tenth are barely supported — and answered with identical confidence. · LLM · generate · OpenInference span kinds are what make a trace queryable rather than a pile of strings —…

Two and a half minutes. Two habits, one slide.

Left: never ONLY a mean — the wording matters, because means are useful and the problem is relying on them alone. These two figures are a worked illustration and the slide says so; do not cross-reference them against the dashboard. A mean of 0.92 is green, above threshold, nobody paged, and arithmetically true. The p10 of 0.45 says: the worst tenth of answers are barely supported and are delivered with identical confidence. Those users do not experience an average.

Land the line, then pause: a steady 0.85 with no catastrophes beats a 0.92 average hiding a disaster tail. Then the move that wins an engineering room — this is the same argument as p95 latency, which they already accept without discussion. Nobody ships on mean latency. Do not ship on mean quality.

Right: a trace, not just a log. Four spans from one request. The important detail is that we wrote no tracing code to get that shape — every node boundary in the LangGraph agent became a span boundary. That is a genuinely good reason to reach for a graph framework, better than elegance.

And the caption: OPENINFERENCE span kinds are what make a trace queryable rather than a pile of strings. Be precise about the name — CHAIN, RETRIEVER, EVALUATOR and LLM are values of `openinference.span.kind`, which is a different concept from OpenTelemetry's own SpanKind (SERVER, CLIENT, INTERNAL and so on). A span called "step_3" with a JSON blob attached is technically a trace and operationally useless. With an OpenInference span kind, "show me all retrievals that returned nothing" becomes a filter.

Takeaway adds trends, phrased carefully: a sustained slide should trigger INVESTIGATION even while the absolute score is still above threshold. Not every drift is an incident, and calling it one devalues the word. Fixed thresholds either fire constantly or never; watch the derivative as well as the level.

---

## 8 · Reading one request in Phoenix

`0:18–0:21:30 · 3.5 min`

> **On screen** — DEMO 3/7 · make phoenix → open the trace for the question you just asked · the span tree · classify · retrieve · assess · generate — no per-node tracing code, once instrumentation is configured · retriever span · the documents it returned, their scores, the index version · assess span · the sufficiency score against its threshold — a decision, recorded · then kill Phoenix · re-run: the agent still answers. Spans b…

**DEMO 3.** Three and a half minutes in Phoenix, already running in the compose stack on port 6006.

Open the trace for the question from demo 1. You will see **two layers of spans in one trace, and that is deliberate**: LangGraph's auto-instrumentation emits a span per node, named after the node and carrying its state in and out, while this project's own spans are prefixed `rag.` and carry the OpenInference semantics. Without the prefix you get same-named siblings from two instrumentation layers and an unreadable trace.

Walk the tree: `rag.classify`, `rag.retrieve` (RETRIEVER), `rag.assess`, `rag.generate` (LLM), `rag.judge` (EVALUATOR). Note the precise claim: no PER-NODE tracing code was written, once auto-instrumentation is configured — the instrumentation itself is a setup step, and saying otherwise invites a fair challenge. Expand `rag.retrieve` and show the documents table with scores and the index version. Expand `rag.assess` and show the sufficiency score against its threshold — that span is a DECISION, recorded. When someone asks in three months why the system refused a particular question, that number is the answer.

If you have a spare minute, `make phoenix` uploads the golden set as a Phoenix **dataset** and runs it as an **experiment**, tagged with `index_version` and `prompt_version` so two runs are comparable. Mention that it is wrapped and falls back to a local summary — the API surface moves, and pointing this at a real server surfaced two mismatches while the numbers kept printing. That fallback is the design, not a workaround.

**And a mistake worth naming, because it is the kind everyone ships.** That command used to run the golden set *twice*: once locally to print the table, then again as a Phoenix experiment. With a hosted model that means two different sets of answers, so the table on the terminal and the experiment in Phoenix disagreed by a couple of points — and it charged twice. It now records the run that already happened. Checked through the Phoenix REST API: `groundedness 0.9306` and `citation_coverage 0.9278`, identical to four decimal places in both places, over the same 30 rows.

The line to land: **"the trace is the source of truth" stops being true the moment your report and your trace are two different executions.** Nothing about the code looked wrong. The only way to catch it was to compare the two numbers, which is the thing nobody does because they are obviously supposed to be the same.

**Show the annotations while you are here.** The right-hand sidebar of the project page carries running means for `groundedness`, `citation_coverage`, `context_relevance`, `answer_relevance` and `sufficiency`. Those are not span attributes being displayed — they are **span annotations**, pushed through Phoenix's REST API after each request. Worth one sentence on why both exist: an attribute is something you read on one span; an annotation is something the project can aggregate and filter, so *"show me every generated answer that scored under 0.7 for groundedness"* is a query rather than a script. Note the `sufficiency` annotation is labelled with the routing decision — `generate` or `refuse` — so the gate's verdict filters alongside the quality it produced.

If someone asks whether this is just a nicer log: `annotator_kind` distinguishes `CODE` from `LLM`. The offline judge is deterministic code; a model-graded judge is an opinion. Putting a lexical overlap score and a model's judgement in one bucket would hide that they fail in completely different ways.

Two mistakes worth mentioning if the room is technical, because both are the kind that look like someone else's bug. Annotating inline returns `404 Spans with IDs ... do not exist` — the batch processor holds spans for seconds, so you are annotating something Phoenix has not received, and the error reads like a bad span id. And it belongs off the request path anyway: this is a session about latency, and an HTTP round trip per answer would inflate the number on the panel with the cost of reporting it.

Then do the thing that makes the point. Stop Phoenix. Re-run the question in the dashboard. It still answers; spans become no-ops; nothing errors.

Say why: every Phoenix import in the project is optional and every export failure is swallowed. Observability must never be able to take down the thing it observes — and the number of teams who have taken a production outage because their tracing collector filled a disk is not small.

If Phoenix will not start on the day, this demo degrades gracefully: walk the span structure in the code and move on. Nothing later depends on it.

---

## 9 · The ruler, and the bug it found

`0:21:30–0:24:30 · 3 min`

> **On screen** — one row of the golden set · why it survives a model upgrade · {"id":"g13", "question":"What does section 18 say about bereavement leave?", "must_cite":["s.18"]} · · asserts CITATIONS, not prose · and asks what a provision SAYS, not what you are entitled to · · stratified by intent · a fix to one topic cannot mask a break in another · · includes out-of-scope cases · refusing correctly is a behaviour worth testing ·…

Three minutes. The golden set, and the bug it found — tell the second half as a story, because it is a real one from building this.

Left: one row. A question, the provisions it must cite, whether it should refuse.

Note the register of the question: "what does section 18 SAY about bereavement leave", not "am I entitled to bereavement leave". That is deliberate. The system reports what a provision says; it does not advise whether an entitlement is currently in force. Keeping the golden set in that register stops the evaluation suite quietly asserting current law — and stops the demo doing it in front of an audience.

Right, the design advice: it asserts CITATIONS, not prose. Wording is free to change — models get updated, prompts get tuned — but the law it must cite does not. If you assert on exact output text, your suite breaks every time anything improves, and a suite that cries wolf gets deleted. Stratified by intent so a fix to one topic cannot mask a break in another. Includes out-of-scope cases, because refusing correctly is a behaviour worth testing and most suites only contain questions the system can answer.

Then g17. "Can a confidentiality clause cover harassment complaints?" That is a valid question — the Act covers it at section 24. And the system refuses, because the Act says "contractual duties of confidentiality" and the user says "clause". Sufficiency 0.41 against a 0.45 gate.

Three options, and the room will recognise all of them. Delete the case: the suite goes green and the bug ships — this is what happens under deadline pressure, every time. Tune until it passes: you have fitted your gate to one example, and it feels like progress. Or mark it KNOWN: the gate asserts the list does not GROW, and fails loudly if the case starts passing so you remember to remove the marker.

The third is what mature suites do and almost nobody teaches it.

Takeaway: a suite that has to be perfect gets its failing cases quietly deleted. Keep them visible.

---

## 10 · The gate

`0:24:30–0:28 · 3.5 min`

> **On screen** — DEMO 4/7 · Run the CI gate · the whole suite · offline, under a minute, no API key, no flake · structural only · citations, refusals, routing, telemetry fields, the audit chain · the index is rebuilt · so a change to the chunker is actually exercised · break it live · delete the citation instruction from the prompt and re-run · Nothing here asks a model for an opinion — which is exactly why it may block a merge.

**DEMO 4.** Three and a half minutes.

Click "Run the CI gate". About nine seconds, and the Output panel prints ten gates with observed value, threshold and verdict.

Say the thing at the bottom of its own output: **it ran against `stub-local`, not against DeepSeek**, even though DeepSeek is serving the chat. That is enforced, not a convention. A merge gate whose result depends on a hosted provider's availability, latency and price list is not a gate — and running the golden set through a hosted model would make this button take three minutes and cost money. Evaluating a hosted model is a separate, deliberate command.

If you want the scar that makes that claim credible, it is a good one. Three places pinned the gate model: this button, the pytest suite, and the CLI. Each defined the constant independently, and the CLI's had drifted to no pin at all — so `make gate` on a terminal graded a *non-deterministic hosted model* against thresholds measured on the stub, and printed **GATE FAILED** on a `groundedness p10` that moves between runs. There is now one definition, and the report prints the model it graded with. **A convention duplicated in three files is not enforced; it is three chances to disagree.**

The categories: structure — every answer carries a citation or is a refusal that explains itself. Routing — out-of-scope refuses, known failures have not grown. Retrieval — the expected provision appears. Telemetry — every request records index version, model, tokens, cost; you can and should test your own observability. Arithmetic — end-to-end covers the sum of the stages. Integrity — the audit chain verifies from genesis.

Worth naming if it comes up: "citation hit rate" being 1.000 depends on comparing citations at **provision level**. A model that cites `s.19(4)` where the retrieved block was headed `s.19` has cited more precisely, not wrongly, and exact string matching scored every one of those zero when this was first pointed at a real model.

Point out that CI rebuilds the index rather than checking one in, so a change to the chunker is actually exercised. Otherwise your eval tests a stale artefact and passes while the thing you changed is broken.

One more thing about that row marked `known failures 2`. **CI's number is 6, on the same suite and the same corpus** — because CI runs the offline embedder and the demo runs the API one, and which rows fail is a property of the retrieval config. That is why `evals/datasets/` has one directory per embedder, and why the gate refuses to run against an index its dataset was not generated for: one legible failure naming the fix, instead of twenty citation failures that read like a broken retriever.

If the room is engaged, break it live: delete the citation instruction from the system prompt, re-run, watch the citation test fail and name the case. Thirty seconds, and it demonstrates exactly what a gate is for.

The line at the bottom: nothing here asks a model for an opinion, which is precisely why it may block a merge. The moment your merge gate depends on a non-deterministic judgement, you have a gate that fails randomly — and a gate that fails randomly gets disabled within a month.

---

## 11 · Three questions about one answer

`0:28–0:31 · 3 min`

> **On screen** — QUESTION · Context relevance · did we retrieve material capable of answering this? · RETRIEVAL is the suspect · context relevance · answer relevance · Groundedness · is every claim supported by that material? · GENERATION is the suspect · Answer relevance · CONTEXT · ANSWER · does it address what was actually asked? · question ↔ answer alignment · groundedness · + citation coverage — is each claim attributable? ·…

Three minutes. Draw the triangle with your hand.

Three things in a RAG answer: the question, the context retrieved, the answer produced. The triad is the three edges.

Context relevance, question to context: did we retrieve material capable of answering this? A judgement about RETRIEVAL — and note you can compute it without generating an answer at all, which makes it the cheapest useful eval there is.

Groundedness, context to answer: is every claim supported by that material? A judgement about GENERATION.

Answer relevance, question to answer: does it address what was actually asked? A perfectly grounded answer to a slightly different question scores well on the other two and is useless.

Plus citation coverage, which is ours rather than canonical: is each claim attributable? In a compliance setting an uncited true claim is still unverifiable.

Then the reading, which is the lookup table they will use on Monday. Context relevance low: the material never arrived — chunking, embeddings, top-k, or a filter. Do not touch the prompt; that is what the amber bar says. High context relevance with low groundedness: it had the evidence and invented anyway — now it is a generation problem. Both high, answer relevance low: question-to-answer alignment is the suspect. Usually a rewrite step that drifted from the user's intent, but it can also be over-hedging or answering a narrower sub-question — a direction to investigate, not one diagnosis.

Worth naming the bias: teams overwhelmingly guess "generation" first, because prompts are editable in an afternoon and chunkers are a week plus a re-index. The only cure is measuring the stages separately.

If asked how this maps to RAGAS: context relevance ≈ context precision, groundedness = faithfulness, answer relevance keeps its name.

---

## 12 · Evaluate the evaluator

`0:31–0:35 · 4 min`

> **On screen** — DEMO 5/7 · Calibrate the judge · 12 clean examples · a perfect Cohen's kappa. Ask the room: would you trust it now? · + 5 realistic ones · kappa falls. Read both numbers off the screen. · ground AND citations · gating on both beats groundedness alone — two complementary signals can beat either alone · case h001 · a correct paraphrase scores zero. A lexical judge cannot see paraphrase. · The judge did not get worse when…

**DEMO 5.** Four minutes. The intellectual high point — slow down.

Click "Calibrate the judge". Instant — no model call, nothing to wait for.

Twelve clean-cut examples: **kappa 1.000**. Read it off the screen, then stop and ask the room whether they would trust the judge now. Most will say yes.

Then read the second line: add five realistic borderline cases and it falls to **0.883**. (These three lines are deterministic — the calibration file and the judge are both fixed — so they will reproduce exactly, but still read them from the screen rather than from memory.)

Say the line: the judge did not get worse. The test got honest. Kappa 1.00 on twelve easy examples is a property of the examples, not of the judge — and that is exactly how teams end up trusting an evaluator that cannot do its job.

Third line is the payoff, and read it in this order: gating on **groundedness alone scores 0.764**, worse than the 0.883 you get from gating on groundedness AND citation coverage together. Two COMPLEMENTARY signals beat either alone — they fail on different cases, which is the property that matters. Combining two signals that fail the same way buys nothing.

**If you have thirty seconds, this is the best story in the deck.** Clean kappa was not 1.000 for a while — it was 0.824, and one clean case disagreed. The judge scored it 0.00; the label said "acceptable". The judge was right. The answer was `[s.1(2)] provides: (2) of that section.` — the whole answer. A statutory cross-reference had wrapped across its own number in the PDF (`...that comply with subsection` / `(2) of that section.`), the parser read the second line as a new subsection, and that fragment went into the searched index as a citable chunk. Sixty-three leaves were fragments like it. Fixing the parser removed the row, and clean kappa returned to 1.000.

Land it: **the instrument found a bug in the corpus, not the other way round.** That is what "gate the instrument first" buys you, and it is a much better argument than anything I could say in the abstract.

Then the two disagreements the output names, one in each direction.

**h001** — a correct paraphrase. Right answer, right meaning, none of the context's vocabulary, scored zero on groundedness. A false negative, and a structural one: a lexical judge cannot see paraphrase. It is not a threshold to tune. It is the reason you also sample with a model-graded judge, and the reason you never run only one kind.

**h005** — real, correctly-cited boilerplate that does not answer the question. Perfect on support, useless as an answer. A support-only judge scores it 1.0. That is the failure in the opposite direction, and it is why "is it grounded" is not the same question as "is it any good".

Explain kappa if the room needs it: it corrects for chance. If ninety percent of your answers are good, a judge that says "good" every time gets ninety percent agreement and is worth nothing. Kappa removes that floor.

Mention that the CI gate has a test asserting kappa stays above 0.6 — gate the instrument before you gate with it.

---

## 13 · What changes while your code stands still

`0:35–0:38 · 3 min`

> **On screen** — THE QUESTIONS MOVE · THE BILL MOVES · generation · input · → re-index; date filters · Corpus drift · the chunks you chose to send — every call, forever · the source documents changed · generation · output · → check coverage first · Intent drift · materially dearer per token than input on many models — brevity is a cost control · trace storage · the QUESTIONS moved · cheap each, enormous in aggregate · → could be e…

Three minutes. Two halves, both quick.

**LEFT —** three things get called drift, and separating them is worth real money. Corpus drift: the source documents changed under the index; re-index, add effective-date filters. Intent drift: the QUESTIONS moved; nothing is broken, attention moved. Performance drift: the scores fell.

The naming is deliberate. "Data drift" in the wider ML sense means the INPUT distribution moved — which here is intent drift — so calling the corpus one "data drift" collides with how most of the room already uses the term. And "performance drift" is the more standard name for scores falling.

Keep them apart because quality drift can be caused by either of the others, or by the model changing underneath you, and the investigations are completely different. Re-indexing does not fix a prompt regression, and prompt engineering does not fix a corpus that moved.

The green bar is the actionable inference: if intent mix moved AND quality fell in the same window, suspect a coverage gap rather than a regression. Users started asking about something your index covers badly. Fix content, not temperature.

**RIGHT —** where the money goes. Generation input is the largest model cost here, and what drives it is the chunks you chose to send: every extra chunk is input tokens on every call, forever. Output is materially dearer per token than input on many current models — for the two used in this demo it is five times — so brevity is a cost control and not just a style preference. Say "on many models" rather than quoting a universal ratio. Trace storage is genuinely large in aggregate — and be honest that its share here is inflated because our stub model is free; the next demo fixes that.

Then the amber bar, the bit most teams miss: the judge is on this list. It is an LLM call, so scoring every request can materially increase the bill — how much depends on how heavy your judge is relative to generation. That is the real reason production evaluation is sampled: money, not statistical elegance. It also gives you the natural opening to talk about sampling strategy if the room asks.

Footnote: dated prices from one table in the config. The DeepSeek rows are the published list; the rest are illustrative. Change the table and every number downstream moves. That property is the difference between a cost model and a spreadsheet.

The dashboard reports FIVE components, not four: generation input, generation output, the judge, trace storage, and infrastructure. The judge line is labelled *modelled, not incurred* — the offline judge makes no model call, so the figure is what a model-graded judge would add at the configured sample rate. A cost panel that cannot tell incurred from modelled is a cost panel that gets disbelieved.

---

## 14 · Drift, then the same traffic at two prices

`0:38–0:42 · 4 min`

> **On screen** — DEMO 6/7 · Shift intents · 20 → Drift report then Reprice · PSI + new-intent check · the known mix shifts, and unseen intents are flagged separately · not a bug report · it says the QUESTIONS changed — a content investigation · Haiku vs Sonnet · identical recorded tokens, a materially bigger bill · which component dominates · is a property of the price list as much as the workload — read it off the panel · Rec…

**DEMO 6.** Four minutes. Two things back to back.

First, click "Shift intents · 20", then "Drift report".

The report gives you two numbers and a list, deliberately.

PSI over the intents present in BOTH windows — a moderate shift. Then PSI including the unseen categories, which is far larger. Then an explicit list of intents that are new.

Explain why, because this is a real gotcha and someone will know it: PSI divides by the baseline probability, so a category that was absent from the baseline sends the formula to infinity. Every implementation smooths with an epsilon, which keeps the number finite but makes it depend heavily on the epsilon you chose. A single new intent can dominate the score. So quoting one PSI figure for a window containing new categories is quoting your smoothing constant as much as your data.

The practical shape: PSI for the shift among known intents, plus a separate unseen-category check. The second is usually the more actionable finding and needs no threshold at all — a topic that did not exist last week is interesting regardless of what any index says.

The framing: this is not a bug report. It says the questions changed. If you page someone with "PSI alert" they will go and look at the model, and the model is fine. The emerging-intents list is the useful output — that is what you take to whoever owns the corpus.

One design note worth making because it is counter-intuitive: intent is classified by keyword, deliberately. An embedding classifier has better coverage, but a label that changes when you re-run it makes every time series meaningless. If you group metrics by a dimension, that dimension must be stable.

Caveat: the PSI bands — under 0.1 stable, over 0.25 significant — come from credit-risk modelling. A useful convention, not a law of nature.

Second, pick a model in the dropdown next to "Reprice" and click it. Nothing is re-run; we already have the token counts. The table puts Haiku beside your choice, component by component, with a monthly projection at fifty thousand requests a day.

Sonnet lands at roughly **2.9x** Haiku overall, not 3x — and the discrepancy is the better beat. The model lines *are* exactly 3x; the judge, trace storage and infrastructure lines are model-independent and dilute the total. The panel shows the components, so you can point at why.

Then the detail that actually matters, and read it off the screen rather than from memory: **which component dominates is a property of the price list as much as the workload.** On this traffic against DeepSeek, with roughly a thousand of eleven hundred prompt tokens served from cache at about a thirty-first of the fresh rate, generation **output** leads — even though the prompt is six times the answer. Uncached, the same traffic is input-dominated. The report looks the lever up rather than asserting one, and it names it: brevity for output, top-k and chunk size for input.

The durable version of the point: in a RAG system the biggest cost lever is usually the chunks you chose to send, on every call, forever — but check your prompt cache is working before you accept that, because a steep cache discount moves the answer.

If someone asks why the earlier panels showed infrastructure dominating: because the offline stub is priced at zero. Pricing any real model moves generation to the top. That contrast is worth naming.

One more repricing you can do if the room is engaged, and it is unusual: reprice DeepSeek **peak against off-peak**. Same model, same traffic, and the only variable is what time of day you ran it — about 1.65x. It is a pricing row rather than a model id, and the app refuses it as a serving model rather than letting you find out from a 400.

Closing observation: recorded tokens plus a price table is a cost model, with no new instrumentation. And it makes routing easy questions to the cheap model an arithmetic question rather than a taste one.

---

## 15 · Explainability — and making edits detectable

`0:42–0:45 · 3 min`

> **On screen** — a practical audit record · EU AI Act, Article 12 · engineering design — not an Article 12 field list · Requires high-risk systems to technically allow automatic recording of events over the system's lifetime. It is a LOGGING CAPABILITY duty. · who · actor, tenant, role, lawful basis · what · the question; answered or refused · how · index + document VERSIONS, model, prompt · Certain employment uses are listed in A…

Three minutes, and this is the slide most likely to be challenged by anyone who works near compliance. Be precise.

**ARTICLE 12 IS A LOGGING-CAPABILITY DUTY.** It requires high-risk systems to technically allow the automatic recording of events over the system's lifetime. It is not, by itself, the source of a retention number — resist the shorthand of "Article 12 means six months", which is how this gets misquoted.

**AND THE RECORD ON THE RIGHT IS OUR DESIGN, NOT A LEGAL SCHEMA.** The slide says so under the heading. Actor, tenant, lawful basis, prompt version, document versions, hashes — that is a practical audit record an engineer would build. Article 12's requirement for Annex III systems is considerably less prescriptive than that field list. Present it as "here is a shape that works", never as "here is what the Act requires you to log", because someone will check.

**THE DATE AND THE SCOPE.** Annex III high-risk obligations were originally due from 2 August 2026 and were postponed by the Digital Omnibus on AI to 2 December 2027 for standalone Annex III systems. If someone quotes August 2026 at you, that was the original schedule.

Phrase the scope the way the slide does: certain employment uses are LISTED in Annex III — recruitment and candidate selection, decisions affecting the employment relationship, task allocation, and worker monitoring or evaluation — and where such a system is classified high-risk, these requirements apply. Article 6(3) lets a provider conclude that a listed system does not pose a significant risk, subject to documenting that assessment. "All employment AI is high-risk" is wrong.

**RETENTION IS TWO BOUNDARIES, NOT ONE —** and note it does not come from Article 12.

The AI Act's retention duty sits in Articles 19 and 26(6): providers and deployers keep the logs a high-risk system automatically generates, to the extent they control them, "for a period appropriate to the intended purpose... of at least six months, unless provided otherwise in applicable Union or national law". Six months is a FLOOR, not a fixed period, and sector law — financial services, employment — routinely requires longer.

The other boundary is GDPR, which does NOT set a universal maximum. It requires that you keep personal data no longer than necessary for the purpose and that you can justify the period you chose. Worth noting the two are not really in conflict: the AI Act text itself defers to data-protection law in that same clause.

So: write down both boundaries and be able to defend each. That is a more useful instruction than any single number.

In this build the floor is ENFORCED rather than documented: `RIGHTS_RETENTION_DAYS` below 183 is a startup error that quotes Articles 19 and 26(6) back at you, and a longer period is accepted without argument because the floor is a floor. Worth showing if the room is compliance-adjacent — a policy nobody can violate by accident beats a policy in a wiki.

**REPRODUCIBILITY NEEDS VERSIONS, NOT JUST HASHES.** The record stores document ids, VERSION and a hash rather than a copy of the text. Be honest about what that buys: an id plus a hash proves the source changed, but on its own it does not let you reconstruct what the source said. The version does. If your corpus is not versioned, hashing gives you detection without recovery — which is worth knowing before an audit rather than during one.

**THE LOG IS NOW SENSITIVE.** The trace store holds user questions, retrieved text, model outputs and user ids. The thing that makes your system explainable is now one of the most sensitive stores you operate. Redact at capture, not in the UI — a redaction applied on read has already been exported by anyone with an API key.

AND THE HONEST LIMIT, which sets up the demo: a hash chain is tamper-EVIDENT, not tamper-proof. It catches a local edit. Someone who can rewrite the entire store can recompute every subsequent hash and produce a chain that verifies. To strengthen it you anchor outside the audit store: sign checkpoints, publish a periodic root hash somewhere separately controlled, use WORM storage, or ship records to a separately permissioned account.

Close on the takeaway: RAG expands the model's knowledge boundary; your trace store expands your data-protection boundary.

---

## 16 · Tamper with the audit log — then where to start

`0:45–0:50 · 5 min, then Q&A`

> **On screen** — DEMO 7/7 · Tamper with the audit log · before · chain intact — every record verified from genesis · edit one field · in the FIRST record — a decision from an hour ago · after · CHAIN BROKEN at record 0 — a local edit is detected. Anchor the chain elsewhere to survive a full-store rewrite. · one structured log line per request · TTFT + ITL as percentiles · 30 golden questions from real traffic · Monday: · determini…

DEMO 7 and the close. Five minutes.

Click "Tamper with the audit log".

Before: chain intact, every record verified from genesis. Then it edits one field in the FIRST record — a decision from an hour ago. After: CHAIN BROKEN at record 0, and every record after it is now unverifiable, because each one's hash depends on the one before.

The framing, and this is the line to end the technical content on: this is not a blockchain and does not need to be. Each record carries the hash of the one before it, so an edit cannot be made SILENTLY between the decision and the audit.

State the limit in the same breath, because it is the honest version and the previous slide set it up: this detects a local edit. An attacker who can rewrite the whole store can recompute every hash after the one they changed. The fix is to anchor outside the store — sign checkpoints, publish a periodic root hash to a separately controlled system, or write to WORM storage. Tamper-evident, not tamper-proof.

Then the Monday ladder along the bottom. Read the steps, not the descriptions. Step zero is one structured log line per request: free, an afternoon, and everything else needs it. If they do nothing else, that is it. Two — thirty golden questions from real traffic — and five — calibrate the judge — are where teams skip ahead and regret it.

Close: the repo runs with no API key, so trying it is genuinely five minutes.

Then set up session 6: the security of LLM-based agents. Make the connection explicit rather than leaving it as a topic announcement, because today's material is the foundation for it.

Everything we built today is also the detection layer. The audit record — who asked, what the system retrieved, which tools it could reach, what it decided — is exactly what you need to investigate an incident where a document told the model what to do. The trace is how you find the request where retrieved content became an instruction. And the refusal gate is a control, not just a quality feature.

Session 6 covers: prompt injection, direct and indirect, with retrieved content as the delivery mechanism; the confused-deputy problem when an agent holds credentials the user does not; least privilege for tool calls, and why tool permissions must come from application IAM and never from what a document says; sandboxing and egress control; data exfiltration through tool arguments; supply-chain risk in MCP servers and plugins; and where guardrails genuinely help versus where they are theatre.

The one-line bridge to use: "session 3 said RAG expands the model's knowledge boundary and must not expand its permission boundary. Session 6 is about what happens when someone tries to make it."

Thank the room and take questions.

Thank the room and take questions.

**LIKELY QUESTIONS.** Phoenix or Langfuse — either; they speak OTel, pick on deployment model. How much traffic to sample for quality — start where the judge cost is tolerable and check the confidence interval, not a fixed percentage. Can I skip the deterministic tests and just use a judge — no; a non-deterministic gate gets disabled. Do I need an agent framework for tracing — no, but you get span boundaries free. Is kappa the right statistic — the cheap defensible one for binary labels; Krippendorff's alpha for multiple raters or ordinal scores.

---

_Generated from the speaker notes embedded in `CoderCo-Session5-Evaluation-Observability.pptx`._
