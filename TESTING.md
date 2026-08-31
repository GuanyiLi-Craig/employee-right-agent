# How to test this yourself

Everything below has been run end to end on a clean checkout. Expected output is
quoted so you can tell a pass from a plausible-looking failure.

Nothing here needs an API key or the network — with one exception, noted where it
applies.

---

## 0. Which port?

Something on your machine may already own 8000, 6006 or 4317 — if so
`docker compose up` fails with `Bind for 0.0.0.0:4317 failed: port is already
allocated`. Every published port is overridable, and the reliable place to do it
is `.env`, which Compose loads automatically for **every** `docker compose`
command:

```bash
cat >> .env <<'ENV'
DASHBOARD_PORT=8010
PHOENIX_PORT=6016
PHOENIX_GRPC_PORT=4327
ENV
```

Exporting the variables in your shell works too, but you must export all three
in *every* shell you run a compose command from: a `docker compose run` that
sees different port values than the running stack will try to recreate Phoenix on
the original ports and fail. The rest of this document assumes 8010 / 6016.

---

## 1. Docker Compose — the full stack

The embedding pipeline is a **job**, not a service: it is the only writer of the
index, it runs to completion and exits, and no service ever starts one
implicitly.

```bash
docker compose build                       # ~2 min cold, seconds warm
docker compose --profile tools build evals # the dev image — NOT built by the line above

docker compose run --rm ingest             # the embedding pipeline
```

Expect:

```
index_version      parser-6+openai-text-embedding-3-small+bc461767
embedding_model    openai-text-embedding-3-small
corpus             ukpga_20250036_en.pdf (bc461767)
corpus_leaves      2078 rows
corpus_parents     546 rows
tree               document=1, heading=56, inserted=87, part=24, schedule=12, section=459, subsection=1971
built in           16.4s
```

That `index_version` is the same string a local `uv run rights-ingest --no-onnx`
produces on macOS. If yours differs, the corpus or the parser changed — not the
platform.

```bash
docker compose run --rm ingest-simple      # the fixed-window baseline (optional)
docker compose up -d                       # phoenix + dashboard
docker compose ps
```

Expect both services `(healthy)`:

```
SERVICE     STATUS
dashboard   Up 12 seconds (healthy)
phoenix     Up 7 minutes (healthy)
```

Then:

- dashboard → http://localhost:8010
- traces → http://localhost:6016 (project `rights-rag-agent`)

### 1.1 The contract that makes the pipeline separate

A reader with no index must fail with the command that fixes it, not with a
traceback:

```bash
docker volume create rights-empty
docker run --rm -v rights-empty:/var/lib/rights-agent/runs \
  -e RIGHTS_WAIT_FOR_INDEX=4 employment-rights-agent:local ask "anything"
echo "exit code: $?"
docker volume rm rights-empty
```

Expect exit code **2** and:

```
no index at /var/lib/rights-agent/runs/index_manifest.json after 4s.

The embedding pipeline runs separately from the query service:

    docker compose run --rm ingest          # hierarchical index (required)
    docker compose run --rm ingest-simple   # fixed-window baseline (optional)
```

### 1.2 Observability cannot break the app

```bash
docker compose stop phoenix
curl -s -X POST http://localhost:8010/api/ask -H 'Content-Type: application/json' \
  -d '{"question":"What does the document say about penalty notices?"}' | head -c 300
docker compose ps dashboard
docker compose start phoenix
```

Expect a normal, cited answer; the dashboard stays `(healthy)`; the only change
is that spans stop appearing in Phoenix. This is non-negotiable — if killing the
collector can break the agent, the instrumentation is a liability.

### 1.3 The other jobs

```bash
docker compose run --rm ask "How are tips allocated between workers?"
docker compose run --rm compare        # fixed windows vs. hierarchical
docker compose run --rm evaluate       # the CI gate's numbers, reported
docker compose run --rm evals          # the CI gate itself (pytest, 80 tests)
docker compose down                    # index volume survives
docker compose down -v                 # discard the index too
```

---

## 2. Local — `uv`

```bash
uv sync --extra trace --extra models --group dev
uv run rights-ingest --no-onnx          # < 2s
uv run rights-ask "What does the document say about bereavement leave?"
```

Expect an answer with citations, and beneath it:

```
citations   s.19, Employment Rights Act 1996 s.80EA (as inserted by s.23), s.20
gate        sufficiency 0.741 (threshold 0.45, attempts 0, route generate)
latency     ttft 2.39 ms · itl 2.03 ms (p95 2.27) · e2e 181.30 ms
            non-generation 16.09 ms · orchestration 5.35 ms · formula gap -82.75 ms
  generate  ████████████████████████████████████████  167.41 ms
  retrieve  ███                                        10.22 ms
tokens      prompt 855 (cached 0) · completion 142
cost        $0.001252  [cached_input_usd $0.000000 · input_usd $0.000684 · output_usd $0.000568]
model       stub-local (priced as claude-haiku-4-5, reference only) · prices as of 2026-08-01
scores      answer_relevance 0.75 · citation_coverage 1.00 · context_relevance 1.00 · groundedness 1.00
index       parser-6+openai-text-embedding-3-small+bc461767 · embedder openai-text-embedding-3-small
```

Note the second latency line. `formula gap` is negative because ITL is measured
per stream *chunk* while the multiplier counts *tokens* — the identity
`e2e ≈ TTFT + ITL×(n−1)` is an approximation on both sides, which is why the
measured numbers are the ones on the row.

An out-of-scope question must be **refused, with its numbers**:

```bash
uv run rights-ask "How do I mine cryptocurrency on company laptops?"
```

```
I cannot answer this from the indexed document. Retrieval sufficiency scored 0.16
against a threshold of 0.45, after 2 refinement attempt(s), so any answer would
not be supported by the source.
```

---

## 3. The test suites

```bash
uv run pytest -q                              # 475 tests, ~21s
uv run pytest tests/ -q                       # 395 unit tests, no index needed
uv run pytest evals/test_deterministic.py -q  # 55 tests: the merge gate
uv run pytest evals/test_quality.py -q        # 11 tests: aggregate thresholds
```

| Suite | Asserts | Needs an index |
|---|---|---|
| `tests/` | every module in isolation: parser traps, citations, embedder determinism, sufficiency arithmetic, context assembly, latency accounting, kappa, PSI and its epsilon, the cost components, the audit chain, follow-up resolution, both history tiers | no |
| `evals/test_deterministic.py` | **structural only** — manifest completeness, tree shape, embedder pinning, id uniqueness, refusals, expected citations, required metrics fields, `e2e ≥ Σ stages`, **audit-chain integrity**, chat streaming, chat history across both tiers, the dashboard's panels | yes |
| `evals/test_quality.py` | aggregates — mean **and p10** groundedness and citation coverage, context and answer relevance, and the judge's kappa **first** | yes |

Nothing in `test_deterministic.py` asks a model for an opinion. That is exactly
why it is allowed to fail a build.

### 3.1 The numbers, without the assertions

```bash
uv run python -m rights_agent evaluate --gate
```

```
gate                          observed  threshold  result
judge kappa                      0.764      0.600  PASS   gate the instrument first
groundedness mean                1.000      0.900  PASS
groundedness p10                 1.000      0.700  PASS   the tail, not the mean
citation coverage mean           1.000      0.900  PASS
context relevance mean           1.000      0.900  PASS   blames retrieval
answer relevance mean            0.819      0.700  PASS
refusal accuracy                 1.000      1.000  PASS
citation hit rate                1.000      1.000  PASS   expected citations retrieved
known failures                   4.000      4.000  PASS   must not grow
audit chain verifies            79.000     79.000  PASS   from genesis

GATE PASSED
```

### 3.2 Is the judge worth believing?

```bash
uv run python -m rights_agent evaluate --calibration
```

```
set                        n    kappa  agreement
clean examples only       12    1.000      1.000
plus realistic cases      17    0.764      0.882
groundedness alone        17    0.643      0.824

disagreements:
  h001   human 1 machine 0 score 0.00  correct paraphrase using none of the context's vocabulary
  h005   human 0 machine 1 score 1.00  correctly-cited boilerplate that does not answer the question
```

Read the three rows in order. Clean examples give a perfect kappa and tell you
nothing. The hard cases drop it to 0.76 — the judge did not get worse, the
measurement got honest. And scoring groundedness *alone* is worse still (0.64),
because an accurate-but-uncited answer looks fine to a support-only judge.

### 3.3 The eval datasets

37 golden rows (28 answerable, 5 out-of-scope, 4 known failures) and 17
calibration rows (12 clean, 5 hard), all generated against the live index and
verified before being written.

```bash
head -1 evals/datasets/openai-text-embedding-3-small/golden.jsonl
cat evals/datasets/openai-text-embedding-3-small/baseline.json
uv run python -m rights_agent goldens --write-baseline   # regenerate deliberately
```

Golden rows assert **citations, not prose**: wording changes when models change,
the source does not. `evals/datasets/<embedder>/baseline.json` holds the known-failure list and the
thresholds, kept out of the dataset it gates so that regenerating the dataset
cannot silently relax the gate.

---

## 3.4 The chat interface

```bash
docker compose up -d && open http://localhost:8010     # or: uv run rights-demo
```

Ask, then follow up. The second turn only works if the follow-up was resolved
against the first:

```bash
curl -sN -X POST localhost:8010/api/chat -H 'Content-Type: application/json' \
  -d '{"question":"What does the document say about bereavement leave?","session_id":"t1"}' \
  | tail -1 | python3 -m json.tool | head -20

curl -sN -X POST localhost:8010/api/chat -H 'Content-Type: application/json' \
  -d '{"question":"How long is it?","session_id":"t1"}' | tail -1 \
  | python3 -c "import json,sys; a=json.load(sys.stdin)['answer']; \
      print('history_used:', a['history_used'], '| borrowed:', a['contextualisation']['borrowed'], \
            '| refused:', a['refused'])"
```

```
history_used: True | borrowed: ['bereavement', 'leave'] | refused: False
```

The response is a stream of newline-delimited JSON: `start`, then one `token`
event per chunk, then a single `answer` event with the full record. That first
token event is what makes TTFT something you *watch* in the UI rather than a
figure reported after the fact.

### Chains of follow-ups

```bash
for q in "What does the document say about bereavement leave?" \
         "How long is it?" \
         "And for agency workers?"; do
  curl -sN -X POST localhost:8010/api/chat -H 'Content-Type: application/json' \
    -d "{\"question\":\"$q\",\"session_id\":\"chain\"}" | tail -1 \
    | python3 -c "import json,sys; a=json.load(sys.stdin)['answer']; \
        print(a['history_used'], a['contextualisation']['borrowed'], a['citations'][:1])"
done
```

```
False  []                          ['s.19']
True   ['bereavement', 'leave']    ['ERA 1996 s.80EB (as inserted by s.23)']
True   ['bereavement', 'leave']    ['s.6']
```

The third turn matters: it borrows from the *first* question, not the second. A
follow-up borrows from the last question that was not itself a follow-up —
otherwise "and for agency workers?" would inherit "long" from "how long is it?"
and retrieve nothing useful.

### History, and what survives a restart

```bash
curl -s localhost:8010/api/chat/sessions | python3 -m json.tool | head -20
```

Every conversation is listed with a `source` badge. Then restart and look again:

```bash
docker compose restart dashboard && sleep 14
curl -s localhost:8010/api/chat/sessions | python3 -c "
import json,sys
for x in json.load(sys.stdin)['sessions']:
    print(f\"  [{x['source']}] {x['requests']}q  {x['title'][:50]}\")"
```

```
  [audit] 3q  What does the document say about bereavement leave?
  [audit] 1q  How are tips allocated between workers?
```

The conversations survived; the badge flipped from `live` to `audit`. Reopen one:

```bash
curl -s "localhost:8010/api/chat/history?session_id=chain" | python3 -c "
import json,sys; d=json.load(sys.stdin); print('source:', d['source'])
for t in d['turns']: print(f\"  {t['role']:<6} rec={t['reconstructed']}  {t['content'][:60]}\")"
```

The questions and citations are there; each agent turn says the prose was not
retained. That is the design: transcripts live in memory, and the durable record
is the audit log with redaction applied at capture. Adding a second durable copy
of every model output would be a larger personal-data store under weaker
retention rules than the trace already has.

Two more things worth checking:

```bash
# synthetic traffic is audited but is not a conversation
curl -s -X POST localhost:8010/api/job -H 'Content-Type: application/json' \
  -d '{"job":"baseline_traffic"}' >/dev/null
sleep 12
curl -s localhost:8010/api/chat/sessions | python3 -c "
import json,sys; print('listed:', [s['session_id'] for s in json.load(sys.stdin)['sessions']])"
# ... no baseline-* session, while /api/audit?limit=1 shows the total climbing

# reopening restores follow-up context, so a chain works again after a restart
curl -s "localhost:8010/api/chat/history?session_id=chain" >/dev/null
curl -sN -X POST localhost:8010/api/chat -H 'Content-Type: application/json' \
  -d '{"question":"How long is it?","session_id":"chain"}' | tail -1 \
  | python3 -c "import json,sys; print('history_used:', json.load(sys.stdin)['answer']['history_used'])"
```

Two things to check deliberately:

```bash
# a follow-up in a FRESH session has nothing to borrow, and is left alone
curl -sN -X POST localhost:8010/api/chat -H 'Content-Type: application/json' \
  -d '{"question":"How long is it?","session_id":"fresh"}' | tail -1 \
  | python3 -c "import json,sys; print('history_used:', json.load(sys.stdin)['answer']['history_used'])"

# the transcript records both sides, and is what groups cost per conversation
curl -s "localhost:8010/api/chat/history?session_id=t1" | python3 -m json.tool | head
```

## 3.5 Running DeepSeek instead of the stub

```bash
printf 'RIGHTS_MODEL=deepseek-v4-flash\nDEEPSEEK_API_KEY=sk-…\n' >> .env
docker compose up -d --force-recreate dashboard
docker compose run --rm ask "What does the document say about bereavement leave?"
```

Check three things on the output:

```
model       deepseek-v4-flash (peak rate) · prices as of 2026-08-28
tokens      prompt 900 (cached 768) · completion 140
latency     ttft 480.12 ms · itl 18.40 ms
```

- **`model` is not `stub-local`.** If it is, the key was missing and it fell
  back. You do not have to read the log to find out:

  ```bash
  curl -s localhost:8010/api/state | python3 -c "
  import json,sys; r=json.load(sys.stdin)['runtime']
  print(r['model'], '| can actually serve:', r['model_available'])"
  ```

  and every message carries `requested_model`, `model` and `fallback`, so a
  failover is visible on the row rather than only in the log.
- **`cached` is climbing** across repeated requests. DeepSeek's cache-hit input
  is ~1/31 of cache-miss, so this is the single largest cost lever on this model.
- **TTFT is hundreds of milliseconds, not tens.** If it is *seconds*, thinking is
  probably on: `RIGHTS_THINKING` should be `false`, which is the default, and the
  parameter is sent explicitly because the provider's own default is `enabled`.

Reasoning can never leak into an answer, whether or not the flag is honoured:
it arrives in its own `reasoning_content` field, is excluded, and is counted.
The wire-level behaviour is pinned without a key or the network:

```bash
uv run pytest tests/test_deepseek.py -q      # 21 tests, fake SDK
```

**The gate always evaluates `stub-local`, whatever `.env` says.** That is
enforced in `evals/conftest.py`, not left to convention: a merge gate whose
result depends on whether an API key happened to be in the environment is not a
gate. Evaluate a hosted model deliberately and separately:

```bash
RIGHTS_MODEL=deepseek-v4-flash uv run python -m rights_agent evaluate
```

That costs money per run and is non-deterministic, which is exactly why it is
not what blocks a merge.

## 4. The dashboard, in the order to demo it

```bash
docker compose up -d && open http://localhost:8010
# or locally: uv run rights-demo
```

The seven demos from the deck, in order.

0. **Conversations** panel — past chats, newest first, each badged `live` or
   `audit`. Click one to reopen it; the active session is remembered across a
   page reload.

1. **Ask** an in-scope question — try the *qualifying period for unfair
   dismissal* chip. Watch the answer stream. The chips give TTFT, ITL, e2e,
   tokens, all-in cost, sufficiency and the four judged scores; *how this answer
   was produced* expands the stage bar and the retrieved provisions. Read the
   stage bar out loud: classify is effectively zero, retrieve is tens of
   milliseconds, generate is the overwhelming majority. **In this workload**
   retrieval is a rounding error on latency and still drives most of the quality
   decisions. Then ask the cryptocurrency chip and watch it refuse, with its
   score and threshold on screen.

2. **Baseline · 24**, then **Incident · 18**. The incident asks the *same 24
   questions* — the only thing that changes is how they are answered:

   On `deepseek-v4-flash` (measured):

   ```
                     baseline        incident
   ttft   p50          760.6          2145.9
   citation_coverage    1.000           0.059     mean
                        1.000           0.000     p10
   groundedness         0.963           0.911     ← moves far less
   ```

   Separately, each of those is what a busy team closes as "no repro". Together
   they point at one story. Be precise about what it establishes: the two series
   say *when* something changed and that the symptoms share a cause. They do not
   prove which model answered — the trace does.

   **Read the caveat off the Output panel, do not recite it.** The job measures
   both signals and prints the deltas, because how far groundedness moves depends
   on the model: an extractive one lifts sentences verbatim and a lexical judge
   stays happy, while a capable one paraphrases and the same judge marks it down.
   The durable claim — true either way, and what the panel shows — is that
   **citation coverage collapses far further than groundedness**. One headline
   quality number would have hidden that, which is the argument for a small
   family of signals.

   The panels are **windowed** (last 20 of N, labelled). That is not a
   convenience: a cumulative percentile cannot move during an incident, so the
   panel that should catch it stays green.

3. **Phoenix** at http://localhost:6016 — see §6.1. Then stop Phoenix and ask
   again: it still answers.

4. **Run the CI gate** — the numbers from §3.1, including `audit chain verifies`.
   It runs against `stub-local` even with DeepSeek configured, and says so at the
   bottom of its output: ~9s, offline, deterministic. A merge gate that depends on
   a hosted provider's latency and price list is not a gate.

5. **Calibrate the judge** — the three lines from §3.2, read in order.

6. **Shift intents · 20**, then **Drift report**:

   ```
   PSI over intents in BOTH windows     1.4640  (significant)
   PSI including unseen categories      2.1331  (significant, epsilon=0.0001)
   NEW intents                        harassment, unions
   ```

   Explain why there are two: PSI divides by the baseline probability, so an
   absent category sends it to infinity and every implementation smooths with an
   epsilon — which makes the figure depend on that constant. A single new intent
   can dominate it. The first figure is epsilon-free; the new-intent list needs
   no threshold at all and is usually the more actionable finding. This is not a
   bug report: it says the *questions* changed.

   Then **Reprice** with `claude-sonnet-5` selected — the table from §"Cost" in
   the README. Nothing is re-run; the token counts are already on the rows.

7. **Tamper with the audit log**:

   ```
   BEFORE   CHAIN INTACT — 99/99 records verified from genesis
   EDIT     record 0, field 'question'
   AFTER    CHAIN BROKEN at record 0
            verified 0/99 before the break
   ```

   Every record after 0 is now unverifiable, because each one's hash depends on
   the one before it. Say the limit in the same breath: this detects a *local*
   edit; someone who can rewrite the whole store can recompute every hash. The
   fix is to anchor outside the store — **Verify audit chain** writes exactly
   such a checkpoint, and says in its own output that a checkpoint stored beside
   the log it protects is a demonstration rather than a control.

8. **Reset** clears the metrics, restarts the chain from genesis and drops the
   transcripts.

Long jobs run on a worker thread; the page polls `/api/state` and never blocks.
Job failures land in the Output panel, not in a terminal nobody is projecting.

---

## 5. Break it on purpose

Each of these exercises a defence. Every command has been run; the quoted output
is what actually happens.

### 5.1 Why the sufficiency gate exists

```bash
RIGHTS_SUFFICIENCY=0.05 uv run rights-ask "How do I mine cryptocurrency on company laptops?"
```

With the gate dropped, you get a confident, correctly-cited answer about
"information about the workforce" and "offences by bodies corporate". Fluent,
attributable, and nonsense. The gate is the difference.

```bash
RIGHTS_SUFFICIENCY=0.95 uv run rights-ask "What does the document say about bereavement leave?"
```

Refuses at 0.78 against 0.95 — a threshold set too high refuses good questions,
which is why it is configuration and not a constant buried in code.

### 5.2 Embedder mismatch (pitfall 1)

```bash
cp runs/index_manifest.json /tmp/manifest.bak
python3 -c "
import json,pathlib
p=pathlib.Path('runs/index_manifest.json'); d=json.loads(p.read_text())
d['embedding_model']='onnx-all-MiniLM-L6-v2'; p.write_text(json.dumps(d,indent=2))"
uv run rights-ask "anything"; echo "exit: $?"
cp /tmp/manifest.bak runs/index_manifest.json
```

Exit **2**, and:

```
error: collection 'corpus_leaves' rejected embedder 'onnx-all-MiniLM-L6-v2': …
Cross-embedder queries return confident nonsense rather than an error, so this
is fatal. Set RIGHTS_EMBEDDER to match the index, or rebuild it.
```

### 5.3 Reused `thread_id` (pitfall 2)

```bash
uv run python - <<'PY'
from rights_agent.graph import AgentDeps, build_graph, initial_state
from rights_agent.config import settings
g = build_graph(AgentDeps(settings=settings()))
qs = ["What does the document say about bereavement leave?",
      "What is the rate of statutory sick pay?",
      "How are tips allocated between workers?",
      "What is a penalty notice?"]

print("partial state + one reused thread_id (the trap):")
for i, q in enumerate(qs, 1):
    out = g.invoke({"question": q}, config={"configurable": {"thread_id": "SAME"}})
    print(f"  {i}: attempts={out.get('attempts')} rewritten={str(out.get('rewritten_query'))[:44]!r}")

print("full state, unique thread per request (what Agent.ask does):")
for i, q in enumerate(qs, 1):
    out = g.invoke(initial_state(q), config={"configurable": {"thread_id": f"unique-{i}"}})
    print(f"  {i}: attempts={out['attempts']} refused={out['refused']}")
PY
```

The partial case leaks: by request 3, `attempts=1` and `rewritten_query` still
holds the *previous* question's rewrite. The full case does not. Both halves of
the fix matter — a unique thread per request, **and** every state field
initialised — and it is invisible in logs, which is why it is worth doing once
deliberately.

### 5.4 Context budget starvation (pitfall 3)

```bash
RIGHTS_CONTEXT_BUDGET=700 RIGHTS_MIN_BLOCK_CHARS=200 \
  uv run rights-ask "What does the document say about bereavement leave?" --show-context
```

The context ends `[…truncated]`. A naive assembler would `break` on the first
block that does not fit and hand the model an **empty** context — which it would
answer anyway, with nothing in the logs to say why.

### 5.5 Parent expansion cap (pitfall 4)

```bash
RIGHTS_MAX_PARENT_CHARS=300 uv run rights-ask "What does the document say about bereavement leave?"
```

Citations become leaf-level (`s.19(1)` instead of `s.19`): expansion was skipped
rather than allowed to consume the budget, and `expand_skipped` is recorded on
the document.

### 5.6 Retention below the floor is refused at startup

```bash
RIGHTS_RETENTION_DAYS=30 uv run rights-ask "anything"; echo "exit: $?"
```

Exit **2**, with the basis in the message: the AI Act's Articles 19 and 26(6)
require a period appropriate to the purpose and generally at least six months.
Six months is a floor, not a policy — `RIGHTS_RETENTION_DAYS=2555` is accepted.

### 5.7 A full-store rewrite is undetectable — on purpose

```bash
uv run pytest tests/test_audit.py -q -k rewritten -v 2>&1 | tail -4
```

`test_a_rewritten_store_verifies_and_that_is_the_honest_limit` asserts that
recomputing every hash produces a chain that verifies. The limitation is tested,
not just claimed in a docstring, because it is the whole reason the mitigation
is to anchor outside the store.

### 5.8 Break the parser (pitfall 7)

Delete the inserted-provision rule and watch which assertions catch it:

```bash
cp src/rights_agent/document/parser.py /tmp/parser.bak
python3 - <<'PY'
import pathlib
p = pathlib.Path("src/rights_agent/document/parser.py"); s = p.read_text()
new = s.replace("if inserted_match and len(inserted_match.group(2)) <= MAX_HEADING_CHARS:",
                "if False and inserted_match:")
assert new != s, "the line moved - find it and adjust"
p.write_text(new)
PY
uv run pytest tests/test_parser.py evals/test_deterministic.py -q
cp /tmp/parser.bak src/rights_agent/document/parser.py    # restore
```

Six assertions catch it:

```
FAILED tests/test_parser.py::test_quoted_pseudo_heading_is_not_a_cross_heading
FAILED tests/test_parser.py::test_inserted_provision_is_nested_and_names_its_host
FAILED tests/test_parser.py::test_consecutive_inserted_provisions_inherit_the_host_document
FAILED tests/test_parser.py::test_generated_corpus_clears_the_gate
FAILED evals/test_deterministic.py::test_leaf_count_is_sane
FAILED evals/test_deterministic.py::test_inserted_provisions_are_not_attributed_to_their_host
```

Now note which ones *do not*. With the break still in place,
`uv run pytest evals/test_quality.py -q` passes: **11 passed**. The judged
thresholds sail straight through, because mis-citing a provision does not make an
answer any less lexically grounded. A quality metric would never have caught
this; a structural assertion did.

### 5.9 Make a known failure pass

```bash
cp evals/datasets/openai-text-embedding-3-small/baseline.json /tmp/baseline.bak
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("evals/datasets/openai-text-embedding-3-small/baseline.json"); d = json.loads(p.read_text())
d["known_failures"] = [i for i in d["known_failures"] if i != "g027"]
p.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")
PY
uv run pytest evals/test_deterministic.py -q -k known
cp /tmp/baseline.bak evals/datasets/openai-text-embedding-3-small/baseline.json    # restore
```

```
FAILED evals/test_deterministic.py::test_known_failures_have_not_grown
1 failed, 1 passed, 53 deselected
```

`test_known_failures_have_not_grown` fails, naming `g027`. The reverse case is
gated too: if a recorded known failure starts passing,
`test_known_failures_have_not_silently_started_passing` fails so the marker gets
removed instead of quietly protecting nothing.

---

## 6. Verify the observability claims

### 6.1 The trace shape

With the stack up and Phoenix running, ask something, then:

```bash
PROJECT=$(curl -s http://localhost:6016/v1/projects \
  | python3 -c "import json,sys;print([p['id'] for p in json.load(sys.stdin)['data'] if p['name']=='rights-rag-agent'][0])")
curl -s "http://localhost:6016/v1/projects/$PROJECT/spans?limit=20" \
  | python3 -c "
import json,sys
for s in json.load(sys.stdin)['data']:
    print(f\"{s['name']:<24} {s['span_kind']:<10} parent={(s.get('parent_id') or '-')[:8]}\")"
```

You should see a root `rag-agent` and, beneath it, `rag.retrieve` (RETRIEVER),
`rag.generate` (LLM), `rag.assess`, `rag.judge` (EVALUATOR) — plus LangGraph's
own node spans, named after the nodes. The `rag.` prefix is what tells the two
instrumentation layers apart; without it you get same-named siblings and an
unreadable trace.

In the Phoenix UI, open `rag.retrieve`: it renders a **documents table** with
ids, scores and breadcrumbs, because the span carries
`retrieval.documents.N.document.*`. Open `rag.generate`: prompt and completion
token counts, so the cost number has an honest basis.

### 6.1.1 The golden set as a Phoenix dataset

```bash
make phoenix
```

Then check what landed, without opening the UI:

```bash
DS=$(curl -s localhost:6016/v1/datasets | python3 -c "
import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
curl -s "localhost:6016/v1/datasets/$DS/examples" | python3 -c "
import json,sys; d=json.load(sys.stdin)['data']
ex = d['examples'] if isinstance(d, dict) else d
print('examples:', len(ex)); print('input   :', ex[0]['input'])
print('output  :', ex[0]['output']); print('metadata:', ex[0]['metadata'])"
curl -s "localhost:6016/v1/datasets/$DS/experiments" | python3 -c "
import json,sys
for x in json.load(sys.stdin)['data']:
    print(x['id'], x.get('metadata', {}).get('model'), x.get('metadata', {}).get('index_version'))"
```

Expect 37 examples shaped as nested `input` / `output` / `metadata`, each tagged
with `index_version` and `prompt_version`, and one experiment per run tagged with
the model that served it. If the upload fails, the local summary still prints —
that fallback is the point, not a workaround.

### 6.2 The metrics log

```bash
python3 -c "
import json, pathlib
rows=[json.loads(l) for l in pathlib.Path('runs/metrics.jsonl').read_text().splitlines() if l.strip()]
print('rows:', len(rows))
print('every row has index_version:', all(r['index_version'] for r in rows))
print('non-refusals priced and timed:', all(r['cost_usd']>0 and r['ttft_ms']>0 for r in rows if not r['refused']))
print('e2e >= sum of stages:', all(r['e2e_ms']+1e-6 >= sum(r['stage_ms'].values()) for r in rows))
print('context never duplicated here:', all('context' not in r for r in rows))"
```

All four must be `True`. The last one is deliberate: retrieved context belongs in
the trace under its own retention rules, not copied into a third store.

---

## 7. Reproducibility

```bash
# the corpus is a pure function of its generator
uv run rights-corpus --out /tmp/corpus.check.txt && cmp data/corpus.layout.txt /tmp/corpus.check.txt && echo "corpus reproducible"

# two ingests produce the same index_version and the same chunk ids
uv run rights-ingest --no-onnx --quiet >/dev/null && cp runs/index_manifest.json /tmp/m1.json
uv run rights-ingest --no-onnx --quiet >/dev/null
python3 -c "
import json
a,b=[json.load(open(p)) for p in ('/tmp/m1.json','runs/index_manifest.json')]
assert a['index_version']==b['index_version'] and a['collections']==b['collections']
print('identical across rebuilds:', a['index_version'])"
```

`test_rebuilding_produces_identical_ids` and
`test_building_twice_produces_identical_rows` assert the same thing in CI.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Bind for 0.0.0.0:4317 failed: port is already allocated` | another stack owns the port | put the port overrides in `.env` (§0), not just in one shell |
| `error: no index manifest at …` (exit 2) | the embedding pipeline has not run | `docker compose run --rm ingest` |
| dashboard exits after ~90s | started before the index existed | run the ingest job, then `docker compose up -d dashboard` |
| `disk I/O error` from Chroma | SQLite on a network or virtualised mount | point `RIGHTS_RUNS_DIR` at local disk |
| `pdftotext is not installed` | PDF corpus, no poppler | `brew install poppler` / `apt-get install poppler-utils`, or use the layout `.txt` |
| answers changed after re-ingest | new `index_version` | it is on every metrics row — compare them |
| tracing chip says `disabled (…)` | Phoenix unreachable | expected and harmless; the agent keeps answering |
| stale answers from the dashboard after a re-ingest | the service opened the old index at startup | `docker compose up -d --force-recreate dashboard` |
| `error: RIGHTS_RETENTION_DAYS=… is below the …-day floor` | retention below Articles 19/26(6) | raise it, or document a basis in other law |
| `falling back to stub-local: DEEPSEEK_API_KEY is not set` | no key in the environment the container sees | put it in `.env`; compose passes it through |
| `'deepseek-v4-flash-offpeak' is a pricing row` | that id prices the same model at a different time of day | set `RIGHTS_MODEL=deepseek-v4-flash` and reprice against the off-peak row |
| answers arrive after several seconds | reasoning is on | `RIGHTS_THINKING=false` (the default) |
| `phoenix  unavailable (…)` on `make phoenix` | the experiments API moved again | the local summary above it is still valid; the mismatch is in `push_to_phoenix` |
| the experiment link says `<your Phoenix UI>` | the client saw the container-internal hostname | prefix it with your own Phoenix URL, e.g. `http://localhost:6016` |
| audit chip says `BROKEN at 0` | the tamper demo was run | press **Reset** to restart the chain from genesis |
| every follow-up refuses | the session id changed between turns | send the same `session_id`; the chat UI does this for you |

---

## 8.5 Browser-driven validation

The whole walkthrough is automated. It drives the real UI in Chromium and
asserts on what is *on screen*, not on what the API returned:

```bash
make ui-test UI_BASE=http://localhost:8010
# or, by hand:
cd uitest && npm install
BASE=http://localhost:8010 npm run all      # 38 + 51 checks
```

See [`uitest/README.md`](uitest/README.md), including how to reuse a Chromium
already in the machine's Playwright cache instead of downloading one.

`demo.mjs` covers the header, the suggestion chips, streamed first-paint before
completion, every metric chip, the disclosure with the stage bar and audit hash,
the follow-up, and the refusal. `demo2.mjs` runs each control through the real
button and asserts the beat: TTFT jumping, citation coverage collapsing, the
p10 hitting zero, the gate's rows, the three kappa lines, the PSI pair plus the
new-intent list, the reprice table, `CHAIN BROKEN at record 0`, the audit panel
turning red, and history badging.

Screenshots land in `shots/` — worth a glance before presenting, since they are
what the projector will show.

One known artefact: Chromium logs a *completed* streamed fetch as
`net::ERR_ABORTED` once the reader is finished with it. Nothing is lost — every
answer renders in full, curl gets a clean `200` with correct chunked framing —
and the harness asserts on the thing that would matter instead: no other network
failure, and no truncated answers.

## 8.6 Security

```bash
make pentest                      # nuclei: this project's templates, then the community library
```

On a loopback-only demo the four app templates fire **by design** — no credential
is required and none should be. The test is that they close when the control is
on:

```bash
RIGHTS_DEMO_TOKEN=some-secret docker compose up -d --force-recreate dashboard
make pentest                      # -> No results found.
docker compose up -d --force-recreate dashboard   # back to the local posture
```

Confirm the exposure is actually gone:

```bash
docker compose ps --format "{{.Service}}\t{{.Ports}}"   # want 127.0.0.1:8010->8000
curl --max-time 4 "http://$(ipconfig getifaddr en0):8010/api/health"   # want: refused
```

Findings, fixes and accepted risk: [`security/README.md`](security/README.md).

## 9. Presenting from `presentation/`

The deck's seven demos map to these controls. All were run end to end against
the stack on the ports in §0.

| # | Slide | Control | The beat |
|---|---|---|---|
| 1 | 4 | type a question, or a suggestion chip | streams; chips give TTFT / ITL / e2e / cost / citations; *how this answer was produced* has the stage bar |
| 2 | 6 | **Baseline · 24** → **Incident · 18** | TTFT p50 761 → 2146 ms; citation coverage 1.000 → 0.059, p10 0.000; groundedness moves far less — **read the deltas off the Output panel** |
| 3 | 8 | Phoenix at :6016, then stop Phoenix | span tree with `rag.retrieve` / `rag.generate`; the agent keeps answering |
| 4 | 10 | **Run the CI gate** | ten gates including `audit chain verifies`, ~9s, offline on `stub-local` even when DeepSeek is serving |
| 5 | 12 | **Calibrate the judge** | 1.000 → 0.764 → 0.643, read in that order |
| 6 | 14 | **Shift intents · 20** → **Drift report** → **Reprice** | PSI known vs with-unseen vs new-intent list; Haiku vs Sonnet ~2.7× |
| 7 | 16 | **Tamper with the audit log** | CHAIN INTACT 99/99 → CHAIN BROKEN at record 0, verified 0/99 |

Two things worth knowing before you stand up:

- **The corpus is the real Act** — Employment Rights Act 2025 (c. 36), 335 pages.
  If the deck says 334, that is the only correction it needs. The generated Act
  is still in `data/` as the offline fallback and as the fixture the parser unit
  tests run against.
- **Reprice shows ~2.7×, not 3×.** The model lines are exactly 3× — the
  model-independent components (judge, trace storage, infrastructure) dilute the
  total. The panel shows the components, so this is a better beat than the round
  number, not a worse one.
- **Timings are DeepSeek's, not the stub's.** Baseline · 24 takes ~35 s and
  Incident · 18 ~45 s, both with a live progress counter. Budget for that in
  demo 2 rather than talking over silence.
- **The corpus cannot answer "the qualifying period for unfair dismissal".** The
  suggestion chips are all checked against the live index by
  `test_every_suggestion_behaves_as_advertised`; type something off-piste and you
  may get a well-earned refusal.

Before the room fills:

```bash
printf 'DASHBOARD_PORT=8010\nPHOENIX_PORT=6016\nPHOENIX_GRPC_PORT=4327\n' >> .env
docker compose build && docker compose --profile tools build evals
docker compose run --rm ingest && docker compose run --rm ingest-simple
docker compose up -d
open http://localhost:8010          # click Baseline · 24 once so the panels are not empty
open http://localhost:6016          # traces, on a second tab
```

`Reset` between rehearsals: it clears the metrics, restarts the audit chain from
genesis and drops the transcripts.
