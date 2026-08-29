# Security assessment

YAML-template scanning with [Nuclei](https://github.com/projectdiscovery/nuclei),
run against the dashboard on a laptop. Two passes: the community library for the
generic web surface, and templates written for *this* application's own risks.

```bash
make pentest                    # both passes, against UI_BASE
```

## Pass 1 — the community library

10,689 templates · 18,651 requests · **13 matches, all `info`**: technology
detection, form detection, and missing security headers. No injection, no path
traversal, no exposed panel, no CVE.

That is a real result rather than luck. The dashboard is `http.server` plus one
static page: there is no template engine, no SQL, no user-controlled filesystem
path, and no dependency stack to have a CVE in. The headers finding was worth
fixing and is fixed.

## Pass 2 — templates for this application

`nuclei/*.yaml`. These are **conditional on reachability**, which is the whole
point: every one of them asks "can this be done without a credential", and on a
loopback-only demo the answer is *yes, and that is correct*. Run them against the
address you actually publish on.

| template | severity | what it proves |
|---|---|---|
| `unauth-job-execution` | high | `POST /api/job` runs `reset` and `tamper_audit` — deleting and corrupting the audit record — with no credential |
| `unauth-audit-log-read` | high | `GET /api/audit` returns the compliance log: actor, tenant, lawful basis, redacted question, citations, hashes |
| `unauth-state-mutation` | medium | `POST /api/degraded` degrades every later answer for everyone |
| `guessable-session-id` | medium | a transcript readable from an id an attacker could enumerate |
| `missing-security-headers` | low | no CSP, nosniff, frame-options or referrer-policy |

Result after the fixes below:

```
exposed posture (RIGHTS_DEMO_TOKEN set)   No results found.
local posture   (no token, loopback)      4 matches — by design
```

## What was fixed

**The published port is loopback only.** `docker-compose.yml` publishes
`127.0.0.1:8000` rather than `0.0.0.0`. It was reachable from the LAN before;
it now refuses:

```
http://192.168.x.x:8010 -> connection refused
```

Phoenix gets the same treatment, and for a stronger reason: the dashboard exposes
a redacted audit log, while the **trace store holds the questions, the retrieved
text and the answers**. RAG expands the model's knowledge boundary; the trace
store expands your data-protection boundary.

**The server refuses to run wide open.** Binding a non-loopback address with no
`RIGHTS_DEMO_TOKEN` is a startup error naming both ways out. `/api/job` can erase
the audit chain, so "reachable from the network with no credential" is not a
default worth having. `RIGHTS_DEMO_ALLOW_INSECURE=true` overrides it — which is
what compose sets, because there the control is the loopback publish and the
container's `0.0.0.0` bind is an implementation detail.

**`RIGHTS_DEMO_TOKEN` gates the dangerous subset** — `/api/job`,
`/api/degraded`, `/api/chat/reset`, `/api/audit` — and nothing else. Reading the
page, asking a question and reading `/api/state` stay open: the token exists to
stop the audit record being erased, not to put a login on a demo. No default
value ships, because a shared secret printed in a public compose file is a fake
control, and a fake control is worse than an absent one.

**Session ids come from a CSPRNG.** They were
`Math.random().toString(36).slice(2, 10)` — about 41 bits from a generator that
is not cryptographically secure, and the traffic controls minted
`baseline-<unix seconds>`. A transcript is readable by anyone holding its id, so
the id has to be a secret: it is now 128 bits from `crypto.getRandomValues`.

**The audit store had a side door.** `/api/audit` was protected while
`/api/chat/history` reconstructed the same records for anyone with a session id.
Reading the audit tier now requires the same token. Found because
`guessable-session-id` kept matching after the live transcript had been
cleared — the template caught a hole in the fix for the finding above it.

**Concurrent chats are bounded** (`RIGHTS_DEMO_MAX_CONCURRENT_CHATS`, default 4).
Each one is a thread and a billable model call, so an unbounded endpoint is an
unbounded invoice as well as unbounded threads.

**Security headers** on every response, including the streamed one: CSP with
`frame-ancestors 'none'`, nosniff, `X-Frame-Options: DENY`, `no-referrer`, and
cross-origin isolation.

Each control has an assertion in `evals/test_deterministic.py` — a control that
is not asserted is a control that comes back off in the next refactor.

## Accepted, and why

- **No user accounts.** This is a single-operator demo. The token is a shared
  secret, not identity, and the audit record's `actor` is asserted by the caller
  rather than authenticated. Any real deployment needs real identity before the
  audit record means anything about *who* asked.
- **No TLS.** Loopback. Exposing it means terminating TLS in front, at which
  point the token stops travelling in clear.
- **A session id is a bearer secret in a URL.** 128 bits resists guessing, but it
  still reaches server logs and `Referer`. Real per-user auth is the fix, not
  more entropy.
- **The corpus is trusted.** Retrieved text goes into the prompt, so a corpus an
  attacker can write to is an indirect prompt-injection channel. Here the corpus
  is committed to the repository and the index is built by a separate job, so the
  trust boundary is the repository. That assumption is exactly what
  [session 6](../presentation/) is about.
- **`/api/chunks` is unauthenticated, on purpose.** The index panel reads the
  embedded corpus — public UK legislation, already returned verbatim in every
  answer's "retrieved provisions" detail — plus each chunk's metadata and its
  embedding vector. It changes no state and reads the index rather than the audit
  record, so it is not a side door around `/api/audit` the way `/api/chat/history`
  was. Two things would change that judgement: a corpus that is not public, or an
  embedding model whose vectors are themselves sensitive. Both are worth
  re-checking before this pattern is copied to a private document set.
- **Prompt injection in the question is not prevented**, only bounded: the
  sufficiency gate refuses off-corpus questions, the answer is scored for
  groundedness and citation coverage, and every request lands in the audit
  record. Those are detection and blast-radius controls, not prevention.

## Reproducing

```bash
# one-off: fetch nuclei and its templates
curl -sL -o /tmp/nuclei.zip \
  https://github.com/projectdiscovery/nuclei/releases/download/v3.11.1/nuclei_3.11.1_macOS_arm64.zip
unzip -o -q /tmp/nuclei.zip -d /tmp && /tmp/nuclei -update-templates

# this project's templates, against whatever you publish
/tmp/nuclei -u http://localhost:8010 -t security/nuclei/

# prove the controls close
RIGHTS_DEMO_TOKEN=some-secret docker compose up -d --force-recreate dashboard
/tmp/nuclei -u http://localhost:8010 -t security/nuclei/     # No results found.
```
