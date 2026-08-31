# Browser-driven validation

Drives the dashboard in a real browser and asserts on **what is on screen**, not
on what the API returned. Written because three defects in this project were only
visible that way: degraded mode doing nothing for a hosted model, citation
coverage reading zero while every citation was correct, and a stage-bar label
clipped to `orchestratio`.

```bash
cd uitest
npm install                     # or: npm ci
export BASE=http://localhost:8010            # match DASHBOARD_PORT
npm run all
```

If Playwright complains that a browser is missing, either `npx playwright install
chromium`, or point at one already in the machine's cache:

```bash
export CHROME_BIN=$(ls ~/Library/Caches/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell | head -1)
```

| script | checks | covers |
|---|---|---|
| `demo.mjs` | 38 | header, suggestion chips, streamed first paint before completion, every metric chip, the disclosure (stage bar, retrieved provisions, audit hash), a follow-up, a refusal |
| `demo2.mjs` | 51 | demos 2–7 through the real buttons: TTFT jumping, citation coverage collapsing, the p10 hitting zero, the gate's rows, the three kappa lines, the PSI pair and new-intent list, repricing, `CHAIN BROKEN at record 0`, the audit panel turning red, chat history badging and reopen, the five cost components |

Screenshots land in `shots/` — worth a glance before presenting, since that is
what the projector will show.

## One known artefact

Chromium logs a **completed** streamed fetch as `net::ERR_ABORTED` once the
reader is done with it. Nothing is lost: every answer renders in full and curl
gets a clean `200` with correct chunked framing. The harness therefore asserts on
what would matter — no *other* network failure, and no truncated answers — rather
than on a clean network log.

## `index-explorer.mjs`

Twelve checks on the **Index** panel: that it lists chunks from the live index,
that clicking one shows its metadata, that the embedded text starts with the
breadcrumb, that the vector is drawn and says how many dimensions it is hiding,
and that typing a question ranks by cosine similarity.

```bash
node index-explorer.mjs
```

## `phoenix-annotations.mjs`

Three checks that the judged scores reach Phoenix as **span annotations** and
render in the project sidebar. Needs Phoenix on `PHOENIX` (default
`http://localhost:6016`) and at least one recorded request.

```bash
node phoenix-annotations.mjs
```
