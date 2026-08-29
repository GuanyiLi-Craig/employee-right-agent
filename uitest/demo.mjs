// Drives the dashboard exactly as the presenter will, and fails loudly on
// anything a room would notice. Every assertion is about what is on screen,
// not about what the API returned.
import { chromium } from 'playwright';
import { writeFileSync } from 'node:fs';

const BASE = process.env.BASE || 'http://localhost:8010';
const SHOTS = process.env.SHOTS || './shots';
const results = [];
let failures = 0;

function check(name, ok, detail = '') {
  results.push({ name, ok, detail });
  if (!ok) failures++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  — ${detail}` : ''}`);
}

const shot = async (page, name) =>
  page.screenshot({ path: `${SHOTS}/${name}.png`, fullPage: false });

async function jobDone(page, timeout = 240000) {
  await page.waitForFunction(
    () => document.querySelector('#job-status')?.textContent?.includes('idle'),
    null, { timeout, polling: 500 });
}

const consoleText = (page) => page.locator('#console').innerText();

(async () => {
  // Use the Chromium already in the machine's Playwright cache rather than
  // downloading another one.
  const browser = await chromium.launch(
    process.env.CHROME_BIN ? { executablePath: process.env.CHROME_BIN } : {});
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(`console: ${m.text()}`); });
  const failedRequests = [];
  page.on('requestfailed', (r) =>
    failedRequests.push(`${r.method()} ${r.url()} -> ${r.failure()?.errorText ?? '?'}`));

  // ---------------- warm the connection ----------------
  // The server warms itself at startup (see demo/app.py::warm_up), but a stack
  // that has been idle can have let its TLS connection go cold, and TTFT on that
  // request is not a measurement of anything. The question has to be one the
  // corpus answers: a refusal never reaches the model, so it warms nothing.
  await fetch(`${BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question: 'What does the document say about trade union recognition?',
      session_id: 'uitest-warmup',
    }),
  }).then((r) => r.text()).catch(() => {});

  // ---------------- load ----------------
  await page.goto(BASE, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => !document.querySelector('#h-index')?.textContent?.includes('…'),
    null, { timeout: 20000 });
  await shot(page, '01-loaded');

  check('page title', (await page.title()).includes('Employment rights agent'));
  const header = {
    index: await page.locator('#h-index').innerText(),
    model: await page.locator('#h-model').innerText(),
    gate: await page.locator('#h-gate').innerText(),
    trace: await page.locator('#h-trace').innerText(),
    audit: await page.locator('#h-audit').innerText(),
  };
  console.log('   header:', JSON.stringify(header));
  check('header shows the index version', /parser-\d\+/.test(header.index), header.index);
  check('header shows DeepSeek, not the stub', header.model.startsWith('deepseek-v4-flash'), header.model);
  check('model name is not greyed out (key works)',
    (await page.locator('#h-model').evaluate((e) => getComputedStyle(e).color)) !== 'rgb(232, 117, 122)');
  check('header shows the gate settings', /threshold/.test(header.gate), header.gate);
  check('tracing is exporting', /exporting/.test(header.trace), header.trace);

  // ---------------- reset to a clean slate ----------------
  await page.locator('button[data-job="reset"]').click();
  await jobDone(page);

  // ---------------- suggestion chips ----------------
  const chips = await page.locator('#suggest button').allInnerTexts();
  check('suggestion chips are offered', chips.length >= 4, `${chips.length} chips`);
  console.log('   chips:', JSON.stringify(chips));

  // ---------------- DEMO 1: ask, streamed ----------------
  const t0 = Date.now();
  await page.locator('#suggest button').first().click();
  // first token must appear well before the answer completes
  await page.waitForFunction(() => {
    const b = document.querySelectorAll('.msg.agent .bubble');
    return b.length && (b[b.length - 1].textContent || '').length > 0;
  }, null, { timeout: 60000, polling: 100 });
  const firstPaint = Date.now() - t0;
  await page.waitForSelector('.msg.agent .chips .chip', { timeout: 90000 });
  const answered = Date.now() - t0;
  await shot(page, '02-demo1-answer');

  check('answer streamed (first text before completion)', firstPaint < answered,
    `first paint ${firstPaint}ms, complete ${answered}ms`);
  const bubbles = await page.locator('.msg .bubble').allInnerTexts();
  check('the question and an answer are both on screen', bubbles.length >= 2, `${bubbles.length} bubbles`);
  check('answer is non-empty', bubbles[1].trim().length > 40, `${bubbles[1].length} chars`);
  check('no cursor artefact left behind', !(await page.locator('.bubble .cursor').count()));

  const cites = await page.locator('.msg.agent .cites').last().innerText();
  check('citations rendered', /\[s\.|\[Sch|\[Employment/.test(cites), cites.slice(0, 90));

  const chipText = (await page.locator('.msg.agent .chips').last().innerText()).replace(/\n/g, ' ');
  console.log('   chips:', chipText);
  for (const want of ['ttft', 'itl', 'e2e', 'tokens', 'cost', 'sufficiency', 'route', 'grounded', 'citations', 'intent']) {
    check(`chip: ${want}`, chipText.includes(want));
  }
  check('no fallback chip (DeepSeek really served)', !chipText.includes('→ stub-local'), chipText.slice(0, 120));
  const ttft = Number((chipText.match(/ttft ([\d.]+) ms/) || [])[1]);
  check('TTFT looks like a hosted model, not thinking', ttft > 50 && ttft < 8000, `${ttft} ms`);

  // stage bar + sources behind the disclosure
  await page.locator('.msg.agent details.detail summary').last().click();
  await page.waitForTimeout(200);
  const detail = await page.locator('.msg.agent details.detail').last().innerText();
  await shot(page, '03-demo1-detail');
  for (const want of ['generate', 'retrieve', 'orchestration', 'index']) {
    check(`detail: ${want}`, detail.includes(want));
  }
  check('detail lists retrieved provisions', /\d\.\d{3}\s+(leaf|widened)/.test(detail));
  check('detail shows the audit hash', /audit hash [0-9a-f]{16}/.test(detail));

  // ---------------- follow-up ----------------
  await page.locator('#question').fill('How long is it?');
  await page.locator('#send').click();
  await page.waitForFunction(() => document.querySelectorAll('.msg.agent .chips').length >= 2,
    { timeout: 90000 });
  await page.waitForTimeout(400);
  await shot(page, '04-followup');
  const followChips = (await page.locator('.msg.agent .chips').last().innerText()).replace(/\n/g, ' ');
  check('follow-up was resolved against the session', followChips.includes('follow-up resolved'), followChips.slice(0, 120));
  const followBubble = await page.locator('.msg.agent .bubble').last().innerText();
  check('follow-up was answered, not refused', !/cannot answer this/i.test(followBubble), followBubble.slice(0, 80));

  // ---------------- refusal ----------------
  const refusalChip = page.locator('#suggest button', { hasText: 'cryptocurrency' });
  await refusalChip.click();
  await page.waitForFunction(() => document.querySelectorAll('.msg.agent .chips').length >= 3,
    { timeout: 90000 });
  await page.waitForTimeout(300);
  await shot(page, '05-refusal');
  const refused = await page.locator('.msg.agent .bubble').last().innerText();
  check('out-of-scope question refused', /cannot answer this from the indexed document/i.test(refused),
    refused.slice(0, 100));
  check('refusal states its score and threshold', /scored 0\.\d+ against a threshold of 0\.\d+/.test(refused));
  check('refusal bubble is visually marked',
    await page.locator('.msg.agent .bubble.refused').count() > 0);

  writeFileSync(`${SHOTS}/../part1.json`, JSON.stringify({ results, errors, failedRequests, header, chips }, null, 2));
  console.log(`\nerrors: ${errors.length ? JSON.stringify(errors) : 'none'}`);
  console.log(`failed requests: ${failedRequests.length ? JSON.stringify(failedRequests) : 'none'}`);
  check('no page errors', errors.length === 0, errors.join('; ').slice(0, 200));
  // Chromium logs a completed streamed fetch as net::ERR_ABORTED once the reader
  // is done with it. Nothing is lost -- every answer above rendered in full --
  // so the assertion is on the thing that would matter: no *other* network
  // failure, and no truncated content.
  const realFailures = failedRequests.filter(
    (f) => !(f.includes('/api/chat ') && f.includes('ERR_ABORTED')));
  check('no unexpected network failures', realFailures.length === 0, realFailures.join('; ').slice(0, 200));
  const truncated = (await page.locator('.msg.agent .bubble').allInnerTexts())
    .filter((t) => t.trim().length < 20);
  check('no truncated answers', truncated.length === 0, `${truncated.length} short bubbles`);

  await browser.close();
  console.log(`\n=== part 1: ${results.length - failures}/${results.length} passed ===`);
  process.exit(failures ? 1 : 0);
})();
