// Part 2: the six control-driven demos, the panels, and chat history.
import { chromium } from 'playwright';
import { writeFileSync } from 'node:fs';

const BASE = process.env.BASE || 'http://localhost:8010';
const SHOTS = process.env.SHOTS || './shots';
const results = [];
let failures = 0;
const check = (name, ok, detail = '') => {
  results.push({ name, ok, detail });
  if (!ok) failures++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  — ${detail}` : ''}`);
};
const shot = (page, n) => page.screenshot({ path: `${SHOTS}/${n}.png` });

// Clicks the real button, then polls the job's own output entry. Not
// waitForFunction: an async predicate returns a Promise, which is always
// truthy, so it resolves on the first poll regardless of the answer.
async function state(page) {
  return page.evaluate(async () => (await fetch('/api/state')).json());
}

async function runJob(page, job, timeout = 300000) {
  const entryOf = (s) => (s.output || []).find((o) => o.job === job);
  const before = entryOf(await state(page))?.finished_at ?? 0;

  await page.locator(`button[data-job="${job}"]`).click();

  const deadline = Date.now() + timeout;
  let last = '';
  while (Date.now() < deadline) {
    const s = await state(page);
    const entry = entryOf(s);
    if (!s.job?.running && entry && entry.finished_at !== before) {
      if (!entry.ok) throw new Error(`job ${job} FAILED: ${String(entry.text).slice(0, 400)}`);
      await page.waitForTimeout(2200);   // let the panels catch up
      return entry.text;
    }
    if (s.job?.progress && s.job.progress !== last) {
      last = s.job.progress;
      process.stdout.write(`   …${job}: ${last}\r`);
    }
    await page.waitForTimeout(700);
  }
  throw new Error(`job ${job} did not finish within ${timeout}ms`);
}

const panel = (page, id) => page.locator(`#${id}`).innerText();

(async () => {
  const browser = await chromium.launch(
    process.env.CHROME_BIN ? { executablePath: process.env.CHROME_BIN } : {});
  const page = await browser.newPage({ viewport: { width: 1600, height: 1100 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(`console: ${m.text()}`); });

  await page.goto(BASE, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => !document.querySelector('#h-index')?.textContent?.includes('…'),
    null, { timeout: 20000 });

  await runJob(page, 'reset');
  check('a completed job appears in the Output panel',
    /reset/.test(await page.locator('#console').innerText()));

  // ---------- DEMO 2: baseline then incident ----------
  await runJob(page, 'baseline_traffic');
  await page.waitForTimeout(2200);
  const baseLat = await panel(page, 'latency-body');
  const baseQual = await panel(page, 'quality-body');
  await shot(page, '06-demo2-baseline');
  const num = (text, row, col) => {
    const line = text.split('\n').find((l) => l.startsWith(row));
    return line ? Number(line.split(/\s+/)[col]) : NaN;
  };
  const baseTtftP50 = num(baseLat, 'ttft', 1);
  const baseCitMean = num(baseQual, 'citation coverage', 2);
  console.log(`   baseline: ttft p50 ${baseTtftP50}, citation coverage mean ${baseCitMean}`);
  check('baseline filled the latency panel', Number.isFinite(baseTtftP50) && baseTtftP50 > 0);
  check('baseline citation coverage is high', baseCitMean >= 0.7, String(baseCitMean));

  const incident = await runJob(page, 'incident_traffic');
  await page.waitForTimeout(2200);
  const incLat = await panel(page, 'latency-body');
  const incQual = await panel(page, 'quality-body');
  await shot(page, '07-demo2-incident');
  const incTtftP50 = num(incLat, 'ttft', 1);
  const incCitMean = num(incQual, 'citation coverage', 2);
  const incCitP10 = num(incQual, 'citation coverage', 3);
  const incGround = num(incQual, 'groundedness', 2);
  console.log(`   incident: ttft p50 ${incTtftP50}, citation mean ${incCitMean}, p10 ${incCitP10}, groundedness ${incGround}`);
  check('DEMO 2: TTFT p50 jumps', incTtftP50 > baseTtftP50 * 1.5, `${baseTtftP50} -> ${incTtftP50}`);
  check('DEMO 2: citation coverage collapses', incCitMean < baseCitMean - 0.2, `${baseCitMean} -> ${incCitMean}`);
  check('DEMO 2: the p10 goes to zero', incCitP10 === 0, String(incCitP10));
  // The durable claim, true for an extractive stub and a paraphrasing model
  // alike: citations collapse much further than groundedness. How far
  // groundedness itself moves depends on whether the model quotes or paraphrases.
  const baseGround = num(baseQual, 'groundedness', 2);
  // Direction, not a multiplier. How far groundedness moves depends on how much
  // the model paraphrases, which varies run to run; that citations fall further
  // is the claim that holds for an extractive stub and a paraphrasing model
  // alike. The precise deltas come from the job's own measured comparison,
  // which contrasts the healthy and degraded populations rather than a mixed
  // rolling window.
  check('DEMO 2: citations collapse further than groundedness',
    (baseCitMean - incCitMean) > (baseGround - incGround),
    `citations -${(baseCitMean - incCitMean).toFixed(3)} vs groundedness -${(baseGround - incGround).toFixed(3)}`);
  const measured = incident.split('\n').filter((l) => l.includes('->'));
  const delta = (metric) => {
    const line = measured.find((l) => l.includes(metric));
    return line ? Number((line.match(/\(([-+][\d.]+)\)/) || [])[1]) : NaN;
  };
  check('DEMO 2: the measured comparison agrees',
    delta('citation_coverage') < delta('groundedness'),
    measured.join(' ; ').slice(0, 120));
  check('DEMO 2: output measures both signals rather than scripting one',
    /The two quality signals did not move together/.test(incident)
    && /citation_coverage\s+[\d.]+ -> [\d.]+/.test(incident),
    incident.split('\n').filter((l) => l.includes('->')).join(' ; ').slice(0, 120));
  check('DEMO 2: output explains why one headline number hides it',
    /family of signals/.test(incident));

  // ---------- DEMO 4: the CI gate ----------
  const t0 = Date.now();
  const gate = await runJob(page, 'ci_gate');
  const gateSecs = (Date.now() - t0) / 1000;
  await shot(page, '08-demo4-gate');
  console.log(`   gate ran in ${gateSecs.toFixed(1)}s`);
  check('DEMO 4: gate passed', /GATE PASSED/.test(gate), gate.split('\n').slice(-4).join(' | ').slice(0, 120));
  check('DEMO 4: gate is under a minute', gateSecs < 60, `${gateSecs.toFixed(1)}s`);
  check('DEMO 4: gate ran offline, not against DeepSeek', /Run against stub-local/.test(gate));
  for (const row of ['judge kappa', 'groundedness p10', 'citation hit rate', 'audit chain verifies']) {
    check(`DEMO 4 gate row: ${row}`, gate.includes(row));
  }
  check('DEMO 4: no FAIL rows', !/\bFAIL\b/.test(gate));

  // ---------- DEMO 5: calibrate ----------
  const cal = await runJob(page, 'calibrate_judge');
  await shot(page, '09-demo5-calibrate');
  const kappa = (label) => {
    const line = cal.split('\n').find((l) => l.includes(label));
    return line ? Number(line.trim().split(/\s+/).slice(-2)[0]) : NaN;
  };
  const clean = kappa('clean examples only');
  const hard = kappa('plus realistic cases');
  const ground = kappa('groundedness alone');
  console.log(`   kappa: clean ${clean}, +hard ${hard}, groundedness-only ${ground}`);
  check('DEMO 5: clean examples give a perfect kappa', clean === 1, String(clean));
  check('DEMO 5: hard cases lower it', hard < clean, `${clean} -> ${hard}`);
  check('DEMO 5: the composite beats groundedness alone', hard > ground, `${hard} vs ${ground}`);
  check('DEMO 5: the paraphrase case is named', /paraphrase/i.test(cal));

  // ---------- DEMO 6: intents, drift, reprice ----------
  await page.selectOption('#intent-select', 'enforcement');
  await runJob(page, 'shift_intents');
  const drift = await runJob(page, 'drift_report');
  await shot(page, '10-demo6-drift');
  check('DEMO 6: PSI over shared intents reported', /PSI over intents in BOTH windows/.test(drift));
  check('DEMO 6: PSI with unseen reported with its epsilon', /including unseen categories.*epsilon=/s.test(drift));
  check('DEMO 6: new intents listed explicitly', /NEW intents\s+\S/.test(drift));
  check('DEMO 6: the epsilon gotcha is explained', /depend on the constant|smoothing constant/i.test(drift));
  check('DEMO 6: bands flagged as a convention', /not a law of nature/.test(drift));

  await page.selectOption('#model-select', 'claude-sonnet-5');
  const reprice = await runJob(page, 'reprice');
  await shot(page, '11-demo6-reprice');
  check('DEMO 6: reprice shows both models', /deepseek-v4-flash|claude-haiku/.test(reprice) && /claude-sonnet-5/.test(reprice));
  check('DEMO 6: reprice names the dominant component', /dominant component is generation_/.test(reprice));
  check('DEMO 6: reprice says nothing was re-run', /Nothing was re-run/.test(reprice));
  check('DEMO 6: monthly projection with its assumption', /monthly projection/.test(reprice) && /requests\/day/.test(reprice));

  // ---------- DEMO 7: tamper ----------
  const auditBefore = await panel(page, 'audit-body');
  check('audit panel shows an intact chain before tampering', /intact/.test(auditBefore), auditBefore.replace(/\n/g, ' | ').slice(0, 90));
  const tamper = await runJob(page, 'tamper_audit');
  await page.waitForTimeout(2200);
  await shot(page, '12-demo7-tamper');
  check('DEMO 7: before shows the chain intact', /BEFORE[\s\S]*CHAIN INTACT/.test(tamper));
  check('DEMO 7: after shows it broken at record 0', /AFTER[\s\S]*CHAIN BROKEN at record 0/.test(tamper));
  check('DEMO 7: it states the limitation', /tamper-evident, not tamper-proof/i.test(tamper));
  const auditAfter = await panel(page, 'audit-body');
  const headerAudit = await page.locator('#h-audit').innerText();
  check('DEMO 7: the audit panel turns red', /BROKEN at 0/.test(auditAfter), auditAfter.replace(/\n/g, ' | ').slice(0, 90));
  check('DEMO 7: the header chip says BROKEN', /BROKEN/.test(headerAudit), headerAudit);

  const recovered = await runJob(page, 'verify_audit');
  check('reset path: verify writes a checkpoint', /checkpoint written/.test(recovered));
  await runJob(page, 'reset');
  await page.waitForTimeout(2000);
  check('after reset the chain is clean', !/BROKEN/.test(await page.locator('#h-audit').innerText()));

  // ---------- history ----------
  await page.locator('#question').fill('What is the threshold for a penalty notice?');
  await page.locator('#send').click();
  await page.waitForSelector('.msg.agent .chips .chip', { timeout: 90000 });
  await page.waitForTimeout(6500);
  await shot(page, '13-history');
  const sessions = await page.locator('#sessions .sess').count();
  check('history lists the conversation', sessions >= 1, `${sessions} listed`);
  const sessText = await panel(page, 'sessions');
  check('history row is badged live', /live/i.test(sessText), sessText.replace(/\n/g, ' | ').slice(0, 110));
  check('history excludes synthetic traffic', !/baseline-|incident-|intent-shift/.test(sessText));

  await page.locator('#new-chat').click();
  await page.waitForTimeout(800);
  check('New starts an empty session', /New conversation/.test(await panel(page, 'transcript')));
  await page.locator('#sessions .sess').first().click();
  await page.waitForTimeout(1200);
  await shot(page, '14-history-reopened');
  const reopened = await panel(page, 'transcript');
  check('clicking a history row reopens it', /penalty notice/i.test(reopened), reopened.slice(0, 80));

  // ---------- cost panel ----------
  const cost = await panel(page, 'cost-components');
  const costRows = await panel(page, 'cost-body');
  await shot(page, '15-cost');
  for (const row of ['generation input', 'generation output', 'judge', 'trace storage', 'infrastructure']) {
    check(`cost component: ${row}`, cost.includes(row));
  }
  check('cost panel shows a monthly projection with its volume', /monthly at [\d,]+\/day/.test(costRows), costRows.replace(/\n/g, ' | ').slice(0, 120));
  check('cost note labels the judge line as modelled', /modelled/.test(await panel(page, 'cost-note')));

  writeFileSync(`${SHOTS}/../part2.json`, JSON.stringify({ results, errors }, null, 2));
  console.log(`\nerrors: ${errors.length ? JSON.stringify(errors) : 'none'}`);
  check('no page errors', errors.length === 0, errors.join('; ').slice(0, 200));
  await browser.close();
  console.log(`\n=== part 2: ${results.length - failures}/${results.length} passed ===`);
  process.exit(failures ? 1 : 0);
})();
