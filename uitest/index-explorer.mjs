import { chromium } from 'playwright';
const BASE = process.env.BASE || 'http://localhost:8010';
let pass = 0, fail = 0;
const check = (name, ok, extra = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${extra ? '  — ' + extra : ''}`);
  ok ? pass++ : fail++;
};
const browser = await chromium.launch();
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
await page.goto(BASE, { waitUntil: 'networkidle' });

await page.waitForSelector('#chunks .chunk', { timeout: 15000 });
const rows = await page.locator('#chunks .chunk').count();
check('index panel lists chunks', rows > 0, `${rows} rows`);
const sub = await page.locator('#idx-sub').innerText();
check('subtitle names the count and embedder', /chunks · openai/.test(sub), sub);

await page.locator('#chunks .chunk').first().click();
await page.waitForSelector('#idx-detail .kv', { timeout: 15000 });
const kv = await page.locator('#idx-detail').innerText();
check('detail shows the chunk id', /chunk id/.test(kv));
check('detail shows index_version', /parser-6\+openai/.test(kv), (kv.match(/parser-6\S*/) || [''])[0]);
check('detail shows dimensions 1536', /1536/.test(kv));
check('detail shows the L2 norm', /L2 norm/.test(kv));
const crumb = await page.locator('#idx-detail .idx-embedded .crumb').innerText();
check('embedded text starts with the breadcrumb', crumb.includes('>'), crumb.slice(0, 60));
const bars = await page.locator('#idx-detail .vec i').count();
check('vector is drawn', bars === 16, `${bars} bars`);
const nums = await page.locator('#idx-detail .vec-nums').innerText();
check('vector numbers say how many are hidden', /more\]/.test(nums), nums.slice(0, 70));

await page.fill('#idx-q', 'bereavement leave');
await page.click('#idx-go');
await page.waitForFunction(() => document.querySelector('#idx-note')?.textContent?.includes('cosine'), null, { timeout: 15000 });
const note = await page.locator('#idx-note').innerText();
check('search explains the ranking', /cosine similarity/.test(note), note.slice(0, 72));
const scored = await page.locator('#chunks .chunk .s').first().innerText();
check('search rows show a similarity score', /^0\.\d+$/.test(scored.trim()), scored);

check('no page errors', errors.length === 0, errors.join(' | '));
console.log(`\n=== index explorer: ${pass}/${pass + fail} passed ===`);
await browser.close();
process.exit(fail ? 1 : 0);
