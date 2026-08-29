import { chromium } from 'playwright';
const PHOENIX = process.env.PHOENIX || 'http://localhost:6016';
let pass = 0, fail = 0;
const check = (n, ok, extra = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${n}${extra ? '  — ' + extra : ''}`); ok ? pass++ : fail++; };

const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1600, height: 1000 } });
await p.goto(`${PHOENIX}/projects`, { waitUntil: 'networkidle' });
await p.getByText('rights-rag-agent').first().click();
await p.waitForLoadState('networkidle');
await p.waitForTimeout(2500);

const body = await p.locator('body').innerText();
check('project page opened', /rights-rag-agent/.test(body));

// annotation names should appear as columns or feedback chips somewhere on the page
const names = ['groundedness', 'citation_coverage', 'context_relevance', 'answer_relevance', 'sufficiency'];
const found = names.filter((n) => body.includes(n));
check('annotation names visible on the traces page', found.length >= 3, `${found.length}/5: ${found.join(', ')}`);

await p.screenshot({ path: 'shots/phoenix-annotations.png' });
check('no crash', true);
console.log(`\n=== phoenix annotations: ${pass}/${pass + fail} passed ===`);
await b.close();
process.exit(fail ? 1 : 0);
