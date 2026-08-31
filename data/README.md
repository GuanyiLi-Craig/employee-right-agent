# The corpus

Two documents live here, and they do different jobs.

| File | Role |
|---|---|
| `ukpga_20250036_en.pdf` | The real **Employment Rights Act 2025 (c. 36)**, 335 pages. What the demo serves, what the committed eval datasets were generated against, and what CI gates. |
| `corpus.layout.txt` | A **generated** Act, laid out exactly as `pdftotext -layout` renders a UK Public General Act. The offline fallback, the fixture the parser unit tests are written against, and the default when `RIGHTS_CORPUS` is unset. |

Select one with `RIGHTS_CORPUS`:

```bash
RIGHTS_CORPUS=/app/data/ukpga_20250036_en.pdf     # the real Act (set in .env)
# unset                                            # the generated Act
```

`pdftotext` (poppler) must be installed to read a PDF; it is already in the
Docker image, which is why ingest runs there.

The eval datasets are a function of the corpus — expected citations name
provisions, and a provision exists in exactly one document. `evals/datasets/<embedder>/baseline.json`
records which corpus it was generated for, and the gate refuses to run against
another one rather than reporting twenty citation failures that look like a
retrieval regression. After switching corpus:

```bash
docker compose run --rm ingest
docker compose run --rm -v "$PWD/evals:/app/evals" \
  --entrypoint "/usr/bin/tini -- /usr/local/bin/entrypoint.sh goldens" evals \
  --evals-dir /app/evals --write-baseline
```

## The generated Act

```bash
uv run rights-corpus --out data/corpus.layout.txt      # or: make corpus
```

Output is a pure function of `src/rights_agent/tools/corpus.py`, so two
generations produce identical bytes and therefore an identical `index_version`.
CI asserts that.

It is generated rather than downloaded so the repository has a corpus with no
licensing question attached and no network dependency at demo time. It
deliberately contains the four things that break parsers:

| # | Trap | Where |
|---|------|-------|
| 1 | A table of contents that looks exactly like the body | front matter, before the enacting formula |
| 2 | Running headers on every page, in mixed case (`Part 1 — Employment rights`) so a case-insensitive filter would also eat the real `PART 1` markers | every body page |
| 3 | Quoted material that mimics a cross-heading but is followed by indented text | inside the inserting subsections |
| 4 | Provisions inserted into *another* Act, numbered with a letter suffix (`27BA`) at non-zero indent | ss. 1, 13, 23, 48, 74, 102 |

Structure: 6 Parts, 33 cross-headings, 167 sections, 12 inserted provisions,
2 Schedules with their own internal Parts, ~1,100 subsections, ~109 pages.

## What the real Act did to a parser written against the generated one

All four traps were present, and 47 of the 87 inserted provisions ended up citing
"the host Act" — a citation naming no document — before the last three rows below
were fixed. None of the traps looked the way the fixture had taught
the parser to expect, and every one of them failed *silently* — a smaller tree,
not an error. This is the argument for `validate_tree` and for stamping
`PARSER_VERSION` into `index_version`.

| What the parser assumed | What the Act actually does | Consequence until fixed |
|---|---|---|
| The enacting formula reads `BE IT ENACTED` | A drop cap sets it as `B     E IT ENACTED` | Front-matter detection failed, so the table of contents was parsed as body |
| `SCHEDULE 9` stands alone on its line | `SCHEDULE 9                Section 135(5)` — the authorising section shares the line | The marker was not recognised; Schedule 9 was then pruned as an empty heading |
| A schedule paragraph opens its own line | `1    (1) The Secretary of State may…` — paragraph and first subsection share a line | 39% of leaves ended up with no provision above them, and an uncitable chunk cannot be cited |
| Body provisions start at column 0 | Schedule paragraphs are set one space in | All 180 schedule paragraphs vanished into the preceding block's text |
| An inserted provision heading is separated by two or more spaces | `27BA Right to guaranteed hours` uses one | Only 20 of the 87 inserted provisions were found at all |
| A statute's short title is words and digits | `Trade Union and Labour Relations (Consolidation) Act 1992` has a bracketed qualifier | 18 inserted provisions were cited as "the host Act" — naming no document a reader could look up |
| The host Act is named where the insertion happens | A Schedule names it once, in the body section that introduces it (`Schedule 5 amends the Seafarers' Wages Act 2023`), and never again | 11 more had no attributable host until that cross-reference was followed |

An eighth followed, found only because a degenerate chunk reached the committed
calibration set: statutory cross-references wrap across their own number
(``...that comply with subsection`` / ``(2) of that section.``), and reading the
second line as an opener produced a subsection whose entire text was ``of that
section.`` — a chunk that says nothing, sits in the searched index, and turned up
as cited evidence in an answer. 63 leaves were fragments of that kind.

Final tree: 24 Parts, 12 Schedules, 459 sections and schedule paragraphs, 87
inserted provisions, 1,971 subsections, 2,078 citable leaves, **0** uncitable
leaves and **0** inserted provisions without a named host Act.

Two of the four scoring metrics also moved without any answer changing, because
the new citation shapes broke the *measurement*: `[Sch. 12 para. 4(2)]` ends two
abbreviations in a full stop and a space, which is what the judge's sentence
splitter looked for, and the citation recogniser capped marks at 80 characters
against a corpus whose longest citation is 97. Citation coverage read 0.78 while
every sentence was correctly cited. See `evals/datasets/<embedder>/baseline.json`
(`observed_when_set`) for the measurements behind each threshold.
