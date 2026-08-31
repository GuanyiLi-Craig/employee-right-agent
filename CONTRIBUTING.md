# Contributing

Thanks for taking a look. This is a demonstration codebase with an unusual
emphasis: the retrieval pipeline is the vehicle, and the observability,
evaluation and audit layers are the subject. That shapes what a good change
looks like here, so it is worth reading the two sections on the eval gate before
sending anything that touches `evals/`.

## Setting up

```bash
uv sync --extra trace --extra models --group dev   # models: only for a hosted provider
uv run rights-ingest --no-onnx                     # build the index (a separate job)
uv run pytest tests/ -q                            # unit tests, no index required
```

Every command goes through `uv run`. Never activate a venv by hand — that is
where "works on my machine" comes from. `make help` lists the shortcuts.

Reading a PDF corpus needs `pdftotext`:

```bash
brew install poppler            # macOS
apt-get install -y poppler-utils # Debian/Ubuntu
```

You do not need it to work on most of the codebase. `data/corpus.layout.txt` is
a generated Act committed to the repository, it is the default corpus, and it is
what the parser unit tests are written against. You need poppler only to run the
eval gate, which is pointed at the real Act.

## Before you open a pull request

```bash
make lint     # ruff, and it must be clean
make test     # unit tests and both eval suites
```

CI runs the same things plus a Docker build, so a green local run is a good
predictor. The unit tests need no index and no network; the eval suites need an
index.

## The rules that are not negotiable

**Never lower a threshold to make a build green.** Thresholds live in
`evals/datasets/<embedder>/baseline.json` and are set *below* values observed on
a green build. Ratchet them upward as the system improves. A red gate is
information; editing the number until it goes away destroys the only signal the
suite carries.

**Never mark a failing row `known_failure` to get past the gate.** The same file
records which golden rows are known to fail, the gate asserts that list does not
grow, and it also fails when a known failure starts passing so the marker gets
removed. Both directions are deliberate.

**Nothing in `test_deterministic.py` may ask a model for an opinion.** It
contains structural assertions only, which is exactly why it is allowed to fail
a build. Aggregate and distributional claims go in `test_quality.py`, because
model output is a distribution and asserting that every answer clears a bar
produces a flaky suite that gets deleted.

**Changing the corpus or the embedder invalidates the datasets.** An expected
citation names a provision, and a provision exists in one document. The gate
refuses to run when the dataset and the index disagree, which is one legible
failure instead of twenty that look like a retrieval regression. Regenerating is
a deliberate act:

```bash
uv run python -m rights_agent goldens --write-baseline
```

Commit the regenerated dataset in its own commit, and say in the message what
moved and why.

## Style

`ruff` is the arbiter; the configuration is in `pyproject.toml`. Two things it
cannot check but this codebase cares about:

- **Comments explain why, not what.** The existing comments name the specific
  failure a line prevents, often with the symptom it produces. Match that.
  A `# noqa` carries a reason after the code, always.
- **Error messages name the command that fixes them.** When something cannot
  proceed, say what to run. `IndexNotBuiltError` is the model to copy.

There is no enforced formatter. The prose comments are hand-wrapped and a
blanket reformat would flatten them, so match the surrounding file.

## Tests

Every pitfall in the README's "Pitfalls this codebase defends against" table has
a test naming it, and test names are sentences describing the behaviour
(`test_the_golden_set_was_generated_for_this_index`). New defences get the same
treatment: if you fix a bug, the test name should say what would otherwise go
wrong.

## Security

Do not open a public issue for a vulnerability. See [SECURITY.md](SECURITY.md).
