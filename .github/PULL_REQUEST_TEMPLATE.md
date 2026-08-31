# What this changes

<!-- One or two sentences. What was wrong, or missing, and what does this do about it. -->

## Why

<!-- The failure it prevents, or the question it lets someone answer. If it fixes
     a bug, name the symptom someone would have seen. -->

## How it was verified

<!-- Delete what does not apply. -->

- [ ] `make lint` is clean
- [ ] `uv run pytest tests/ -q` passes
- [ ] `uv run pytest evals/test_deterministic.py -q` passes
- [ ] `uv run pytest evals/test_quality.py -q` passes
- [ ] Ran it against the real thing (`uv run rights-ask "..."`, or the dashboard)

## Gate and datasets

<!-- Answer these only if you touched evals/, the corpus, the embedder, or the
     prompt. They are the changes most likely to look fine and quietly break the
     meaning of the gate. -->

- [ ] No threshold was lowered. (Thresholds ratchet upward only; a red gate is
      information, not an obstacle.)
- [ ] No row was newly marked `known_failure` to get the build green.
- [ ] If the corpus, embedder or prompt changed, the datasets were regenerated
      (`uv run python -m rights_agent goldens --write-baseline`) in their own
      commit, and the numbers that moved are described below.

<!-- What moved, and why: -->

## Anything a reviewer should look at first

<!-- The part you are least sure about. -->
