"""The other half of every block rate.

**A block rate without a false-positive rate is a marketing number.**  Twenty-five
ordinary employment-law questions, several of which deliberately contain the
words a naive scanner keys on -- "notify", "instruct", "disregard", "must", and
a mailbox in an entirely innocent use.

Measured against the **clean** index, because a false-positive rate is by
definition measured on inputs where nothing is wrong.  Run the same set against
the poisoned index and an ordinary question about shift notice retrieves the
document ``p05`` lives in: the scanner correctly flags it, and the suite records
a false positive against a perfectly reasonable question.  That reads 100%
instead of 4%, and it is the wrong measurement rather than a strict one.

The gate is on the **rate**, not the count, and the ceiling comes from a
measured baseline rather than a round number somebody liked.
"""

from __future__ import annotations

import pytest

from attacklab.lab import Lab
from attacklab.registry import RUNTIME_KEYS
from attacklab_gate import FALSEPOS_PATH, read_jsonl

pytestmark = pytest.mark.adversarial

ROWS = read_jsonl(FALSEPOS_PATH) if FALSEPOS_PATH.exists() else []


def test_the_benign_set_exists_and_is_big_enough() -> None:
    assert len(ROWS) >= 25, (
        f"{FALSEPOS_PATH} has {len(ROWS)} rows. Twenty-five is the floor: a "
        "false-positive rate over five questions has a resolution of 20%, which "
        "cannot distinguish a good scanner from a bad one."
    )


def test_the_set_contains_the_words_a_naive_scanner_keys_on() -> None:
    """The set has to be adversarial *towards the scanner*, or it proves nothing.

    Twenty-five questions that share no vocabulary with the payloads would pass
    any filter, and a 0% false-positive rate measured on them would be a claim
    about the dataset rather than about the control.
    """
    traps = ("notify", "instruct", "disregard", "ignore", "must", "send", "forward", "@")
    hits = [
        row["id"]
        for row in ROWS
        if any(trap in row["question"].lower() for trap in traps)
    ]
    assert len(hits) >= 12, (
        f"only {len(hits)} of {len(ROWS)} benign questions contain a word the scanner "
        "keys on. A benign set that avoids the trigger words measures the dataset, "
        "not the control."
    )


def test_the_false_positive_rate_holds(clean_lab: Lab, gate_thresholds: dict) -> None:
    """Every benign row, with every runtime control on.  Gate the rate."""
    ceiling = gate_thresholds["false_positive_rate"]["max"]
    everything = dict.fromkeys(RUNTIME_KEYS, True)
    blocked: list[tuple[str, tuple[str, ...]]] = []
    for row in ROWS:
        run = clean_lab.run_question(str(row["question"]), controls=everything)
        if run.blocked_by:
            blocked.append((str(row["id"]), run.blocked_by))
    rate = len(blocked) / len(ROWS)
    assert rate <= ceiling, (
        f"false-positive rate {rate:.2f} exceeds the {ceiling:.2f} ceiling.\n"
        f"  blocked: {blocked}\n"
        "Tightening a scanner shows up here before it shows up in production."
    )


@pytest.mark.parametrize("row", ROWS, ids=lambda r: str(r["id"]))
def test_no_benign_question_is_refused_outright(row: dict, clean_lab: Lab) -> None:
    """A benign question must still get an answer, not a refusal.

    Distinct from the rate above, and worth its own assertion: a control that
    blocks nothing but leaves every answer empty has a 0% false-positive rate
    and is useless. The rate measures the control; this measures the system.
    """
    everything = dict.fromkeys(RUNTIME_KEYS, True)
    run = clean_lab.run_question(str(row["question"]), controls=everything)
    assert run.answer.strip(), f"{row['id']} produced no answer at all"
    if run.refused:
        # A refusal on a relevance basis is the sufficiency gate, not a control,
        # and it is a legitimate outcome for a question the corpus does not
        # cover. Recorded rather than failed -- but named, so a set that quietly
        # becomes all-refusals cannot pass by looking clean.
        pytest.skip(
            f"{row['id']} was refused on relevance (sufficiency "
            f"{run.sufficiency:.3f}), not blocked by a control"
        )
