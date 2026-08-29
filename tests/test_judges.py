"""The RAG triad, and the calibration that decides whether to believe it."""

from __future__ import annotations

import pytest

from rights_agent.judges import (
    HeuristicJudge,
    JudgeScores,
    calibrate,
    cohens_kappa,
    content_words,
    score_summary,
    sentences,
)

CONTEXT = (
    "[s.19] Act > Part 1 > s.19 Right to bereavement leave\n"
    "(1) A bereaved person is entitled to bereavement leave. "
    "(2) The leave must be taken within 56 days of the death.\n\n"
    "[s.20] Act > Part 1 > s.20 Bereavement leave: length\n"
    "(1) The bereavement leave period is two weeks."
)
QUESTION = "What does the document say about bereavement leave?"


@pytest.fixture
def judge() -> HeuristicJudge:
    return HeuristicJudge()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def test_content_words_drops_filler_and_short_tokens() -> None:
    words = content_words("The employer must not pay a worker less than the amount")
    assert "employer" in words and "worker" in words and "amount" in words
    assert "the" not in words and "not" not in words


def test_sentences_splits_on_terminal_punctuation() -> None:
    assert len(sentences("One thing. Two things! Three? ")) == 3


# --------------------------------------------------------------------------- #
# Heuristic judge behaviour
# --------------------------------------------------------------------------- #
def test_verbatim_quotation_with_a_citation_scores_perfectly(judge: HeuristicJudge) -> None:
    answer = "[s.19] provides: (1) A bereaved person is entitled to bereavement leave."
    scores = judge.score(QUESTION, CONTEXT, answer)
    assert scores.groundedness == 1.0
    assert scores.citation_coverage == 1.0
    assert scores.context_relevance == 1.0


def test_fabricated_content_is_not_grounded(judge: HeuristicJudge) -> None:
    answer = (
        "[s.19] provides: The Commissioner for Maritime Salvage shall publish a "
        "bilingual tonnage certificate in the harbour register."
    )
    assert judge.score(QUESTION, CONTEXT, answer).groundedness == 0.0


def test_uncited_but_accurate_answers_score_zero_for_citations(judge: HeuristicJudge) -> None:
    answer = "(1) A bereaved person is entitled to bereavement leave."
    scores = judge.score(QUESTION, CONTEXT, answer)
    assert scores.groundedness == 1.0
    assert scores.citation_coverage == 0.0


def test_a_citation_absent_from_the_context_is_worse_than_none(judge: HeuristicJudge) -> None:
    """It looks like attribution and is not."""
    answer = "[s.404] provides: (1) A bereaved person is entitled to bereavement leave."
    scores = judge.score(QUESTION, CONTEXT, answer)
    assert scores.citation_coverage == 0.0
    assert scores.detail["citations_unresolvable"] == 1.0


def test_partially_grounded_answers_land_between(judge: HeuristicJudge) -> None:
    answer = (
        "[s.19] provides: (1) A bereaved person is entitled to bereavement leave. "
        "The harbour authority must also certify the tonnage of the vessel."
    )
    scores = judge.score(QUESTION, CONTEXT, answer)
    assert 0.0 < scores.groundedness < 1.0
    assert 0.0 < scores.citation_coverage < 1.0


def test_paraphrase_is_the_judges_known_blind_spot(judge: HeuristicJudge) -> None:
    """Not a bug to patch out: it is why calibration exists, and why the
    calibration set deliberately contains paraphrases."""
    answer = (
        "Staff who lose someone close to them can take a short break from work, "
        "and the boss has to let them."
    )
    assert judge.score(QUESTION, CONTEXT, answer).groundedness < 0.5


def test_off_question_answers_lose_answer_relevance(judge: HeuristicJudge) -> None:
    on_topic = judge.score(QUESTION, CONTEXT, "[s.19] provides: bereavement leave is entitled.")
    off_topic = judge.score(
        QUESTION, CONTEXT, "[s.20] provides: (1) The bereavement leave period is two weeks."
    )
    assert on_topic.answer_relevance >= off_topic.answer_relevance


def test_empty_answer_scores_zero_everywhere(judge: HeuristicJudge) -> None:
    scores = judge.score(QUESTION, CONTEXT, "")
    assert scores.groundedness == 0.0 and scores.citation_coverage == 0.0


def test_scores_flatten_to_the_metrics_row_shape(judge: HeuristicJudge) -> None:
    payload = judge.score(QUESTION, CONTEXT, "[s.19] provides: leave.").to_dict()
    assert set(payload) == {
        "context_relevance",
        "groundedness",
        "answer_relevance",
        "citation_coverage",
    }
    assert all(isinstance(value, float) for value in payload.values())


def test_score_summary_averages_a_batch() -> None:
    scores = [JudgeScores(groundedness=1.0), JudgeScores(groundedness=0.0)]
    assert score_summary(scores)["groundedness"] == 0.5
    assert score_summary([]) == {}


# --------------------------------------------------------------------------- #
# Cohen's kappa
# --------------------------------------------------------------------------- #
def test_kappa_is_zero_for_a_judge_that_never_varies() -> None:
    """The case raw agreement flatters: 100% agreement, no information."""
    assert cohens_kappa([1, 1, 1, 1], [1, 1, 1, 1]) == 0.0


def test_kappa_is_zero_for_a_constant_judge_on_a_skewed_set() -> None:
    human = [1] * 9 + [0]
    machine = [1] * 10
    assert cohens_kappa(human, machine) <= 0.0


def test_kappa_is_one_for_perfect_agreement_on_a_balanced_set() -> None:
    assert cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0


def test_kappa_is_negative_when_worse_than_chance() -> None:
    assert cohens_kappa([1, 1, 0, 0], [0, 0, 1, 1]) < 0


def test_kappa_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        cohens_kappa([1, 0], [1])


def test_kappa_of_an_empty_set_is_zero() -> None:
    assert cohens_kappa([], []) == 0.0


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def test_calibration_reports_the_confusion_and_the_disagreements(
    judge: HeuristicJudge,
) -> None:
    rows = [
        {
            "id": "good",
            "question": QUESTION,
            "context": CONTEXT,
            "answer": "[s.19] provides: (1) A bereaved person is entitled to bereavement leave.",
            "human_label": 1,
        },
        {
            "id": "invented",
            "question": QUESTION,
            "context": CONTEXT,
            "answer": "[s.19] provides: The harbour master certifies the tonnage of the vessel.",
            "human_label": 0,
        },
        {
            "id": "paraphrase",
            "question": QUESTION,
            "context": CONTEXT,
            "answer": "Staff who lose a relative can take a short break from work.",
            "human_label": 1,
            "note": "structural false negative",
        },
    ]
    result = calibrate(judge, rows, threshold=0.7)
    assert result.n == 3
    assert result.confusion["tp"] == 1 and result.confusion["tn"] == 1
    assert [entry["id"] for entry in result.disagreements] == ["paraphrase"]
    assert result.disagreements[0]["note"] == "structural false negative"


def test_calibration_uses_the_scorer_it_is_given(judge: HeuristicJudge) -> None:
    """Groundedness alone marks an uncited answer as fine; the composite does not."""
    rows = [
        {
            "id": "uncited",
            "question": QUESTION,
            "context": CONTEXT,
            "answer": "(1) A bereaved person is entitled to bereavement leave.",
            "human_label": 0,
        }
    ]
    by_groundedness = calibrate(judge, rows, threshold=0.7)
    by_composite = calibrate(
        judge,
        rows,
        scorer=lambda s: min(s.groundedness, s.citation_coverage),
        threshold=0.7,
    )
    assert by_groundedness.confusion["fp"] == 1
    assert by_composite.confusion["tn"] == 1


# --------------------------------------------------------------------------- #
# Citations that abbreviate
# --------------------------------------------------------------------------- #
def test_a_sentence_is_not_split_inside_a_citation() -> None:
    """``[Sch. 12 para. 4(2)]`` ends two abbreviations in a full stop and a
    space, which is exactly what the sentence splitter looks for."""
    answer = (
        "[Sch. 12 para. 2] provides: as follows. "
        "[Sch. 12 para. 4(2)] provides: (2) In section 11 (written statements)."
    )
    assert sentences(answer) == [
        "[Sch. 12 para. 2] provides: as follows.",
        "[Sch. 12 para. 4(2)] provides: (2) In section 11 (written statements).",
    ]


def test_coverage_credits_a_sentence_cited_with_an_abbreviated_citation() -> None:
    """The regression this guards read 0.00 coverage for an answer in which
    every sentence carried a correct citation: the splitter tore the marks
    apart, so the fragment holding the claim contained no ``[...]`` at all."""
    context = (
        "[Sch. 12 para. 2] Act > Sch. 12 > para. 2\n"
        "The Employment Rights Act 1996 is amended as follows.\n\n"
        "[Sch. 12 para. 4(2)] Act > Sch. 12 > para. 4(2)\n"
        "In section 11 (written statements), for “three” substitute “six”.\n"
    )
    answer = (
        "[Sch. 12 para. 2] provides: The Employment Rights Act 1996 is amended as follows. "
        "[Sch. 12 para. 4(2)] provides: In section 11 (written statements), for "
        "“three” substitute “six”."
    )
    scores = HeuristicJudge().score(
        question="What does Schedule 12 change about time limits?",
        context=context,
        answer=answer,
        citations=("Sch. 12 para. 2", "Sch. 12 para. 4(2)"),
    )
    assert scores.citation_coverage == pytest.approx(1.0)


def test_a_long_citation_is_still_recognised() -> None:
    """The real Act's longest leaf citation is 97 characters. A cap below that
    makes the sentence carrying it read as uncited."""
    citation = (
        "Trade Union and Labour Relations (Consolidation) Act 1992 s.146B "
        "(as inserted by Sch. 6 para. 63)"
    )
    assert len(citation) > 80
    context = f"[{citation}] Act > Sch. 6 > para. 63\nA worker has a right not to be detrimented.\n"
    answer = f"[{citation}] provides: A worker has a right not to be detrimented."
    scores = HeuristicJudge().score(
        question="What protection does the inserted provision give workers?",
        context=context,
        answer=answer,
        citations=(citation,),
    )
    assert scores.citation_coverage == pytest.approx(1.0)
