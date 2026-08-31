"""Session transcripts and follow-up resolution."""

from __future__ import annotations

import pytest

from rights_agent.conversation import (
    ConversationStore,
    Turn,
    contextualise,
    looks_like_follow_up,
    reconstruct_turns,
    summarise_sessions,
)

TOPIC = "What does the document say about bereavement leave?"


# --------------------------------------------------------------------------- #
# The store
# --------------------------------------------------------------------------- #
def test_turns_are_kept_in_order_per_session() -> None:
    store = ConversationStore()
    store.add("a", Turn("user", "one"))
    store.add("a", Turn("agent", "two"))
    store.add("b", Turn("user", "other"))
    assert [t.content for t in store.history("a")] == ["one", "two"]
    assert [t.content for t in store.history("b")] == ["other"]


def test_only_user_turns_count_as_questions() -> None:
    store = ConversationStore()
    store.add("a", Turn("user", "q"))
    store.add("a", Turn("agent", "a"))
    assert store.questions("a") == ["q"]


def test_transcripts_are_bounded() -> None:
    """An unbounded in-memory transcript store is a slow leak with a
    data-protection surface attached."""
    store = ConversationStore(max_turns=4)
    for index in range(10):
        store.add("a", Turn("user", str(index)))
    assert [t.content for t in store.history("a")] == ["6", "7", "8", "9"]


def test_sessions_are_bounded_and_evict_oldest_first() -> None:
    store = ConversationStore(max_sessions=3)
    for index in range(5):
        store.add(f"s{index}", Turn("user", "q"))
    assert store.sessions() == ["s2", "s3", "s4"]
    assert store.history("s0") == []


def test_clear_drops_one_session_or_all() -> None:
    store = ConversationStore()
    store.add("a", Turn("user", "q"))
    store.add("b", Turn("user", "q"))
    store.clear("a")
    assert store.history("a") == [] and store.history("b")
    store.clear()
    assert store.sessions() == []


def test_turns_serialise_to_plain_dicts() -> None:
    payload = Turn("agent", "text", citations=("s.19",), refused=False).to_dict()
    assert payload["citations"] == ["s.19"]
    assert isinstance(payload["ts"], float)


# --------------------------------------------------------------------------- #
# Detecting a follow-up
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "question",
    [
        "How long is it?",
        "What about pregnancy loss?",
        "and the notice period?",
        "Does that apply to agency workers?",
        "How much?",
        "Why is that?",
    ],
)
def test_continuations_are_recognised(question: str) -> None:
    assert looks_like_follow_up(question)


@pytest.mark.parametrize(
    "question",
    [
        TOPIC,
        "What is the qualifying period for unfair dismissal?",
        "Which provision covers the allocation of tips?",
        "How are penalty notices calculated?",
        # Seven words with a topic of its own: resolving this would append an
        # unrelated subject to a question that never needed one.
        "What does the document say about tips?",
    ],
)
def test_self_contained_questions_are_not_treated_as_follow_ups(question: str) -> None:
    assert not looks_like_follow_up(question)


def test_an_empty_question_is_not_a_follow_up() -> None:
    assert not looks_like_follow_up("   ")


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
def test_a_follow_up_borrows_the_previous_questions_terms() -> None:
    resolved = contextualise("How long is it?", [TOPIC])
    assert resolved.used_history
    assert "bereavement" in resolved.query and "leave" in resolved.query
    assert resolved.query.startswith("How long is it?")
    assert "bereavement" in resolved.borrowed


def test_a_self_contained_question_is_never_rewritten() -> None:
    resolved = contextualise("Which provision covers the allocation of tips?", [TOPIC])
    assert not resolved.used_history
    assert resolved.query == "Which provision covers the allocation of tips?"
    assert resolved.reason


def test_the_first_question_of_a_session_has_nothing_to_borrow() -> None:
    resolved = contextualise("How long is it?", [])
    assert not resolved.used_history
    assert resolved.query == "How long is it?"


def test_only_the_previous_turn_is_used() -> None:
    """Reaching further back drags in stale topics."""
    resolved = contextualise(
        "How long is it?",
        ["What about penalty notices?", "Which provision covers the allocation of tips?"],
    )
    assert "allocation" in resolved.query
    assert "penalty" not in resolved.query


def test_scaffolding_is_never_borrowed_but_short_topics_are() -> None:
    resolved = contextualise("How long?", ["What does the document say about tips?"])
    assert "document" not in resolved.borrowed, "scaffolding is not a topic"
    assert "tips" in resolved.borrowed, "a four-letter noun is still the subject"


def test_borrowing_is_capped() -> None:
    long_question = "bereavement leave pregnancy statutory paternity parental adoption shared"
    resolved = contextualise("How long is it?", [long_question], max_terms=3)
    assert len(resolved.borrowed) <= 3


def test_terms_already_present_are_not_repeated() -> None:
    resolved = contextualise("Is bereavement leave paid?", [TOPIC])
    assert "bereavement" not in resolved.borrowed


def test_resolution_is_deterministic() -> None:
    """No model call: a rewrite that changes between runs makes every downstream
    measurement unreproducible."""
    first = contextualise("How long is it?", [TOPIC])
    second = contextualise("How long is it?", [TOPIC])
    assert first == second


def test_resolution_serialises_for_the_api() -> None:
    payload = contextualise("How long is it?", [TOPIC]).to_dict()
    assert payload["used_history"] is True
    assert isinstance(payload["borrowed"], list)


# --------------------------------------------------------------------------- #
# History across both tiers
# --------------------------------------------------------------------------- #
def audit_row(**overrides: object) -> dict[str, object]:
    base = {
        "sequence": 0,
        "session_id": "c-1",
        "actor": "dashboard",
        "ts": "2026-08-28T10:00:00Z",
        "question": TOPIC,
        "citations": ["s.19"],
        "refused": False,
        "request_id": "r0",
    }
    base.update(overrides)
    return base


def test_reconstructed_turns_alternate_user_and_agent() -> None:
    turns = reconstruct_turns([audit_row(), audit_row(sequence=1, question="How long is it?")])
    assert [t.role for t in turns] == ["user", "agent", "user", "agent"]
    assert all(t.reconstructed for t in turns)


def test_reconstruction_follows_sequence_not_input_order() -> None:
    turns = reconstruct_turns(
        [audit_row(sequence=2, question="third"), audit_row(sequence=0, question="first")]
    )
    assert [t.content for t in turns if t.role == "user"] == ["first", "third"]


def test_a_reconstructed_answer_says_the_prose_is_not_retained() -> None:
    """Rendering an empty bubble would let a reader assume the model said
    nothing; the audit record simply never stored the words."""
    agent_turn = reconstruct_turns([audit_row()])[1]
    assert "not retained" in agent_turn.content
    assert agent_turn.citations == ("s.19",)
    assert not agent_turn.refused


def test_a_reconstructed_refusal_is_marked_as_one() -> None:
    agent_turn = reconstruct_turns([audit_row(refused=True, citations=[])])[1]
    assert agent_turn.refused
    assert "Refused" in agent_turn.content


def test_reconstruction_of_nothing_is_empty() -> None:
    assert reconstruct_turns([]) == []


def test_sessions_are_listed_newest_first_regardless_of_tier() -> None:
    """The tier is a badge, not a rank: sorting live above audit would bury the
    conversation you had five minutes before a restart."""
    store = ConversationStore()
    store.add("live-old", Turn("user", TOPIC, ts=1_600_000_000.0))
    summaries = summarise_sessions(
        store.live(), [audit_row(session_id="audit-new", ts="2026-08-28T10:00:00Z")]
    )
    assert [s.session_id for s in summaries] == ["audit-new", "live-old"]
    assert {s.source for s in summaries} == {"live", "audit"}


def test_a_live_transcript_wins_over_its_own_audit_records() -> None:
    """One session in both tiers is listed once, as live: the transcript has the
    answers and the audit records do not."""
    store = ConversationStore()
    store.add("c-1", Turn("user", TOPIC, ts=1_800_000_000.0))
    store.add("c-1", Turn("agent", "answer", ts=1_800_000_001.0))
    summaries = summarise_sessions(store.live(), [audit_row(session_id="c-1")])
    assert len(summaries) == 1
    assert summaries[0].source == "live"
    assert summaries[0].turns == 2
    assert summaries[0].requests == 1, "the request count still comes from the audit tier"


def test_the_title_is_the_opening_question() -> None:
    store = ConversationStore()
    store.add("s", Turn("user", TOPIC))
    store.add("s", Turn("user", "a later question"))
    assert summarise_sessions(store.live(), [])[0].title == TOPIC


def test_a_long_title_is_truncated() -> None:
    store = ConversationStore()
    store.add("s", Turn("user", "x " * 200))
    title = summarise_sessions(store.live(), [])[0].title
    assert len(title) <= 64 and title.endswith("…")


def test_cost_is_grouped_by_session_like_the_cost_panel() -> None:
    metrics = [
        {"session_id": "c-1", "cost_total_usd": 0.001},
        {"session_id": "c-1", "cost_total_usd": 0.002},
        {"session_id": "c-2", "cost_total_usd": 0.005},
    ]
    summaries = {s.session_id: s for s in summarise_sessions({}, [audit_row(), audit_row(sequence=1, session_id="c-2")], metrics)}
    assert summaries["c-1"].cost_usd == pytest.approx(0.003)
    assert summaries["c-2"].cost_usd == pytest.approx(0.005)


def test_the_actor_filter_keeps_synthetic_traffic_out_of_the_history() -> None:
    """The same audit log holds the traffic controls and the golden set. A
    history list swamped by ``eval-g029`` is a list nobody reads."""
    rows = [
        audit_row(actor="dashboard", session_id="mine"),
        audit_row(actor="demo", session_id="baseline-1", sequence=1),
        audit_row(actor="eval", session_id="eval-g001", sequence=2),
    ]
    listed = {s.session_id for s in summarise_sessions({}, rows, actors=["dashboard"])}
    assert listed == {"mine"}
    assert len(summarise_sessions({}, rows)) == 3, "unfiltered still sees everything"


def test_refusals_are_counted_per_session() -> None:
    rows = [audit_row(refused=True), audit_row(sequence=1, refused=False)]
    assert summarise_sessions({}, rows)[0].refused == 1


def test_the_list_is_capped() -> None:
    rows = [audit_row(session_id=f"s{index}", sequence=index) for index in range(40)]
    assert len(summarise_sessions({}, rows, limit=5)) == 5


def test_a_session_with_no_questions_is_skipped() -> None:
    assert summarise_sessions({}, [audit_row(question="")]) != []
    assert summarise_sessions({"empty": []}, []) == []


def test_summaries_serialise_for_the_api() -> None:
    store = ConversationStore()
    store.add("s", Turn("user", TOPIC))
    payload = summarise_sessions(store.live(), [])[0].to_dict()
    assert set(payload) >= {"session_id", "title", "source", "turns", "requests", "cost_usd"}


def test_a_chain_of_follow_ups_keeps_the_original_topic() -> None:
    """The question immediately before is itself topicless; borrowing from it
    would borrow nothing."""
    resolved = contextualise("And for agency workers?", [TOPIC, "How long is it?"])
    assert resolved.used_history
    assert "bereavement" in resolved.borrowed


def test_the_lookback_is_bounded_so_stale_topics_cannot_resurface() -> None:
    history = ["What does the document say about penalty notices?"] + [
        "How long is it?"
    ] * 8
    resolved = contextualise("And for agency workers?", history)
    assert "penalty" not in resolved.borrowed
    assert not resolved.used_history


def test_a_topic_question_still_wins_over_an_older_one() -> None:
    resolved = contextualise(
        "How long is it?",
        ["What about penalty notices?", "Which provision covers the allocation of tips?"],
    )
    assert "allocation" in resolved.borrowed
    assert "penalty" not in resolved.borrowed


@pytest.mark.parametrize(
    "question",
    ["And for agency workers?", "Or does it apply to part-time staff?", "But what if they refuse?"],
)
def test_a_leading_conjunction_marks_a_continuation(question: str) -> None:
    """It carries its own distinctive terms and is still a follow-up: the topic
    it narrows lives in the previous turn."""
    assert looks_like_follow_up(question)
    resolved = contextualise(question, [TOPIC])
    assert resolved.used_history
    assert "bereavement" in resolved.borrowed
