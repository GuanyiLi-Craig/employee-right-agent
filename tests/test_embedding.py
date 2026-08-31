"""The embedders and the pinning rule."""

from __future__ import annotations

import numpy as np
import pytest

from rights_agent.embedding import (
    HASHING_NAME,
    ONNX_NAME,
    OPENAI_BATCH,
    OPENAI_LARGE_NAME,
    OPENAI_MAX_ATTEMPTS,
    OPENAI_SMALL_NAME,
    EmbedderError,
    HashingEmbedder,
    OpenAIEmbedder,
    assert_embedder_matches,
    build_embedder,
    cosine_similarity,
    embedder_name,
    get_embedder,
    tokenize,
)


def _vector(embedder: HashingEmbedder, text: str) -> list[float]:
    return [float(value) for value in embedder.embed_documents([text])[0]]


def test_implements_the_full_chroma_protocol() -> None:
    """A bare ``__call__`` is no longer sufficient in Chroma 1.x."""
    embedder = HashingEmbedder()
    assert HashingEmbedder.name() == HASHING_NAME
    assert embedder.default_space() == "cosine"
    assert embedder.get_config() == {"dim": 512}
    rebuilt = HashingEmbedder.build_from_config(embedder.get_config())
    assert rebuilt.get_config() == embedder.get_config()
    assert len(embedder.embed_documents(["x"])[0]) == 512
    assert len(embedder.embed_query(["x"])[0]) == 512
    assert not embedder.is_legacy()


def test_is_deterministic_across_instances_and_processes() -> None:
    """``hash()`` is salted per process; a salted index is an unreadable index."""
    left = _vector(HashingEmbedder(), "guaranteed hours offer")
    right = _vector(HashingEmbedder(), "guaranteed hours offer")
    assert left == right


def test_vectors_are_l2_normalised() -> None:
    vector = _vector(HashingEmbedder(), "An employer must make a guaranteed hours offer")
    assert np.isclose(np.linalg.norm(vector), 1.0)


def test_bigrams_distinguish_a_phrase_from_its_words() -> None:
    """Without bigrams "guaranteed hours" is the union of two common words."""
    embedder = HashingEmbedder()
    phrase = _vector(embedder, "guaranteed hours")
    shuffled = _vector(embedder, "hours guaranteed")
    assert cosine_similarity(phrase, shuffled) < 1.0


def test_related_text_scores_above_unrelated_text() -> None:
    embedder = HashingEmbedder()
    document = _vector(embedder, "An employer must make a guaranteed hours offer to the worker")
    related = _vector(embedder, "guaranteed hours offer")
    unrelated = _vector(embedder, "cryptocurrency mining rigs and hardware wallets")
    assert cosine_similarity(document, related) > 0.4
    assert cosine_similarity(document, unrelated) < 0.1


def test_empty_input_does_not_produce_a_zero_vector() -> None:
    """Cosine distance is undefined for a zero vector."""
    for text in ("", "   ", "the and of"):
        vector = _vector(HashingEmbedder(), text)
        assert np.isclose(np.linalg.norm(vector), 1.0), text


def test_tokenize_keeps_modals_and_numbers() -> None:
    tokens = tokenize("An employer must not pay less than £500 in 1996")
    assert "must" in tokens and "not" in tokens
    assert "£500" in tokens and "1996" in tokens
    assert "an" not in tokens


def test_dimension_floor_is_enforced() -> None:
    with pytest.raises(ValueError, match="at least 32"):
        HashingEmbedder(dim=8)


def test_changing_dimension_is_refused_as_a_config_update() -> None:
    embedder = HashingEmbedder()
    with pytest.raises(ValueError, match="rebuild it instead"):
        embedder.validate_config_update({"dim": 512}, {"dim": 256})


# --------------------------------------------------------------------------- #
# Selection and pinning
# --------------------------------------------------------------------------- #
def test_forcing_the_hashing_embedder_never_touches_the_network() -> None:
    embedder, name = get_embedder(prefer="hashing")
    assert name == HASHING_NAME
    assert isinstance(embedder, HashingEmbedder)


def test_require_resolves_exactly_and_ignores_preference() -> None:
    embedder, name = get_embedder(require=HASHING_NAME, prefer="onnx")
    assert name == HASHING_NAME
    assert isinstance(embedder, HashingEmbedder)


def test_unknown_embedder_names_are_rejected() -> None:
    with pytest.raises(EmbedderError, match="unknown embedder"):
        build_embedder("word2vec-300")
    with pytest.raises(EmbedderError, match="unknown embedder preference"):
        get_embedder(prefer="whatever")


def test_mismatch_between_index_and_process_is_fatal() -> None:
    """Pitfall 1: a mismatch returns confident nonsense, not an error."""
    with pytest.raises(EmbedderError) as excinfo:
        assert_embedder_matches(ONNX_NAME, HASHING_NAME)
    message = str(excinfo.value)
    assert "confident nonsense" in message
    assert "rebuild the index" in message


def test_matching_names_pass_and_no_recorded_name_is_permitted() -> None:
    assert_embedder_matches(HASHING_NAME, HASHING_NAME)
    assert_embedder_matches(None, HASHING_NAME)
    assert_embedder_matches("", HASHING_NAME)


# --------------------------------------------------------------------------- #
# The OpenAI embedder
# --------------------------------------------------------------------------- #
class _FakeEmbeddings:
    """Stands in for ``client.embeddings``, recording every request."""

    def __init__(self, dim: int = 8, fail_times: int = 0, reorder: bool = False) -> None:
        self.dim = dim
        self.fail_times = fail_times
        self.reorder = reorder
        self.batches: list[list[str]] = []
        self.calls = 0

    def create(self, *, model: str, input: list[str]):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("429 rate limited")
        self.batches.append(list(input))
        items = [
            type("Item", (), {"index": i, "embedding": [float(i)] * self.dim})()
            for i in range(len(input))
        ]
        if self.reorder:
            items.reverse()
        usage = type("Usage", (), {"total_tokens": 7 * len(input)})()
        return type("Response", (), {"data": items, "usage": usage})()


def _openai(monkeypatch, fake: _FakeEmbeddings, name: str = OPENAI_SMALL_NAME) -> OpenAIEmbedder:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)
    embedder._name = name
    embedder.model = "text-embedding-3-small"
    embedder.api_key_env = "OPENAI_API_KEY"
    embedder._client = type("Client", (), {"embeddings": fake})()
    embedder.tokens_used = 0
    return embedder


def test_a_missing_key_names_the_two_embedders_that_need_none(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(EmbedderError) as excinfo:
        OpenAIEmbedder()
    message = str(excinfo.value)
    assert "OPENAI_API_KEY" in message
    assert "onnx" in message and "hashing" in message


def test_inputs_are_split_into_batches(monkeypatch) -> None:
    fake = _FakeEmbeddings()
    embedder = _openai(monkeypatch, fake)

    vectors = embedder([f"chunk {i}" for i in range(OPENAI_BATCH * 2 + 5)])

    assert len(vectors) == OPENAI_BATCH * 2 + 5
    assert [len(batch) for batch in fake.batches] == [OPENAI_BATCH, OPENAI_BATCH, 5]
    assert embedder.tokens_used == 7 * (OPENAI_BATCH * 2 + 5)


def test_vectors_are_reordered_to_match_the_inputs(monkeypatch) -> None:
    """The API returns an ``index`` per item and does not promise input order.

    Trusting arrival order would attach every vector to the wrong chunk, and
    nothing downstream can detect that: the index still answers, confidently,
    with the wrong provisions.
    """
    fake = _FakeEmbeddings(reorder=True)
    embedder = _openai(monkeypatch, fake)

    vectors = embedder(["a", "b", "c"])

    assert [vector[0] for vector in vectors] == [0.0, 1.0, 2.0]


def test_a_transient_failure_is_retried_then_succeeds(monkeypatch) -> None:
    fake = _FakeEmbeddings(fail_times=2)
    embedder = _openai(monkeypatch, fake)

    vectors = embedder(["a", "b"])

    assert len(vectors) == 2
    assert fake.calls == 3


def test_retries_are_bounded_and_the_error_says_so(monkeypatch) -> None:
    """An ingest that hangs for minutes is harder to diagnose than one that
    stops and says which batch failed."""
    fake = _FakeEmbeddings(fail_times=99)
    embedder = _openai(monkeypatch, fake)

    with pytest.raises(EmbedderError) as excinfo:
        embedder(["a"])

    assert fake.calls == OPENAI_MAX_ATTEMPTS
    assert "429" in str(excinfo.value)


def test_an_empty_chunk_does_not_stop_an_ingest(monkeypatch) -> None:
    """The API rejects an empty string, and layout extraction can leave one."""
    fake = _FakeEmbeddings()
    embedder = _openai(monkeypatch, fake)

    embedder(["", "   ", "real text"])

    assert all(text.strip() != "" or text == " " for text in fake.batches[0])
    assert len(fake.batches[0]) == 3


def test_a_short_count_is_refused_rather_than_guessed(monkeypatch) -> None:
    class _Short(_FakeEmbeddings):
        def create(self, *, model: str, input: list[str]):
            response = super().create(model=model, input=input)
            response.data = response.data[:-1]
            return response

    embedder = _openai(monkeypatch, _Short())
    with pytest.raises(EmbedderError, match="refusing to guess"):
        embedder(["a", "b"])


def test_the_instance_name_is_used_not_the_class_name(monkeypatch) -> None:
    """``name()`` is static in Chroma's protocol, and this class serves two
    models. A cross-embedder check comparing the static name is a check that
    passes when it should fail."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    large = _openai(monkeypatch, _FakeEmbeddings(), name=OPENAI_LARGE_NAME)

    assert OpenAIEmbedder.name() == OPENAI_SMALL_NAME
    assert embedder_name(large) == OPENAI_LARGE_NAME
    assert embedder_name(HashingEmbedder()) == HASHING_NAME


def test_changing_the_model_is_refused_as_a_config_update(monkeypatch) -> None:
    embedder = _openai(monkeypatch, _FakeEmbeddings())
    with pytest.raises(ValueError, match="rebuild"):
        embedder.validate_config_update(
            {"embedder_name": OPENAI_SMALL_NAME}, {"embedder_name": OPENAI_LARGE_NAME}
        )


def test_auto_never_selects_an_embedder_that_bills(monkeypatch) -> None:
    """A default that quietly starts spending is a default nobody thanks you
    for."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    _embedder, name = get_embedder(prefer="auto")
    assert name in {ONNX_NAME, HASHING_NAME}


def test_short_and_full_embedder_names_mean_the_same_thing(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert get_embedder(prefer="hashing")[1] == get_embedder(prefer=HASHING_NAME)[1]
