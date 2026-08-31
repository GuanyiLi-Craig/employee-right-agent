"""Embedders, and the rule that stops you querying an index with the wrong one.

Three implementations behind one interface. Measured on the real Employment
Rights Act 2025 (2,078 leaves), recall of the golden set's 30 expected citations:

=================================  ========  ===========  ========  ============
embedder                            recall    query p50    ingest    needs
=================================  ========  ===========  ========  ============
``openai-text-embedding-3-small``    100.0%      143 ms      ~17 s   a key, ~1c
``onnx-all-MiniLM-L6-v2``             90.0%       53 ms       60 s   an 80MB model
``hashing-bow-512``                   86.7%        2 ms        3 s   nothing
=================================  ========  ===========  ========  ============

The 17-point spread between the cheapest and the best is the argument for
measuring retrieval separately from generation: none of it is visible in the
answer, which reads fluently either way, and all of it is visible in whether the
right provision was in the context at all.

The hashing fallback is **lexical, not semantic**.  It works on statutes because
legislation repeats its own vocabulary and because every embedded string starts
with a breadcrumb full of exact terms.  It would be a poor choice for a
paraphrase-heavy corpus such as support tickets, and saying so is part of
teaching it honestly.  It is also the only one of the three that is
*bit-deterministic*, which is why it stays the offline default: a gate whose
vectors can move under it is a gate that fails for reasons unrelated to the
change under test.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import time
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise
from typing import Any

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings, Space
from chromadb.utils.embedding_functions import register_embedding_function

from rights_agent.log import get_logger

log = get_logger("embedding")

HASHING_NAME = "hashing-bow-512"
ONNX_NAME = "onnx-all-MiniLM-L6-v2"
OPENAI_SMALL_NAME = "openai-text-embedding-3-small"
OPENAI_LARGE_NAME = "openai-text-embedding-3-large"

#: The API model behind each hosted name. The name carries the model because it
#: goes into ``index_version``, and two indexes with the same version string have
#: to contain the same vectors.
OPENAI_MODELS = {
    OPENAI_SMALL_NAME: "text-embedding-3-small",
    OPENAI_LARGE_NAME: "text-embedding-3-large",
}

#: Short forms accepted in ``RIGHTS_EMBEDDER``, because nobody wants to type
#: ``openai-text-embedding-3-small`` into a shell twice.
PREFERENCE_ALIASES = {
    "hashing": HASHING_NAME,
    "onnx": ONNX_NAME,
    "openai": OPENAI_SMALL_NAME,
    "openai-small": OPENAI_SMALL_NAME,
    "openai-large": OPENAI_LARGE_NAME,
}

#: Names this process knows how to build, in preference order.
KNOWN_EMBEDDERS = (OPENAI_SMALL_NAME, OPENAI_LARGE_NAME, ONNX_NAME, HASHING_NAME)


class EmbedderError(RuntimeError):
    """Raised when the requested embedder cannot be constructed."""


# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[a-z0-9£][a-z0-9£'\-]*")

#: Small, explicit stoplist.  Long stoplists remove terms that carry meaning in
#: legal text ("no", "not", "must", "may"), so this one is deliberately short
#: and keeps every modal verb.
STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for", "from", "had", "has", "have", "he", "her", "his", "if", "in", "into", "is", "it", "its", "of", "on", "or", "our", "ours", "she", "that", "the", "their", "them", "there", "these", "they", "this", "those", "to", "was", "were", "which", "who", "whom", "whose", "will", "with", "would", "you", "your"]
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with the stoplist removed."""
    return [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def _bucket(term: str, dim: int) -> int:
    """Stable bucket for ``term``.

    ``hash()`` is salted per process, so using it here would make an index
    unreadable by the next process that opened it -- a silent, total failure.
    BLAKE2b is stable across processes, machines and Python versions.
    """
    digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


# --------------------------------------------------------------------------- #
# Offline embedder
# --------------------------------------------------------------------------- #
@register_embedding_function
class HashingEmbedder(EmbeddingFunction[Documents]):
    """Deterministic hashed bag-of-words with bigrams, L2-normalised.

    Bigrams matter: without them "guaranteed hours" is indistinguishable from
    the union of "guaranteed" and "hours", and a statute is full of multi-word
    terms of art whose parts are common words.

    Sub-linear term weighting (``1 + log(count)``) stops a provision that
    repeats one word forty times from dominating the vector.
    """

    DIM = 512

    def __init__(self, dim: int = DIM) -> None:
        if dim < 32:
            raise ValueError("dim must be at least 32 to avoid pathological collisions")
        self.dim = dim

    # ---- Chroma 1.x embedding-function protocol ---------------------------
    @staticmethod
    def name() -> str:
        return HASHING_NAME

    def default_space(self) -> Space:
        return "cosine"

    def supported_spaces(self) -> list[Space]:
        return ["cosine", "ip"]

    def get_config(self) -> dict[str, Any]:
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> HashingEmbedder:
        return HashingEmbedder(dim=int(config.get("dim", HashingEmbedder.DIM)))

    def validate_config(self, config: dict[str, Any]) -> None:
        if int(config.get("dim", self.DIM)) < 32:
            raise ValueError("dim must be at least 32")

    def validate_config_update(
        self, old_config: dict[str, Any], new_config: dict[str, Any]
    ) -> None:
        if old_config.get("dim") != new_config.get("dim"):
            raise ValueError(
                "changing the embedding dimension invalidates the whole index; "
                "rebuild it instead"
            )

    # ---- embedding --------------------------------------------------------
    def _embed_one(self, text: str) -> list[float]:
        tokens = tokenize(text)
        counts: Counter[str] = Counter(tokens)
        counts.update(f"{a}_{b}" for a, b in pairwise(tokens))
        vector = [0.0] * self.dim
        for term, count in counts.items():
            vector[_bucket(term, self.dim)] += 1.0 + math.log(count)
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # An empty or all-stopword input.  A zero vector would make cosine
            # distance undefined, so pin it to a single reserved bucket.
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 - Chroma's name
        return [self._embed_one(text or "") for text in input]  # type: ignore[return-value]

    def embed_documents(self, input: Documents) -> Embeddings:  # noqa: A002
        return self.__call__(input)

    def embed_query(self, input: Documents) -> Embeddings:  # noqa: A002
        # Symmetric model: queries and documents share one vector space.
        return self.__call__(input)


# --------------------------------------------------------------------------- #
# ONNX embedder
# --------------------------------------------------------------------------- #
@register_embedding_function
class OnnxMiniLMEmbedder(EmbeddingFunction[Documents]):
    """Chroma's bundled ONNX ``all-MiniLM-L6-v2``, wrapped for a stable name.

    Wrapped rather than used directly so the manifest records a name that says
    which model produced the vectors, instead of Chroma's generic ``default``.
    """

    def __init__(self) -> None:
        try:
            from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import (
                ONNXMiniLM_L6_V2,
            )
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise EmbedderError(f"ONNX embedder unavailable: {exc}") from exc
        try:
            self._inner = ONNXMiniLM_L6_V2()
            # Force the model download/verification now, so a network failure
            # surfaces here rather than half way through an ingest.
            self._inner(["warmup"])
        except Exception as exc:
            raise EmbedderError(
                f"ONNX MiniLM could not be initialised ({exc}). It downloads a model on "
                "first use; run with RIGHTS_EMBEDDER=hashing to stay offline."
            ) from exc

    @staticmethod
    def name() -> str:
        return ONNX_NAME

    def default_space(self) -> Space:
        return "cosine"

    def get_config(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> OnnxMiniLMEmbedder:
        return OnnxMiniLMEmbedder()

    def validate_config(self, config: dict[str, Any]) -> None:
        return None

    def validate_config_update(
        self, old_config: dict[str, Any], new_config: dict[str, Any]
    ) -> None:
        return None

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        return self._inner(input)

    def embed_documents(self, input: Documents) -> Embeddings:  # noqa: A002
        return self.__call__(input)

    def embed_query(self, input: Documents) -> Embeddings:  # noqa: A002
        return self.__call__(input)


# --------------------------------------------------------------------------- #
# OpenAI embedder
# --------------------------------------------------------------------------- #
#: Inputs per request. The endpoint accepts far more, but a failed request costs
#: the whole batch, and a 2,141-chunk ingest that dies on the last input and
#: retries everything is a worse trade than a few extra round trips.
OPENAI_BATCH = 128

#: Bounded, and small. A rate limit that survives four tries is a rate limit
#: that needs a smaller batch or a bigger quota, not more patience -- and an
#: ingest that hangs silently for minutes is harder to diagnose than one that
#: stops and says which batch failed.
OPENAI_MAX_ATTEMPTS = 4
OPENAI_BACKOFF_S = 1.5

#: text-embedding-3-* accept 8,191 tokens. Chunks here are capped well below
#: that by ``max_parent_chars``, but a corpus this code has not seen could
#: exceed it, and the API's answer to that is a 400 for the whole batch. Four
#: characters per token is the usual rough ratio for English prose.
OPENAI_MAX_CHARS = 8_000 * 4


@register_embedding_function
class OpenAIEmbedder(EmbeddingFunction[Documents]):
    """OpenAI ``text-embedding-3-*`` over the REST API.

    The strongest retrieval of the three and the only one with a per-token bill,
    which makes it the honest place to point out that an embedder is a cost
    centre as well as a quality knob: re-indexing this corpus costs about a
    penny, and re-indexing is something you do every time the parser changes.

    Not bit-deterministic. The service does not promise identical floats across
    calls, so two indexes built from the same corpus can rank near-ties
    differently. That is fine for serving and wrong for a gate, which is why
    :data:`HASHING_NAME` remains the offline default.
    """

    def __init__(self, name: str = OPENAI_SMALL_NAME, api_key_env: str = "OPENAI_API_KEY") -> None:
        if name not in OPENAI_MODELS:
            raise EmbedderError(f"unknown OpenAI embedder {name!r}; expected one of {sorted(OPENAI_MODELS)}")
        self._name = name
        self.model = OPENAI_MODELS[name]
        self.api_key_env = api_key_env
        key = os.environ.get(api_key_env)
        if not key:
            raise EmbedderError(
                f"{api_key_env} is not set. Set it, or run with "
                f"RIGHTS_EMBEDDER=onnx (a local model) or =hashing (no model)."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise EmbedderError(f"the openai package is not installed: {exc}") from exc
        # Passed explicitly: the SDK otherwise reads OPENAI_API_KEY from the
        # environment, which silently ignores api_key_env and made an earlier
        # bug here look like an auth problem at the provider.
        self._client = OpenAI(api_key=key)
        self.tokens_used = 0

    @staticmethod
    def name() -> str:
        # Chroma's protocol wants a static name; the instance's own name is on
        # ``instance_name`` because two models share this class.
        return OPENAI_SMALL_NAME

    @property
    def instance_name(self) -> str:
        return self._name

    def default_space(self) -> Space:
        return "cosine"

    def supported_spaces(self) -> list[Space]:
        return ["cosine", "ip"]

    def get_config(self) -> dict[str, Any]:
        return {"embedder_name": self._name}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> OpenAIEmbedder:
        return OpenAIEmbedder(str(config.get("embedder_name") or OPENAI_SMALL_NAME))

    def validate_config(self, config: dict[str, Any]) -> None:
        name = config.get("embedder_name")
        if name and name not in OPENAI_MODELS:
            raise ValueError(f"unknown OpenAI embedder {name!r}")

    def validate_config_update(
        self, old_config: dict[str, Any], new_config: dict[str, Any]
    ) -> None:
        if old_config.get("embedder_name") != new_config.get("embedder_name"):
            raise ValueError(
                "changing the embedding model invalidates the whole index; "
                "rebuild it instead"
            )

    # ---- embedding --------------------------------------------------------
    def _request(self, batch: list[str]) -> list[list[float]]:
        last: Exception | None = None
        for attempt in range(1, OPENAI_MAX_ATTEMPTS + 1):
            try:
                response = self._client.embeddings.create(model=self.model, input=batch)
            except Exception as exc:  # noqa: BLE001 - the SDK raises many types
                last = exc
                if attempt == OPENAI_MAX_ATTEMPTS:
                    break
                delay = OPENAI_BACKOFF_S * (2 ** (attempt - 1))
                log.warning(
                    "embedding batch of %d failed (attempt %d/%d): %s; retrying in %.1fs",
                    len(batch), attempt, OPENAI_MAX_ATTEMPTS, exc, delay,
                )
                time.sleep(delay)
                continue
            usage = getattr(response, "usage", None)
            self.tokens_used += int(getattr(usage, "total_tokens", 0) or 0)
            # Sorted by index, not trusted to arrive in order: a reordered batch
            # would attach every vector to the wrong chunk, and nothing
            # downstream could detect it.
            items = sorted(response.data, key=lambda item: item.index)
            vectors = [list(item.embedding) for item in items]
            if len(vectors) != len(batch):
                raise EmbedderError(
                    f"asked for {len(batch)} embeddings and got {len(vectors)}; "
                    "refusing to guess which input each vector belongs to"
                )
            return vectors
        raise EmbedderError(
            f"embedding failed after {OPENAI_MAX_ATTEMPTS} attempts: {last}"
        ) from last

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 - Chroma's name
        # The API rejects an empty input, and layout extraction can leave a chunk
        # that is empty or whitespace-only. Those become a single space rather
        # than an error that stops an ingest 2,000 chunks in -- and a space is
        # used rather than a placeholder word so the vector carries no content
        # that could make an empty chunk retrievable.
        texts = [
            (text[:OPENAI_MAX_CHARS] if (text or "").strip() else " ") for text in input
        ]
        out: list[list[float]] = []
        for start in range(0, len(texts), OPENAI_BATCH):
            out.extend(self._request(texts[start : start + OPENAI_BATCH]))
        return out  # type: ignore[return-value]

    def embed_documents(self, input: Documents) -> Embeddings:  # noqa: A002
        return self.__call__(input)

    def embed_query(self, input: Documents) -> Embeddings:  # noqa: A002
        # Symmetric model: text-embedding-3-* uses one space for both sides.
        return self.__call__(input)


# --------------------------------------------------------------------------- #
# Selection and pinning
# --------------------------------------------------------------------------- #
def embedder_name(embedder: EmbeddingFunction[Documents]) -> str:
    """The name of *this* embedder, not of its class.

    Chroma's protocol makes ``name()`` a static method, which is fine while one
    class means one set of vectors. :class:`OpenAIEmbedder` serves two models, so
    its ``name()`` can only name one of them -- and a cross-embedder check that
    compares a static name is a check that passes when it should fail. Instances
    that cover several models expose ``instance_name``; everything else answers
    with ``name()``.
    """
    return str(getattr(embedder, "instance_name", None) or embedder.name())


def build_embedder(name: str) -> EmbeddingFunction[Documents]:
    """Construct the embedder called ``name``, or raise.

    Exact, never approximate: this is the function a *reader* goes through, and
    reproducing the manifest's embedder is the difference between a correct
    answer and confident nonsense.
    """
    if name == HASHING_NAME:
        return HashingEmbedder()
    if name == ONNX_NAME:
        return OnnxMiniLMEmbedder()
    if name in OPENAI_MODELS:
        return OpenAIEmbedder(name)
    raise EmbedderError(f"unknown embedder {name!r}; expected one of {KNOWN_EMBEDDERS}")


def get_embedder(
    *,
    require: str | None = None,
    prefer: str = "auto",
) -> tuple[EmbeddingFunction[Documents], str]:
    """Return ``(embedder, name)``.

    ``require`` is the name recorded in the index manifest.  When it is set the
    resolution is exact: we build that embedder or we fail.  **Querying a hashed
    index with MiniLM vectors does not raise -- it returns confident nonsense**,
    which is the nastiest silent failure in retrieval, so the only safe
    behaviour is to refuse to start.
    """
    if require:
        embedder = build_embedder(require)
        return embedder, require

    # A short alias or a full name -- symmetrically for all three. Asking for
    # `hashing` and asking for `hashing-bow-512` mean the same thing, which
    # matters because the full name is what the manifest and every error message
    # print, so it is the one an operator has in front of them.
    chosen = PREFERENCE_ALIASES.get(prefer, prefer)
    if chosen in KNOWN_EMBEDDERS:
        return build_embedder(chosen), chosen
    if prefer != "auto":
        raise EmbedderError(
            f"unknown embedder preference {prefer!r}; expected auto or one of "
            f"{sorted(set(PREFERENCE_ALIASES) | set(KNOWN_EMBEDDERS))}"
        )

    # `auto` never reaches for a paid API on its own. An embedder that bills per
    # token is a deliberate choice, and a default that quietly starts spending
    # is the kind of default nobody thanks you for.
    try:
        embedder = OnnxMiniLMEmbedder()
    except EmbedderError as exc:
        log.warning("falling back to the offline embedder: %s", exc)
        return HashingEmbedder(), HASHING_NAME
    return embedder, ONNX_NAME


def assert_embedder_matches(recorded: str | None, resolved: str) -> None:
    """The pinning rule (§10.2).  Call this before the first query."""
    if recorded and recorded != resolved:
        raise EmbedderError(
            f"index was built with embedder {recorded!r} but this process resolved "
            f"{resolved!r}. Querying across embedders returns confident nonsense "
            f"rather than an error, so this is fatal. Either set "
            f"RIGHTS_EMBEDDER to match the index, or rebuild the index."
        )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Only used by tests and diagnostics; Chroma does this internally."""
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
