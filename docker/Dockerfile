# syntax=docker/dockerfile:1.7
#
# Three targets:
#   runtime  — the query service and the pipelines (no dev dependencies)
#   dev      — runtime plus pytest, for the eval suites
#
# The embedding pipeline and the query service share this image deliberately:
# they must agree about the parser, the embedder and the manifest format, and the
# cheapest way to guarantee that is to ship one artefact and pick a subcommand.

ARG PYTHON_VERSION=3.13
ARG UV_VERSION=0.8.4

# --------------------------------------------------------------------------- #
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# --------------------------------------------------------------------------- #
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# poppler-utils provides pdftotext, needed only when the corpus is a PDF.
# tini gives the demo server a real PID 1 so SIGTERM reaches Python.
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt-get update \
 && apt-get install -y --no-install-recommends poppler-utils tini curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app

# --------------------------------------------------------------------------- #
FROM base AS deps

# Dependencies first, from the lockfile only: this layer is cached until
# pyproject.toml or uv.lock changes, which is what keeps rebuilds fast.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra trace --extra models

# --------------------------------------------------------------------------- #
FROM deps AS deps-dev
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra trace --extra models --group dev

# --------------------------------------------------------------------------- #
FROM base AS runtime

COPY --from=deps /opt/venv /opt/venv
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY data ./data
COPY evals ./evals
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra trace --extra models --no-editable \
 && chmod +x /usr/local/bin/entrypoint.sh \
 && useradd --create-home --uid 10001 rights \
 && mkdir -p /var/lib/rights-agent/runs /home/rights/.cache/chroma \
 && chown -R rights:rights /var/lib/rights-agent /app /home/rights/.cache

# Bake the ONNX MiniLM model into the image.
#
# Chroma downloads it (79MB) on first use, into the *container's* home cache.
# Every embedding job here runs as `docker compose run --rm`, so without this
# that download happens once per invocation -- and the first one it would happen
# in front of an audience is the dashboard's startup. Fetching at build time
# turns a recurring 20s network dependency into image bytes.
#
# The build fails if the fetch fails, on purpose: a silently-absent model is
# indistinguishable from a baked one until the demo, when it is a download. For
# an air-gapped build pass --build-arg PREFETCH_EMBEDDER=0 and run with
# RIGHTS_EMBEDDER=hashing, which needs no model at all.
ARG PREFETCH_EMBEDDER=1
USER rights
RUN if [ "$PREFETCH_EMBEDDER" = "1" ]; then \
      python -c "from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2 as E; E()(['warm'])" \
      && echo "onnx-all-MiniLM-L6-v2 baked in"; \
    else echo "embedder prefetch skipped"; fi
USER root

ENV RIGHTS_DATA_DIR=/app/data \
    RIGHTS_RUNS_DIR=/var/lib/rights-agent/runs \
    RIGHTS_DEMO_HOST=0.0.0.0 \
    RIGHTS_DEMO_PORT=8000 \
    RIGHTS_LOG_JSON=true \
    RIGHTS_EMBEDDER=hashing \
    PHOENIX_COLLECTOR_ENDPOINT=http://phoenix:6006

USER rights
VOLUME ["/var/lib/rights-agent/runs"]
EXPOSE 8000

# The index lives on a volume the pipeline writes and this service reads, so the
# health check reports readiness of *both*.
HEALTHCHECK --interval=15s --timeout=4s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["demo"]

# --------------------------------------------------------------------------- #
FROM runtime AS dev

USER root
COPY --from=deps-dev /opt/venv /opt/venv
COPY tests ./tests
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra trace --extra models --group dev --no-editable \
 && chown -R rights:rights /app
USER rights
CMD ["evals"]
