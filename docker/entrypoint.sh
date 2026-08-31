#!/bin/sh
# Dispatch a subcommand, and (for readers only) wait for the index the embedding
# pipeline produces.
#
# The embedding pipeline is a separate job on purpose: it is the only writer of
# the Chroma index, it runs to completion and exits, and the query service must
# never start one implicitly. So a reader either finds a manifest or says which
# command produces one.
set -eu

MANIFEST="${RIGHTS_RUNS_DIR:-/var/lib/rights-agent/runs}/index_manifest.json"
WAIT_FOR_INDEX="${RIGHTS_WAIT_FOR_INDEX:-0}"

needs_index() {
    case "$1" in
        demo|ask|compare|evals|evaluate|goldens) return 0 ;;
        *) return 1 ;;
    esac
}

wait_for_index() {
    waited=0
    while [ ! -f "$MANIFEST" ]; do
        if [ "$waited" -ge "$WAIT_FOR_INDEX" ]; then
            cat >&2 <<MSG
no index at $MANIFEST after ${waited}s.

The embedding pipeline runs separately from the query service:

    docker compose run --rm ingest          # hierarchical index (required)
    docker compose run --rm ingest-simple   # fixed-window baseline (optional)

Then start this service again.
MSG
            exit 2
        fi
        if [ $((waited % 10)) -eq 0 ]; then
            echo "waiting for the embedding pipeline to publish $MANIFEST (${waited}/${WAIT_FOR_INDEX}s)" >&2
        fi
        sleep 2
        waited=$((waited + 2))
    done
}

COMMAND="${1:-demo}"
shift 2>/dev/null || true

if needs_index "$COMMAND"; then
    wait_for_index
fi

case "$COMMAND" in
    evals)
        exec python -m pytest evals/ "$@"
        ;;
    tests)
        exec python -m pytest tests/ evals/ "$@"
        ;;
    shell)
        exec /bin/sh
        ;;
    *)
        exec python -m rights_agent "$COMMAND" "$@"
        ;;
esac
