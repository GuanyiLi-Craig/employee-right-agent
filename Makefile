# Local development. Docker equivalents are in docker-compose.yml.
#
# Every target uses `uv run`: never activate a venv by hand, that is where
# "works on my machine" comes from.

.DEFAULT_GOAL := help
.PHONY: help install lint lint-fix corpus ingest ingest-simple ask compare demo goldens \
        evaluate gate calibrate test test-unit test-evals clean \
        docker-build docker-ingest docker-up docker-down docker-evals docker-logs ui-test phoenix pentest

UV ?= uv
QUESTION ?= What does the document say about bereavement leave?
UI_BASE ?= http://localhost:8000
NUCLEI  ?= nuclei

help: ## Show this help
	@grep -hE '^[a-z0-9-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Sync the environment from the lockfile
	$(UV) sync --extra trace --extra models --group dev

lint: ## Ruff, the same check CI runs
	$(UV) run ruff check .

lint-fix: ## Apply the fixes ruff can make safely
	$(UV) run ruff check . --fix

corpus: ## Regenerate the committed demonstration corpus
	$(UV) run rights-corpus --out data/corpus.layout.txt

ingest: ## Build the hierarchical index (the embedding pipeline)
	$(UV) run rights-ingest --no-onnx

ingest-simple: ## Build the fixed-window baseline index
	$(UV) run rights-ingest-simple --no-onnx

ask: ## Ask one question: make ask QUESTION="..."
	$(UV) run rights-ask "$(QUESTION)"

compare: ## Fixed windows against the hierarchical index
	$(UV) run rights-compare

demo: ## Serve the dashboard on http://127.0.0.1:8000
	$(UV) run rights-demo

goldens: ## Regenerate the eval datasets and the baseline
	$(UV) run python -m rights_agent goldens --write-baseline

evaluate: ## Run the golden set and report quality
	$(UV) run python -m rights_agent evaluate

gate: ## Report the CI gate's numbers without asserting them
	$(UV) run python -m rights_agent evaluate --gate

calibrate: ## Judge calibration: kappa with and without the hard cases
	$(UV) run python -m rights_agent evaluate --calibration

test: ## Unit tests and both eval suites
	$(UV) run pytest -q

test-unit: ## Unit tests only (no index required)
	$(UV) run pytest tests/ -q

test-evals: ## The CI gate: deterministic then quality
	$(UV) run pytest evals/test_deterministic.py -q
	$(UV) run pytest evals/test_quality.py -q

clean: ## Remove local artefacts (keeps the corpus and the datasets)
	rm -rf runs .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

docker-build: ## Build both image targets (dev is on the tools profile)
	docker compose build
	docker compose --profile tools build evals   # NOT covered by `docker compose build`

docker-ingest: ## Run the embedding pipeline as a one-shot job
	docker compose run --rm ingest
	docker compose run --rm ingest-simple

docker-up: ## Start phoenix and the dashboard
	docker compose up -d

docker-down: ## Stop everything (keeps the index volume)
	docker compose down

docker-evals: ## Run the CI gate in the container
	docker compose run --rm evals

docker-logs: ## Follow the dashboard log
	docker compose logs -f dashboard

.PHONY: pentest
pentest: ## YAML-template scan of the dashboard (needs nuclei on PATH or NUCLEI=)
	$(NUCLEI) -u $(UI_BASE) -t security/nuclei/ -no-color -disable-update-check
	@echo "--- community library (slow, ~8 min) ---"
	$(NUCLEI) -u $(UI_BASE) -silent -no-color -exclude-tags dos,fuzz -rate-limit 40

.PHONY: phoenix
phoenix: ## Upload the golden set to Phoenix and run an experiment (costs money)
	docker compose --profile tools run --rm evaluate --phoenix

.PHONY: ui-test
ui-test: ## Drive the dashboard in a browser and assert on what is on screen
	cd uitest && npm install --silent && BASE=$(UI_BASE) npm run all
