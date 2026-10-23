PY := uv run python

.PHONY: setup lint format test

setup:  ## Create/refresh the venv from uv.lock
	uv sync

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

test:
	uv run pytest

.PHONY: download
download:  ## Fetch KKBox archives from Kaggle (needs ~/.kaggle/access_token + accepted rules)
	$(PY) -m churnguard.data.download

.PHONY: bronze
MONTHS ?=
bronze:  ## Raw -> bronze Parquet. Optional: make bronze MONTHS=2016-10:2017-03
	$(PY) -m churnguard.data.bronze $(if $(MONTHS),--months $(MONTHS))

.PHONY: dq ingest
dq:  ## Data-quality report on bronze tables (exit 1 on FAIL)
	$(PY) -m churnguard.data.dq

ingest: bronze dq  ## Phase 1 end-to-end: raw -> bronze -> DQ gate

.PHONY: archive restore storage-status
LAYER ?=
archive:  ## Upload a data layer to cold storage, verify, free local disk: make archive LAYER=raw
	$(PY) -m churnguard.storage archive $(LAYER)

restore:  ## Pull an archived layer back: make restore LAYER=bronze/user_logs
	$(PY) -m churnguard.storage restore $(LAYER)

storage-status:  ## Local vs archived layers and free disk
	$(PY) -m churnguard.storage status

.PHONY: silver
silver:  ## Bronze -> silver (member_id keyed, cleaned, contract-checked)
	$(PY) -m churnguard.data.silver

.PHONY: spine reference
spine:  ## Point-in-time label spine (gold) + validation vs official labels
	$(PY) -m churnguard.labels.spine

reference:  ## Fetch KKBox's official labeller (not redistributed in git)
	uv run kaggle competitions download kkbox-churn-prediction-challenge -f WSDMChurnLabeller.scala -p docs/reference -q

.PHONY: features feature-report
features:  ## Point-in-time features for every spine month (gold/features)
	$(PY) -m churnguard.features.build

feature-report:  ## Univariate AUC/null report; fails if any single feature looks leaky
	$(PY) -m churnguard.features.diagnostics

.PHONY: baselines telco
baselines:  ## prior / rule / logreg on time splits + shuffled-label leakage control
	$(PY) -m churnguard.modeling.baselines

telco:  ## IBM Telco sanity check of the shared modelling code (expects AUC 0.82-0.88)
	$(PY) -m churnguard.modeling.telco

.PHONY: train mlflow-ui
TRIALS ?=
train:  ## Tune LightGBM on rolling folds (feeds `make select`)
	$(PY) -m churnguard.modeling.train $(if $(TRIALS),--trials $(TRIALS))

mlflow-ui:  ## Browse runs and the model registry at http://127.0.0.1:5000
	MLFLOW_DISABLE_AGENT_HINT=1 uv run mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000

.PHONY: backtest
backtest:  ## Retention targeting backtest (EV vs risk vs random) with the champion model
	$(PY) -m churnguard.decisioning.backtest

.PHONY: registry
registry:  ## List registered model versions and aliases
	$(PY) -m churnguard.tracking list

.PHONY: select
select:  ## Compare candidates on every out-of-time month, gate v2, register + promote
	$(PY) -m churnguard.modeling.select

.PHONY: score serve docker-build docker-run
MONTH ?=
score:  ## Batch-score everyone due in MONTH (default: month after last transaction)
	$(PY) -m churnguard.serving.batch $(if $(MONTH),--month $(MONTH))

serve:  ## Run the scoring API locally on :8000
	uv run uvicorn churnguard.serving.api:app --port 8000

docker-build:  ## Build the API image
	docker build -t churnguard:latest .

# MLflow records artifact URIs as absolute host paths, so mlartifacts is mounted at the
# same absolute path inside the container.
docker-run:  ## Run the API image with data, registry DB and artifacts mounted
	docker run --rm -p 8000:8000 \
	  -v $(CURDIR)/data:/app/data \
	  -v $(CURDIR)/mlruns.db:/app/mlruns.db \
	  -v $(CURDIR)/mlartifacts:$(CURDIR)/mlartifacts:ro \
	  churnguard:latest

.PHONY: requirements
requirements:  ## Regenerate pinned requirements*.txt from uv.lock (run after uv add/remove)
	uv export --format requirements-txt --no-hashes --no-dev --frozen -o requirements.txt
	uv export --format requirements-txt --no-hashes --only-dev --frozen -o requirements-dev.txt
