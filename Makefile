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
train:  ## Tune + calibrate + explain LightGBM, log to MLflow, register, promotion gate
	$(PY) -m churnguard.modeling.train $(if $(TRIALS),--trials $(TRIALS))

mlflow-ui:  ## Browse runs and the model registry at http://127.0.0.1:5000
	MLFLOW_DISABLE_AGENT_HINT=1 uv run mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000

.PHONY: backtest
backtest:  ## Retention targeting backtest (EV vs risk vs random) with the champion model
	$(PY) -m churnguard.decisioning.backtest
