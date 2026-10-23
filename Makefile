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
