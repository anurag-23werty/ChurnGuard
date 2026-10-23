# ChurnGuard

End-to-end churn prediction and retention platform on the KKBox music-subscription dataset
(~400M listening-log rows): out-of-core data engineering, point-in-time features,
a calibrated and explainable model, value-based retention targeting, serving, and monitoring.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.

## Status

- [x] Phase 0: architecture
- [x] Phase 1: ingestion (raw → bronze Parquet)
- [x] Phase 2: cleaning (silver)
- [x] Phase 3: label spine
- [x] Phase 4: point-in-time features
- [x] Phase 5: baselines
- [x] Phase 6: model selection (LightGBM, blends, calibration, SHAP, MLflow registry)
- [x] Phase 7: retention decisioning (expected-value targeting + backtest)
- [x] Phase 8: serving (batch scorer, FastAPI, Docker)
- [ ] Phase 9: monitoring
- [ ] Phase 10: UI & CI

## Install

Python 3.11+ (developed on 3.13). Either tool works; both use the same pinned versions.

```bash
# uv (recommended)
uv sync

# or plain pip
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt   # runtime + tests/lint
pytest                                                     # 99 tests, ~15 s
```

`requirements*.txt` are generated from `uv.lock` (`make requirements`); don't edit by hand.
System tools, not pip packages: `brew install sevenzip` (ingest), `brew install rclone`
(cold storage), Docker Desktop (container). Kaggle API token in `~/.kaggle/access_token`.

## Quickstart

```bash
brew install sevenzip            # streams .7z archives without extracting to disk
uv sync                          # create .venv from uv.lock
make test

# Kaggle: save your API token to ~/.kaggle/access_token (chmod 600) and accept the rules first
make download                    # data/raw/*.7z
make ingest                      # raw -> bronze Parquet + DQ report
make ingest MONTHS=2016-10:2017-03   # or a smaller window on a constrained machine
make silver                      # bronze -> silver (cleaned, member_id keyed)
make spine                       # label spine + validation vs official labels
make features                    # 52 point-in-time features per spine row
make feature-report              # univariate AUCs + leak smell test
make telco                       # sanity-check modelling code on IBM Telco
make baselines                   # prior / rule / logreg + permutation control
make train                       # tune LightGBM on rolling folds inside train months
make select                      # all candidates x all out-of-time months, gate, register
make registry                    # registered versions + aliases (champion/challenger)
make mlflow-ui                   # browse runs at http://127.0.0.1:5000
make backtest                    # EV vs risk vs random targeting on Feb, save-rate sensitivity
make score                       # batch-score everyone due next month with the champion
make serve                        # scoring API on :8000 (docs at /docs)
make docker-build docker-run     # same API in a container
```

Bronze layout: `data/bronze/<table>/month=YYYY-MM/<source>.parquet` with a `_manifest.json`
reconciling rows read vs. written. DQ reports land in `reports/dq/<table>.json`.

Silver layout: `data/silver/<table>/` keyed by an int32 `member_id` (append-only map in
`silver/member_ids`), with per-rule hit counts and bronze row reconciliation in each
`_manifest.json`. On the 2016-07..2017-03 window:

| Table | Bronze rows | Silver rows | Bronze size | Silver size | Notes |
|---|---|---|---|---|---|
| user_logs | 155,544,791 | 155,544,791 | 6.4 GB | 1.4 GB | 37,244 `total_secs` clipped to [0, 86400] |
| transactions | 22,978,755 | 22,975,416 | 775 MB | 60 MB | 3,339 exact dups dropped; 1,776 1970 expiries nulled |
| members | 6,769,473 | 6,769,473 | 225 MB | 36 MB | 4.55M ages outside 10-80 nulled |
| labels | 1,963,891 | 1,963,891 | 62 MB | 5.7 MB | |

Full silver build: 46 s.

## Cold storage

Raw archives and bronze are cold data: read rarely, large. They live on an rclone remote
(Google Drive by default, `storage.remote` in `configs/base.yaml`); silver/gold stay local.

```bash
make archive LAYER=raw              # upload, verify size+hash with rclone check, free local disk
make restore LAYER=bronze/user_logs # pull back on demand
make storage-status                 # what is local vs archived
```

One-time setup: `rclone config` a Drive remote named `gdrive` with **your own OAuth client ID**
and scope `drive.file` (files are only visible to the client ID that uploaded them, so never
upload with rclone's shared client). The OAuth app stays in Google "Testing" mode, so tokens
expire every 7 days: `rclone config reconnect gdrive:`.

## Label spine

`data/gold/spine/label_month=YYYY-MM/` holds one row per (member, month M) with
`as_of_date` = first day of M: the decision point. Population and paid-through date use only
transactions before it; features must too. Port of KKBox's official `WSDMChurnLabeller.scala`
(`make reference`), with two deliberate, measured changes:

- **Population:** any transaction before M expiring in M, over all history (the Scala file
  uses only M-1's last transaction). Recall vs official files: 86% -> **99%**.
- **Label maturity:** rows are `decided` or `pending` (is_churn NULL) based on whether
  effective expiry + 30 days is inside the data, instead of counting unseen renewals as churn.

| Month | Members | Pending | Churn (decided) | Official pop. recall / precision | Label agreement |
|---|---|---|---|---|---|
| 2016-10 | 944,849 | 1,224 | 4.52% | | |
| 2016-11 | 1,000,052 | 1,489 | 9.44% | | |
| 2016-12 | 991,108 | 1,821 | 8.42% | | |
| 2017-01 | 977,500 | 2,534 | 5.41% | | |
| 2017-02 | 984,196 | 4,395 | 4.71% | 99.1% / 99.9% (train.csv) | 98.0% |
| 2017-03 | 991,602 | 49,676 | (immature) | 98.7% / 96.6% (train_v2) | 96.0% |

Known gap: ~1/3 of disagreements are "renewed then cancelled, never renewed again", which the
official labels count as churn and the Scala logic does not. March needs April
transactions to mature, so the March holdout uses the official train_v2 labels.

## Features

`data/gold/features/label_month=YYYY-MM/` = spine keys + label + 52 features, all computed
from events strictly before `as_of_date`. `src/churnguard/features/registry.yaml` is the
contract (family, source, window, dtype, description); the builder refuses unregistered or
missing columns. Families: **member** (7), **billing** (24), **engagement** (21; windows
capped at 90 days by log coverage). ~1M rows/month in ~4.5 s.

Leakage defences, each with a test that fails if the guard is removed (verified by
mutating the code):

- **Poisoned future:** extreme events on/after `as_of_date` in every source; features must
  be byte-identical.
- **Mixed as-of dates:** one member at two as-of dates in one query; the earlier row must
  not see the later events (the case serving and backfills hit).
- **Late registrations:** `registration_date` after `as_of_date` gives NULL tenure.
- **Leak smell test** (`make feature-report`): fails if any single feature has AUC >= 0.95.
  Current max: `auto_renew_share_180d` 0.775.

## Baselines

Time-ordered splits (`modeling` in `configs/base.yaml`): **train** 2016-10..2017-01 (1M-row
sample of 3.9M decided rows), **valid** 2017-02 (979,801 rows, derived labels), **test**
2017-03 (958,189 rows, **official train_v2 labels**).

| valid (Feb 2017) | ROC AUC | PR AUC | log loss | ECE | precision@10% | lift@10% |
|---|---|---|---|---|---|---|
| prior | 0.500 | 0.047 | 0.1945 | 0.023 | 0.047 | 1.0 |
| rule: auto-renew x cancel | 0.849 | 0.370 | 0.1275 | 0.015 | 0.308 | 6.5 |
| logistic regression | **0.893** | **0.610** | **0.1094** | **0.006** | 0.356 | **7.6** |

| test (Mar 2017, official labels) | ROC AUC | PR AUC | log loss | ECE |
|---|---|---|---|---|
| rule | 0.786 | 0.335 | 0.2299 | 0.025 |
| logistic regression | 0.798 | 0.486 | 0.2409 | 0.038 |

**Valid -> test drop (0.893 -> 0.798) has two parts.** Scoring the same February members
against official `train.csv` labels instead of derived ones gives 0.856: ~0.036 of the drop
is the label definition (official labels also count renew-then-cancel as churn: 5.9% vs
4.7% base rate), the remaining ~0.058 is March itself. Test log loss is worse than the rule
because the model is calibrated to the derived 7% training rate, not the official 8.8%.

Sanity checks: **IBM Telco** logreg AUC 0.852 (expected 0.82-0.88) validates the shared
modelling code; the **permutation control** (logreg on 5 label permutations) gives valid
AUCs 0.41-0.56 around 0.5 (mean 0.483 +/- 0.025 SE) while the real model scores 0.893.
A single permuted model is *not* a valid check here: random coefficients over strongly
predictive features project onto the signal by chance (0.42-0.59 across seeds).

## Model selection

**Champion: logistic regression** (`models:/churnguard@champion`, v1). Five candidates
were compared on every out-of-time month we have: each month scored by a model trained
only on earlier months (expanding window), Feb 2017 included. March stays the untouched
test month.

| ROC AUC by scored month | Nov 16 | **Dec 16** | Jan 17 | Feb 17 | mean | worst |
|---|---|---|---|---|---|---|
| logreg | 0.959 | **0.832** | 0.899 | 0.893 | 0.896 | **0.832** |
| lgbm (Optuna, 25 trials) | 0.964 | 0.781 | 0.908 | 0.915 | 0.892 | 0.781 |
| lgbm_robust (regularised + monotone) | 0.964 | 0.782 | 0.906 | 0.914 | 0.891 | 0.782 |
| blend lgbm + logreg | 0.963 | 0.806 | 0.904 | 0.908 | 0.896 | 0.806 |
| blend robust + logreg | 0.963 | 0.807 | 0.903 | 0.907 | 0.895 | 0.807 |

| PR AUC by scored month | Nov 16 | Dec 16 | Jan 17 | Feb 17 | mean | mean log loss |
|---|---|---|---|---|---|---|
| logreg | 0.867 | 0.658 | 0.650 | 0.610 | 0.696 | 0.1321 |
| lgbm | 0.872 | 0.657 | 0.654 | 0.638 | 0.705 | 0.1325 |
| blend lgbm + logreg | **0.874** | **0.660** | **0.663** | 0.632 | **0.707** | **0.1300** |

**Why logreg, when the blend wins PR AUC in every month?** The promotion gate (v2,
`tracking.promotion_decision`) requires mean PR AUC >= baseline + 0.005, mean log loss <=
baseline, **worst-month ROC AUC within 0.01 of the baseline's**, latest-month ECE <= 0.01,
a passed permutation control and beating the current champion. Every tree-based candidate
fails the worst-month rule on December 2016 (blends 0.806-0.807 vs logreg 0.832); the
single-model LightGBMs also lose on mean log loss. This is a deliberate robustness-first
choice: a model that ranks well in steady months but degrades sharply in a shift month is
the riskier one to put in front of a retention budget. With more months of history the
blend is the first candidate to re-test; the tolerance is a business decision.

What the comparison ruled out:

- **Monotone constraints + heavier regularisation did not fix December** (0.782 vs 0.781),
  so December is not a "wrong direction" overfit; it is a shift the trees cannot extrapolate.
- **Single-month gating is unsafe.** The first LightGBM (`churnguard-gbm` v1) won Feb by
  0.022 AUC and was auto-promoted by a Feb-only gate; it is now demoted to challenger.
- **Calibration was not applied.** Isotonic fitted on Nov-Jan out-of-time scores and tested
  on Feb *worsened* both log loss (0.1094 -> 0.1099) and ECE (0.0063 -> 0.0073), so the
  conditional check left logreg's own probabilities in place.

**March test** (official train_v2 labels; final model trained on all history through Feb,
1.25M rows, Feb subsampled to an average month):

| logreg, Mar 2017 | ROC AUC | PR AUC | log loss | ECE | lift@10% |
|---|---|---|---|---|---|
| all members (958,189) | 0.799 | 0.485 | 0.2427 | 0.039 | 5.7 |
| excluding early renewers (91.2%) | **0.853** | **0.528** | 0.2006 | 0.028 | **6.3** |

*Early renewers* (paid-through date already past March, 8.8% of the test month) churn at
16.5% in the official March labels vs 2.3% in February, and no model ranks them better
than chance (AUC 0.51-0.55): the official labeller saw April events we do not have. They
are reported separately rather than allowed to dominate the test number.

LightGBM's exact TreeSHAP (`pred_contrib`) gives per-member reason codes for tree models
(`ChurnModel.reason_codes`); top global drivers in the tuned LightGBM were
`last_payment_method`, `last_is_auto_renew`, `days_since_last_txn`, `days_to_expiry` and
`last_list_price`. Logistic-regression reason codes are added with serving (Phase 8).

## Retention decisioning

Contacting member *i* is worth **EV = p(churn) x save rate x CLV - cost**, where CLV is
plan-normalised monthly revenue over a 12-month horizon (discounted by 6% baseline monthly
churn, 9.3 expected months) and cost = 2 NTD contact + one free month. Members with EV <= 0
are never contacted; with a budget, selection is greedy by EV/cost
(`decisioning/retention.py`, parameters in `configs/base.yaml: retention`).

**Backtest** (`make backtest`): champion model scores Feb 2017 (979,801 members, 4.7%
churn); realised value uses the members' actual outcomes, assuming a 20% save rate.

| contact top x% | by expected value | by churn risk | random |
|---|---|---|---|
| 1% | **+801k NTD** | +647k | -1.2M |
| 2% | **+969k** | +963k | -2.4M |
| 5% | -466k | -638k | -5.9M |
| 10% | -5.2M | -6.0M | -11.8M |
| 30% | -24.6M | -31.0M | -35.3M |

- **The EV rule sizes the campaign by itself.** Contacting only positive-EV members selects
  1.4% (13,865) and realises **+976k NTD**, slightly more than the best fixed cut-off found
  in hindsight (2%, +969k). That only works because the probabilities are calibrated
  (ECE 0.006): a miscalibrated model would pick the wrong number of members.
- **Ranking by value beats ranking by risk** where it matters: +24% at 1%, and it loses
  less past the optimum (-5.2M vs -6.0M at 10%). Risk ranking alone spends offers on
  members who pay little or nothing.
- **A free month is an expensive offer.** Beyond ~2% of members the incentive cost
  exceeds the churn saved; contacting everyone would lose 117M NTD.

**Save-rate sensitivity: the assumption decides whether to run a campaign at all.**

| assumed save rate | break-even churn risk (149 NTD/month) | members contacted | realised net value |
|---|---|---|---|
| 10% | 1.16 (impossible) | 0 | 0: no campaign |
| 20% | 0.58 | 13,865 (1.4%) | +976k NTD |
| 30% | 0.39 | 24,704 (2.5%) | +2.84M NTD |

KKBox contains no campaign data, so the save rate is an assumption, not an estimate. The
first real campaign must keep a **randomised holdout** of positive-EV members (not
contacted) to measure it; until then the money figures are scenarios, while the
*ranking* holds: EV >= risk > random at every save rate (10/20/30%) and contact
fraction (1-30%) tested, 18 of 18 cases.

## Serving

One scoring path, two entry points (`serving/scoring.py`): population ->
`feature_sql` (the training feature code) -> `prepare_features` (the training dtypes) ->
champion model (`models:/churnguard@champion`) -> expected value + contact decision ->
top-3 reason codes. Population and paid-through date come from the same functions the
label spine uses (`due_population_sql`, `paid_through_sql`); the refactor reproduced the
existing spine with 0 differing rows over 2.97M.

**Batch** (`make score`): everyone due next month. April 2017, real data:

| | |
|---|---|
| members scored | 1,012,577 in 19 s (2.9 GB peak) |
| vs Kaggle's April test population | 100% recall, 89.6% precision |
| contact (positive EV) | 13,893 (1.4%), lowest contacted p = 0.579 = the 0.58 break-even |
| expected net value (20% save rate) | +720k NTD |
| top reason among contacted | `last_is_cancel` (84%), `last_is_auto_renew` (16%) |

Output `data/gold/scores/score_month=YYYY-MM/data.parquet`: `ev_rank`, `msno` (for the
CRM), `p_churn`, `monthly_revenue`, `clv`, `ev`, `contact`, `reason_1..3` (+ log-odds
contribution), model name/version/type, `scored_at`, plus a `_manifest.json`.

**API** (`make serve`, OpenAPI at `/docs`):

- `GET /health`: live model name, version, type, run id
- `GET /members/{member_id}/score?as_of=2017-04-01`: builds that member's features from
  silver as of the date and scores them (~90 ms, thanks to member-sorted silver logs)
- `POST /score`: caller-supplied features; the request schema is generated from the
  feature registry, so a missing or mistyped feature is a 422

**Parity is tested, not assumed:** `tests/test_serving.py` scores members through the API
one by one and requires p_churn, EV, contact and reasons to equal the batch rows exactly
(including an early renewer). On real data the API returned p = 0.99458 / EV 130.743 for
April's top-ranked member, identical to the batch file.

Reason codes: TreeSHAP for LightGBM; for logistic regression, coefficient x standardised
value summed back to each input feature (one-hot levels and missing indicators included),
which adds up exactly to the model's logit (tested).

**Docker:** `Dockerfile` runs the API as a non-root user with a healthcheck; data, the
registry DB and artifacts are mounted. MLflow stores artifact locations as absolute host
paths (`file:///.../mlartifacts`), so `make docker-run` mounts them at the same path.
