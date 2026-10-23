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
- [ ] Phase 5: baselines
- [ ] Phase 6: LightGBM + calibration + SHAP
- [ ] Phase 7: retention decisioning
- [ ] Phase 8: serving (batch + FastAPI)
- [ ] Phase 9: monitoring
- [ ] Phase 10: UI & CI

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
