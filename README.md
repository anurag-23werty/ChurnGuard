# ChurnGuard

End-to-end churn prediction and retention platform on the KKBox music-subscription dataset
(~400M listening-log rows): out-of-core data engineering, point-in-time features,
a calibrated and explainable model, value-based retention targeting, serving, and monitoring.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.

## Status

- [x] Phase 0: architecture
- [x] Phase 1: ingestion (raw → bronze Parquet)
- [x] Phase 2: cleaning (silver)
- [ ] Phase 3: label spine
- [ ] Phase 4: point-in-time features
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
