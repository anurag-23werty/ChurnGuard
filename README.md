# ChurnGuard

End-to-end churn prediction and retention platform on the KKBox music-subscription dataset
(~400M listening-log rows): out-of-core data engineering, point-in-time features,
a calibrated and explainable model, value-based retention targeting, serving, and monitoring.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.

## Status

- [x] Phase 0: architecture
- [ ] Phase 1: ingestion (raw → bronze Parquet)
- [ ] Phase 2: cleaning (silver)
- [ ] Phase 3: label spine
- [ ] Phase 4: point-in-time features
- [ ] Phase 5: baselines
- [ ] Phase 6: LightGBM + calibration + SHAP
- [ ] Phase 7: retention decisioning
- [ ] Phase 8: serving (batch + FastAPI)
- [ ] Phase 9: monitoring
- [ ] Phase 10: UI & CI
