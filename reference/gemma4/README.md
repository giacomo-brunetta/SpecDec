# Gemma 4 reference inputs and aggregates

The greedy verifier used `google/gemma-4-31B-it`. The following revisions were recorded in the download metadata for every cached file in each checkpoint on SAMI. The [study plan](../../configs/study.example.json) accepts local snapshot directories. Download the indicated revisions and use those directories in the plan for a pinned reproduction.

| Checkpoint | Revision |
|---|---|
| google/gemma-4-31B-it | `842da3794eaa0b77d5f08bae87a17459d91ff475` |
| google/gemma-4-E2B-it | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` |
| google/gemma-4-E4B-it | `ee0ef6023621cff504d758262d4e04895a5af4a2` |
| google/gemma-4-12B-it | `43c9d014b231474fa7212dbb48a762566c2b255d` |

The LMSYS Arrow cache used the revision directory `200748d9d3cddcc9d782887541057aca0b18c5da`. The seed-555 selection with the target tokenizer produced workload SHA-256 `289fbb82da1c52383dcff4817ea15954e4e4e716de55fd63ba40f7b7480ea21a`. Access the dataset and checkpoints under their respective terms.

- `sweep.csv` and `native-sweep.csv` are measured campaigns. Each runtime has its own greedy baseline. `native-strata.csv` separates prompt lengths.
- `trace-depth.csv` uses explicit greedy proposals. The fixed-budget trees, full-depth trees and histograms use the separate top-two-rank capture. Each reports nonempty proposal rounds.
- `distance.csv` and `short-depth.csv` are calibrated timing projections. `oracle-check.csv` is the live validation sample. Wall-clock timings and any projections recalibrated from fresh timing samples may vary between runs.

To check the deterministic transformations on the same saved inputs and calibration, run `python scripts/check_study_reference.py "$OUT"`. The checker compares every historical row and allows extra configurations from the unified sweep.
