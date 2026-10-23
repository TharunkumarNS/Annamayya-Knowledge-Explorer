# Retrieval evaluation report

> Exploratory in-sample smoke test only. This is not an independent accuracy or generalization benchmark.

Generated: `2026-10-23T08:23:47.270642+00:00`
Cases: **13**; distinct compositions: **13**; held-out: **0**; all cases are tuning/in-sample.

## Comparison

| Method | Eligible | Succeeded | Failed | Top-1 | Recall@3 | Recall@5 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Metadata Baseline | 13 | 13 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| Text Only | 13 | 0 | 13 | 0.000 | 0.000 | 0.000 | 0.000 |
| Audio Only | 13 | 13 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| Multimodal | 13 | 0 | 13 | 0.000 | 0.000 | 0.000 | 0.000 |

Metrics use every eligible case as the denominator. Failed queries and empty result lists contribute zero. Recall@k is 1 when any expected composition occurs in the top k; MRR is the reciprocal rank of the first relevant result.

## Data and method limits

The query audio files are exact recordings already present in the index. Correct direct self-matches verify decoding, feature extraction, ranking, and composition mapping, but do not test alternate performances. Ground truth is the workbook's recording-to-composition association, not independently checked. The same small package supplies both queries and indexed content; no held-out set exists.

Text engine initialization: failed: `Text engine initialization explicitly skipped by operator.`
Audio index: 13 indexed, 1 missing, 0 decode/feature failures.

Metadata baseline performs strict case-insensitive substring matching on canonical title, singer, raga, and tala only; tie-breaks use number of matching fields, total matched text length, then composition ID. It uses no embeddings, audio, graph, or fusion.

Per-query detail: [`../data/evaluation/evaluation_results.csv`](../data/evaluation/evaluation_results.csv). Machine-readable summary: [`../data/evaluation/evaluation_summary.json`](../data/evaluation/evaluation_summary.json).

## Per-method failures
- **Metadata Baseline**: 0 failures/exclusions.
- **Text Only**: 13 failures/exclusions.
  - `direct-R001`: Text engine initialization failed: Text engine initialization explicitly skipped by operator.
  - `direct-R002`: Text engine initialization failed: Text engine initialization explicitly skipped by operator.
  - `direct-R003`: Text engine initialization failed: Text engine initialization explicitly skipped by operator.
  - `direct-R004`: Text engine initialization failed: Text engine initialization explicitly skipped by operator.
  - `direct-R005`: Text engine initialization failed: Text engine initialization explicitly skipped by operator.
  - `direct-R006`: Text engine initialization failed: Text engine initialization explicitly skipped by operator.
  - `direct-R007`: Text engine initialization failed: Text engine initialization explicitly skipped by operator.
  - `direct-R008`: Text engine initialization failed: Text engine initialization explicitly skipped by operator.
- **Audio Only**: 0 failures/exclusions.
- **Multimodal**: 13 failures/exclusions.
  - `direct-R001`: Text engine initialization failed: Text engine initialization explicitly skipped by operator.
  - `direct-R002`: Text engine initialization failed: Text engine initialization explicitly skipped by operator.
  - `direct-R003`: Text engine initialization failed: Text engine initialization explicitly skipped by operator.
  - `direct-R004`: Text engine initialization failed: Text engine initialization explicitly skipped by operator.
  - `direct-R005`: Text engine initialization failed: Text engine initialization explicitly skipped by operator.
  - `direct-R006`: Text engine initialization failed: Text engine initialization explicitly skipped by operator.
  - `direct-R007`: Text engine initialization failed: Text engine initialization explicitly skipped by operator.
  - `direct-R008`: Text engine initialization failed: Text engine initialization explicitly skipped by operator.

## Incorrect top-ranked cases

- **Metadata Baseline**: 13 successful case(s) did not rank a relevant composition first.
  - `direct-R001` expected `ANN-001`, top result `no result`.
  - `direct-R002` expected `ANN-003`, top result `ANN-002`.
  - `direct-R003` expected `ANN-008`, top result `no result`.
  - `direct-R004` expected `ANN-009`, top result `no result`.
  - `direct-R005` expected `ANN-010`, top result `no result`.
  - `direct-R006` expected `ANN-012`, top result `no result`.
  - `direct-R007` expected `ANN-013`, top result `no result`.
  - `direct-R008` expected `ANN-015`, top result `no result`.
- **Text Only**: 0 successful case(s) did not rank a relevant composition first.
- **Audio Only**: 0 successful case(s) did not rank a relevant composition first.
- **Multimodal**: 0 successful case(s) did not rank a relevant composition first.
