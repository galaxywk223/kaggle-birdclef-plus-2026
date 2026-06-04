# BirdCLEF 2026 Final Window Automation Archive

This archive preserves scripts used during the BirdCLEF 2026 final submission window on 2026-06-03 and 2026-06-04. The scripts are historical competition-operation material and are not part of the reusable baseline training or inference pipeline.

## Contents

| File | Role |
| --- | --- |
| `final_window_submit_20260603.ps1` | Submission queue runner for selected Kaggle code submissions. |
| `submit_code_with_body_20260603.py` | Kaggle API helper for code-submission requests with structured error output. |
| `watch_final_window_scores_20260603.ps1` | Submission score monitor for the final window. |
| `watch_birdclef_reset_queue.ps1` | Earlier reset-window watcher and conditional queue submitter. |
| `keep_awake_until_20260604.ps1` | Windows keep-awake helper for the final submission period. |
| `run_final_window_submit_20260603.cmd` | Windows Task Scheduler wrapper for the final-window submitter. |

## Scope

- The archive records operational context for traceability.
- The main project workflow remains `src/`, `tests/`, and the generated Kaggle CPU notebook.
- The scripts may reference historical Kaggle kernels, timestamps, local logs, and competition-window assumptions.
- The scripts are not required for training, inference, testing, or notebook generation.
