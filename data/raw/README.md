# BirdCLEF+ 2026 Data Files

This directory contains local Kaggle competition data used by training, inference, and smoke-test commands.

Required files:

- `train.csv`
- `taxonomy.csv`
- `sample_submission.csv`
- `recording_location.txt`
- `train_audio/`

Optional files:

- `test_soundscapes/`

Kaggle code submissions provide hidden test soundscapes in the competition runtime. Local smoke tests can use a small copied or linked subset under `test_soundscapes/`.
