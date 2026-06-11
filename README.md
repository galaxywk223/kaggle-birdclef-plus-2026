# BirdCLEF Plus 2026

Bioacoustic multilabel species recognition pipeline for the Kaggle BirdCLEF+ 2026 competition. The project converts long-field audio into species probabilities and includes training, checkpoint export, ensemble inference, and CPU-compatible Kaggle notebook generation.

## Results

| Metric | Recorded value | Evidence |
| --- | ---: | --- |
| Public score | `0.950` | Local Kaggle submission snapshot |
| Private score | `0.941` | Recorded final competition result |
| Public leaderboard | `464 / 4244` | Downloaded public leaderboard evidence |

The public leaderboard rank is based on downloaded public leaderboard evidence. A final/private leaderboard rank is not claimed by this repository.

## Modeling Approach

- Five-second audio windows from long soundscapes.
- Mel-spectrogram preprocessing at 32000 Hz with deterministic crop and padding behavior.
- Multilabel species targets aligned to `sample_submission.csv`.
- EfficientNet-B0 baseline with a compact CNN fallback.
- Fold checkpoints and ensemble manifest inference.
- CPU Kaggle notebook generation for offline `submission.csv` creation.

## System Design

| Stage | Implementation |
| --- | --- |
| Data loading | Competition metadata, taxonomy labels, sample submission columns, and soundscape row IDs. |
| Audio preprocessing | Waveform loading, five-second segmentation, and mel-spectrogram conversion. |
| Training | Fold assignment, multilabel targets, PyTorch dataloaders, EfficientNet-style classifier, and BCE loss. |
| Validation | Macro AUC calculation across label columns. |
| Checkpointing | Per-fold weights plus an ensemble manifest with audio configuration and class order. |
| Inference | Soundscape segmentation, batch prediction, sigmoid probabilities, and column-order validation. |
| Submission packaging | Kaggle CPU notebook generation that reads competition input and writes `submission.csv`. |

## Repository Contents

| Path | Contents |
| --- | --- |
| `src/audio.py` | Audio loading, waveform segmentation, and mel-spectrogram features. |
| `src/data.py` | Metadata loading, class extraction, row ID parsing, and fold assignment. |
| `src/train.py` | Fold training and checkpoint export. |
| `src/infer.py` | Submission inference from a single checkpoint or ensemble manifest. |
| `src/make_notebook.py` | Kaggle CPU notebook generation. |
| `tests/` | Regression tests for core competition contracts. |
| `notebooks/` | Generated CPU inference notebook. |

## Commands

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate kaggle-birdclef-2026
```

Install with pip:

```bash
python -m pip install -r requirements.txt
```

Download competition data:

```bash
python -m src.download_data --competition birdclef-2026
```

Generate an exploratory data summary:

```bash
python -m src.eda --data-dir data/raw
```

Run a debug training job:

```bash
python -m src.train --debug --epochs 1 --limit 256
```

Train the baseline model:

```bash
python -m src.train --model efficientnet_b0 --folds 5 --epochs 10
```

Create a submission file:

```bash
python -m src.infer --checkpoint models/baseline.pt --output submissions/submission.csv
```

Generate the Kaggle CPU notebook:

```bash
python -m src.make_notebook --checkpoint models/baseline.pt
```

Run tests:

```bash
python -m pytest
```

## Baseline Configuration

| Setting | Value |
| --- | --- |
| Sample rate | 32000 Hz |
| Window length | 5 seconds |
| Mel bins | 128 |
| FFT size | 2048 |
| Hop length | 512 |
| Model | EfficientNet-B0 multilabel classifier |
| Loss | `BCEWithLogitsLoss` |
| Folds | 5 |
| Output | Sigmoid probabilities aligned to `sample_submission.csv` |

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE) for details.
