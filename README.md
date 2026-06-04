# BirdCLEF Plus 2026

BirdCLEF Plus 2026 is an acoustic species recognition project for the Kaggle BirdCLEF+ 2026 competition. The system turns long-field audio into multilabel species predictions and packages the full path from training code to a CPU-compatible Kaggle submission notebook.

The project focuses on a practical competition setting: fixed audio windows, mel-spectrogram features, multilabel classification, fold-based checkpointing, ensemble inference, and strict submission-column alignment.

## Highlights

- End-to-end Python pipeline for BirdCLEF+ 2026 audio classification.
- Mel-spectrogram preprocessing with deterministic crop and padding behavior.
- EfficientNet-B0 multilabel baseline with a compact CNN fallback.
- Fold checkpoint export and ensemble manifest support.
- Kaggle CPU notebook generator for offline submission inference.
- Regression tests for audio windows, metadata parsing, macro AUC, and submission shape.

## System Design

| Stage | Implementation |
| --- | --- |
| Data loading | Competition metadata, taxonomy labels, sample submission columns, and soundscape row IDs. |
| Audio preprocessing | 32000 Hz waveform loading, five-second window extraction, and mel-spectrogram conversion. |
| Training | Fold assignment, multilabel targets, PyTorch dataloaders, EfficientNet-style image classifier, and BCE loss. |
| Validation | Macro AUC calculation across label columns. |
| Checkpointing | Per-fold weights plus an ensemble manifest with audio configuration and class order. |
| Inference | Soundscape segmentation, batch prediction, sigmoid probabilities, and column-order validation. |
| Submission packaging | Generated Kaggle notebook that reads competition input and writes `submission.csv`. |

## Inference Pipeline

The inference flow is designed for Kaggle's CPU runtime:

1. Read `sample_submission.csv` to lock row IDs and class column order.
2. Parse each row ID into a soundscape ID and an end timestamp.
3. Load each soundscape once and reuse it across time windows.
4. Extract the target five-second segment for each row.
5. Convert each segment into a normalized mel-spectrogram.
6. Average predictions across ensemble members when a manifest checkpoint is used.
7. Write probabilities in the exact competition submission format.

The generated notebook uses `/kaggle/input/birdclef-2026` for official data and a separate Kaggle Dataset for model weights. Network access and GPU inference are not required.

## Repository Contents

| Path | Contents |
| --- | --- |
| `src/audio.py` | Audio loading, waveform segmentation, and mel-spectrogram features. |
| `src/data.py` | Metadata loading, class extraction, row ID parsing, and fold assignment. |
| `src/train.py` | Baseline fold training and checkpoint export. |
| `src/infer.py` | Submission inference from a single checkpoint or ensemble manifest. |
| `src/make_notebook.py` | Kaggle CPU notebook generation. |
| `tests/` | Focused regression tests for the core competition contract. |
| `notebooks/` | Generated inference notebook artifact. |

## Reproduction

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate kaggle-birdclef-2026
```

Install with pip:

```bash
python -m pip install -r requirements.txt
```

Download competition data through the Kaggle API:

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

Run the test suite:

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
