# Day03 BirdCLEF Plus 2026

BirdCLEF Plus 2026 is a Kaggle bioacoustic multilabel recognition project. The repository packages a reproducible local training workflow, a CPU-compatible Kaggle inference notebook, and regression tests for the core data and submission logic.

The project keeps competition data, model weights, generated submissions, downloaded public kernels, and local run outputs outside Git. The tracked surface is intended for public review and portfolio presentation.

## Project Structure

| Path | Purpose |
| --- | --- |
| `src/` | Audio loading, feature extraction, metadata handling, training, inference, and notebook generation modules. |
| `tests/` | Regression tests for audio segmentation, data parsing, metric calculation, and submission formatting. |
| `notebooks/` | Generated Kaggle CPU inference notebook. |
| `data/raw/` | Local Kaggle competition data directory. Raw files are ignored except for the directory README. |
| `models/` | Local model checkpoints and Kaggle Dataset upload payloads. Model artifacts are ignored. |
| `submissions/` | Local `submission.csv` outputs. Submission files are ignored. |
| `logs/` | Local training, inference, and automation logs. Log files are ignored. |
| `archive/` | Historical competition-operation notes or scripts that are not part of the main reusable pipeline. |

## Environment

Conda environment:

```bash
conda env create -f environment.yml
conda activate kaggle-birdclef-2026
```

Pip environment:

```bash
python -m pip install -r requirements.txt
```

The declared environment includes Python data tooling, PyTorch, torchaudio, timm, librosa, onnxruntime, the Kaggle API client, and pytest.

## Data Preparation

Kaggle API download:

```bash
python -m src.download_data --competition birdclef-2026
```

Manual data preparation requires the official competition files to be extracted under `data/raw/`. The required local files are:

- `train.csv`
- `taxonomy.csv`
- `sample_submission.csv`
- `recording_location.txt`
- `train_audio/`

The optional `test_soundscapes/` directory supports local inference smoke tests. Kaggle submissions use the hidden test soundscapes provided by the competition runtime.

## Workflow

Generate an exploratory data summary:

```bash
python -m src.eda --data-dir data/raw
```

Run a small debug training job:

```bash
python -m src.train --debug --epochs 1 --limit 256
```

Run the baseline training workflow:

```bash
python -m src.train --model efficientnet_b0 --folds 5 --epochs 10
```

Generate a local submission file:

```bash
python -m src.infer --checkpoint models/baseline.pt --output submissions/submission.csv
```

Generate the Kaggle CPU inference notebook:

```bash
python -m src.make_notebook --checkpoint models/baseline.pt
```

The generated notebook reads official data from `/kaggle/input/birdclef-2026`, loads weights from a separate Kaggle Dataset, and writes `/kaggle/working/submission.csv`. Network access and GPU inference are not required by the generated submission notebook.

## Baseline Configuration

| Setting | Default |
| --- | --- |
| Sample rate | 32000 Hz |
| Clip duration | 5 seconds |
| Mel bins | 128 |
| FFT size | 2048 |
| Hop length | 512 |
| Model | EfficientNet-B0 multilabel head |
| Loss | `BCEWithLogitsLoss` |
| Fold count | 5 |
| Inference output | Sigmoid probabilities in `sample_submission.csv` column order |

The training command writes fold checkpoints, an ensemble manifest at `models/baseline.pt`, and summary metrics at `models/baseline_metrics.json`. The `model-dataset-metadata.json` file provides a default Kaggle Dataset metadata template for model-weight uploads.

## Testing

Run the regression suite:

```bash
python -m pytest
```

The tests cover audio crop/pad behavior, metadata parsing, metric calculation, and submission shape/order guarantees.

## Git Policy

The repository intentionally excludes:

- Official Kaggle competition data
- Model checkpoints and intermediate training artifacts
- Generated submissions
- Logs and temporary outputs
- Downloaded or replayed public Kaggle kernels
- Large generated binary files

This policy keeps the public repository lightweight while preserving the reproducible source workflow.

## Chinese Documentation

Chinese documentation is available in [`README.zh-CN.md`](README.zh-CN.md).

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE) for details.
