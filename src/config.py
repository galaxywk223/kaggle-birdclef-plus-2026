from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
LOGS_DIR = PROJECT_ROOT / "logs"

KAGGLE_COMPETITION_SLUG = "birdclef-2026"

TRAIN_CSV_PATH = RAW_DATA_DIR / "train.csv"
TAXONOMY_CSV_PATH = RAW_DATA_DIR / "taxonomy.csv"
SAMPLE_SUBMISSION_PATH = RAW_DATA_DIR / "sample_submission.csv"
RECORDING_LOCATION_PATH = RAW_DATA_DIR / "recording_location.txt"
TRAIN_AUDIO_DIR = RAW_DATA_DIR / "train_audio"
TEST_SOUNDSCAPES_DIR = RAW_DATA_DIR / "test_soundscapes"

ROW_ID_COL = "row_id"
DEFAULT_MODEL_NAME = "efficientnet_b0"
DEFAULT_CHECKPOINT_NAME = "baseline.pt"
DEFAULT_SAMPLE_RATE = 32_000
DEFAULT_DURATION_SECONDS = 5.0
DEFAULT_N_MELS = 128
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_SECONDARY_LABEL_WEIGHT = 0.2
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_EPOCHS = 10
DEFAULT_FOLDS = 5
DEFAULT_SEED = 42
CPU_THREAD_COUNT = int(os.getenv("CPU_THREAD_COUNT", max(1, os.cpu_count() or 1)))


def ensure_project_dirs() -> None:
    for path in (RAW_DATA_DIR, MODELS_DIR, SUBMISSIONS_DIR, NOTEBOOKS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)

