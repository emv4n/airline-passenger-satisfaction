"""Dataset download, loading and cleaning of column names.

Run `python -m src.data` once to download the Kaggle CSVs into data/raw/.
"""

import re
import shutil
from pathlib import Path

import pandas as pd

from src.utils import load_config, project_path


def download_dataset(config):
    """Download the Kaggle dataset and copy train.csv and test.csv into the raw data folder.

    The dataset is public, so kagglehub downloads it without Kaggle credentials.
    """
    import kagglehub  # only needed for the download step

    cache_dir = Path(kagglehub.dataset_download(config["dataset"]["kaggle_handle"]))
    raw_dir = project_path(config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in (config["paths"]["train_file"], config["paths"]["test_file"]):
        shutil.copy(cache_dir / name, raw_dir / name)
    return raw_dir


def to_snake_case(column):
    """Turn a raw column name such as 'Departure/Arrival time convenient' into snake_case."""
    return re.sub(r"[^0-9a-z]+", "_", column.strip().lower()).strip("_")


def clean_columns(frame, config):
    """Rename the columns to snake_case and drop the identifier columns when present."""
    frame = frame.rename(columns=to_snake_case)
    return frame.drop(columns=config["dataset"]["drop_columns"], errors="ignore")


def load_split(config, split="train"):
    """Load the train or test CSV with snake_case columns and without identifier columns."""
    file_name = config["paths"][f"{split}_file"]
    return clean_columns(pd.read_csv(project_path(config["paths"]["raw_dir"]) / file_name), config)


def split_features_target(frame, config):
    """Separate the features from the binary target (1 = satisfied, 0 = neutral or dissatisfied)."""
    target = config["dataset"]["target"]
    y = (frame[target] == config["dataset"]["positive_label"]).astype(int)
    return frame.drop(columns=target), y


if __name__ == "__main__":
    print("Dataset copied to", download_dataset(load_config()))
