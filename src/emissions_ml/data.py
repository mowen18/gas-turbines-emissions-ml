"""Dataset loading helpers for the UCI gas turbine emissions project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from ucimlrepo import fetch_ucirepo


UCI_DATASET_ID = 551
TARGET_COLUMNS = ("CO", "NOX")


@dataclass(frozen=True)
class EmissionsDataset:
    """Container for features, targets, and the joined modeling table."""

    features: pd.DataFrame
    targets: pd.DataFrame
    frame: pd.DataFrame


def load_emissions_data(local_path: Optional[Union[str, Path]] = None) -> EmissionsDataset:
    """Fetch the UCI gas turbine emissions dataset or load a local fallback.

    UCI currently returns this dataset with every column in ``features`` and
    ``targets`` set to ``None``. This helper normalizes that shape into explicit
    feature and target tables.
    """

    if local_path is None:
        dataset = fetch_ucirepo(id=UCI_DATASET_ID)
        frame = _extract_frame(dataset)
    else:
        frame = pd.read_csv(local_path)

    missing_targets = [column for column in TARGET_COLUMNS if column not in frame.columns]
    if missing_targets:
        missing = ", ".join(missing_targets)
        raise ValueError(f"Missing expected target column(s): {missing}")

    targets = frame.loc[:, TARGET_COLUMNS].copy()
    features = frame.drop(columns=list(TARGET_COLUMNS)).copy()
    return EmissionsDataset(features=features, targets=targets, frame=frame)


def _extract_frame(dataset) -> pd.DataFrame:
    """Return a complete dataframe from a ucimlrepo dataset object."""

    if dataset.data.original is not None:
        return dataset.data.original.copy()

    features = dataset.data.features
    targets = dataset.data.targets
    if targets is None:
        return features.copy()

    return pd.concat([features, targets], axis=1)
