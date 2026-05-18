"""Dataset loading helpers for the UCI gas turbine emissions project."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from ucimlrepo import fetch_ucirepo


UCI_DATASET_ID = 551
TARGET_COLUMNS = ("CO", "NOx")


@dataclass(frozen=True)
class EmissionsDataset:
    """Container for features, targets, and the joined modeling table."""

    features: pd.DataFrame
    targets: pd.DataFrame
    frame: pd.DataFrame


def load_emissions_data() -> EmissionsDataset:
    """Fetch the UCI gas turbine emissions dataset."""

    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    frame = pd.concat([features, targets], axis=1)
    return EmissionsDataset(features=features, targets=targets, frame=frame)
