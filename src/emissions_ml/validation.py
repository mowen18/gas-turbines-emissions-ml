"""Chronological validation helpers for annual gas turbine modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass(frozen=True)
class RollingOriginFold:
    """One expanding-window training split with a single validation year."""

    training_years: Tuple[int, ...]
    validation_year: int


def expanding_window_folds(
    *,
    start_year: int = 2011,
    validation_years: Iterable[int] = (2012, 2013, 2014),
    holdout_year: int = 2015,
) -> Tuple[RollingOriginFold, ...]:
    """Return expanding annual folds that never touch the holdout year."""

    validation_years = tuple(validation_years)
    if len(validation_years) != len(set(validation_years)):
        raise ValueError("Validation years must not contain duplicates.")
    if validation_years != tuple(sorted(validation_years)):
        raise ValueError("Validation years must be strictly increasing.")

    folds = []
    for validation_year in validation_years:
        if validation_year >= holdout_year:
            raise ValueError("Validation years must be before the holdout year.")

        training_years = tuple(range(start_year, validation_year))
        if not training_years:
            raise ValueError("Each fold needs at least one historical training year.")

        folds.append(
            RollingOriginFold(
                training_years=training_years,
                validation_year=validation_year,
            )
        )

    return tuple(folds)


def pre_holdout_years(*, start_year: int = 2011, holdout_year: int = 2015) -> Tuple[int, ...]:
    """Return all years available before the final holdout year."""

    if start_year >= holdout_year:
        raise ValueError("The start year must come before the holdout year.")

    return tuple(range(start_year, holdout_year))
