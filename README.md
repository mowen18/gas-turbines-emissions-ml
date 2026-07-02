# Gas Turbine CO and NOx Emissions Modeling

This project analyzes the UCI Gas Turbine CO and NOx Emission Data Set with a reproducible machine learning workflow. The notebook predicts carbon monoxide (`CO`) and nitrogen oxides (`NOX`) emissions from gas turbine operating and ambient sensor measurements, then compares model performance using expanding-window rolling-origin validation and an untouched final holdout year.

The finished analysis is in [`notebooks/01_emissions_modeling.ipynb`](notebooks/01_emissions_modeling.ipynb).

## Dataset

Source: [UCI Machine Learning Repository, dataset id 551](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set)

Citation:

> Gas Turbine CO and NOx Emission Data Set [Dataset]. (2019). UCI Machine Learning Repository. https://doi.org/10.24432/C5WC95

The dataset contains hourly aggregated gas turbine sensor measurements from Turkey's north western region. The analysis uses operating and ambient measurements as predictors and treats `CO` and `NOX` as regression targets.

## Problem Framing

The primary task is supervised regression: estimate `CO` and NOx emissions from turbine conditions such as ambient temperature, pressure, humidity, turbine inlet temperature, turbine energy yield, compressor discharge pressure, and related operating variables.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── .gitkeep
├── images/
│   ├── actual_vs_predicted_best_model.png
│   ├── co_distribution.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   └── nox_distribution.png
├── notebooks/
│   └── 01_emissions_modeling.ipynb
├── src/
│   └── emissions_ml/
│       ├── __init__.py
│       ├── data.py
│       └── validation.py
└── tests/
    └── test_validation.py
```

Raw CSV files are intentionally not committed. The notebook fetches the dataset through `ucimlrepo` via `src/emissions_ml/data.py`; if running offline, place an equivalent raw file at `data/raw/gas_turbine_emissions.csv`.

## Modeling Approach

The notebook compares three regression approaches:

- `StandardScaler` + `Ridge(alpha=1.0)` as a stable linear baseline.
- `RandomForestRegressor` as a nonlinear tree-based model.
- `StandardScaler` + RBF `SVR`, wrapped for multi-output regression.

Model comparison uses expanding-window annual validation:

| Fold | Training years | Validation year |
|---:|---:|---:|
| 1 | 2011 | 2012 |
| 2 | 2011-2012 | 2013 |
| 3 | 2011-2013 | 2014 |

Each fold trains only on years that precede the validation year. The 2015 data is excluded from model comparison, then used once as the final holdout after the selected model is refit on all available pre-holdout data from 2011-2014.

Model selection uses mean normalized RMSE across validation years and targets:

```text
normalized RMSE = RMSE / target standard deviation in that validation fold
```

This avoids selecting a model by directly averaging raw `CO` and `NOX` RMSE values, which are on different numerical scales. Raw target-level R2, RMSE, and MAE are still reported for interpretation. RBF SVR keeps the existing deterministic 5,000-row training cap within each fold and within the final 2011-2014 refit if selected.

## Key Results

Best model by mean normalized RMSE across rolling-origin validation folds: **Random Forest**.

Model-selection summary:

| Model | Mean normalized RMSE | Std normalized RMSE | Mean R2 | Mean RMSE | Mean MAE |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.7217 | 0.1631 | 0.4491 | 4.750 | 3.571 |
| RBF SVR | 0.7442 | 0.2175 | 0.4080 | 4.815 | 3.640 |
| Ridge Regression | 0.7769 | 0.0321 | 0.3903 | 5.295 | 3.808 |

Annual mean normalized RMSE across `CO` and `NOX`:

| Validation year | Ridge Regression | Random Forest | RBF SVR |
|---:|---:|---:|---:|
| 2012 | 0.754 | 0.643 | 0.622 |
| 2013 | 0.762 | 0.613 | 0.615 |
| 2014 | 0.814 | 0.909 | 0.995 |

The annual results are intentionally shown because validation performance is not stationary. Random Forest has the best overall mean normalized RMSE, but its average lead over RBF SVR is modest and only three validation years are available. All models degrade on the 2014 validation fold, especially the nonlinear models.

Random Forest target-level rolling-origin summary:

| Target | Mean R2 | Std R2 | Mean RMSE | Std RMSE | Mean MAE | Std MAE | Mean normalized RMSE | Std normalized RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CO | 0.5059 | 0.1263 | 1.590 | 0.242 | 0.872 | 0.116 | 0.6987 | 0.0940 |
| NOx | 0.3924 | 0.4596 | 7.911 | 2.590 | 6.269 | 2.644 | 0.7448 | 0.2819 |

After selection, Random Forest was refit on the full 2011-2014 development period. 2015 final holdout performance:

| Target | R2 | RMSE | MAE |
|---|---:|---:|---:|
| CO | 0.4719 | 1.624 | 1.037 |
| NOx | -0.0212 | 11.249 | 9.244 |

The rolling-origin results suggest useful nonlinear signal, but the 2014 validation degradation and the weak 2015 NOx holdout R2 are evidence of year-to-year distribution shift and limits to what the available sensor columns capture. The 2015 actual-versus-predicted plots also show underprediction of extreme CO values and compression toward the middle for NOx.

## Visuals

### CO Distribution

![CO distribution](images/co_distribution.png)

### NOx Distribution

![NOx distribution](images/nox_distribution.png)

### Correlation Heatmap

![Correlation heatmap](images/correlation_heatmap.png)

### Model Comparison

![Model comparison](images/model_comparison.png)

### Actual vs Predicted

![Actual vs predicted for best model](images/actual_vs_predicted_best_model.png)

## How To Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests
jupyter nbconvert --to notebook --execute notebooks/01_emissions_modeling.ipynb --inplace
```

The notebook should be run from the repository root so imports from `src/` resolve correctly.

## Limitations And Next Steps

- Rolling-origin validation is more reliable than a single 2014 validation split, but it still has only three annual folds.
- Random Forest won by mean normalized RMSE, but the average improvement over RBF SVR was modest and its 2014 validation performance was weaker than its 2012 and 2013 validation performance.
- NOx final holdout performance remained weaker than CO, with slightly negative R2 on 2015.
- The RBF SVR comparison is intentionally capped at 5,000 deterministic training rows per fold for runtime, so it is not a full-data SVR benchmark.
- Feature engineering could explore lagged operating context, operating regimes, interactions, and target transformations.
