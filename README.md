# Gas Turbine Emissions Analytics

This project is a cleanup of an older school notebook into a reproducible machine learning portfolio project. The analysis uses the UCI Machine Learning Repository's Gas Turbine CO and NOx Emission Data Set to model flue gas emissions from gas turbine sensor measurements.

## Dataset

Source: [UCI Machine Learning Repository, dataset id 551](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set)

The dataset contains hourly aggregated sensor measurements from a gas turbine in Turkey's north western region. UCI describes the prediction targets as carbon monoxide (`CO`) and nitrogen oxides (`NOx`), with gas turbine operating variables and ambient measurements available as predictors.

The original notebook reads a local `gas_emissions.csv` file. The cleaned project will replace that assumption with a reproducible loader using the `ucimlrepo` package:

```python
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=551)
X = dataset.data.features
y = dataset.data.targets
```

## Objective

Build a clear, reproducible analysis that predicts turbine emissions from operating and ambient conditions. The cleaned version should support:

- Regression models for `CO` and `NOx`.
- A transparent train, validation, and test split.
- Standardized preprocessing with scikit-learn pipelines.
- Model comparison using metrics such as R2 and RMSE.
- Optional classification experiments for thresholded emissions, clearly separated from the primary regression task.

## Current Notebook Summary

`EmissionsAnalytics.ipynb` currently has 165 cells and is organized as a homework assignment:

- Project description and a feature table image reference.
- Exploratory data analysis for missing values, correlations, scatter plots, and histograms.
- Regression setup predicting `CO` while dropping both `CO` and `NOX` from predictors.
- Train, validation, and test splits followed by standardization.
- Regression models: Linear Regression, Random Forest, SVR with linear, polynomial, and RBF kernels, Lasso, and Ridge.
- Hyperparameter tuning with `GridSearchCV` for Random Forest and RBF SVR.
- PCA transformation and comparisons against the original standardized feature set.
- Binary classification by thresholding `CO > 3`, followed by SVC and Gaussian Naive Bayes experiments.
- Final test-set comparison tables for selected regressors and classifiers.

## Notebook Cleanup Plan

The first major notebook-edit pass should:

1. Move the cleaned working notebook to `notebooks/`.
2. Replace assignment prompts, point values, `YOUR CODE` markers, and personal-course framing with portfolio-style section titles.
3. Replace `pd.read_csv("gas_emissions.csv")` with the reproducible UCI loader.
4. Decide whether the primary target is `CO`, `NOx`, or both. A portfolio-ready version should model both targets unless there is a clear reason to narrow scope.
5. Convert repeated metric helpers into reusable functions in `src/emissions_ml/`.
6. Use scikit-learn pipelines so preprocessing is fitted only on training data inside the modeling workflow.
7. Separate regression from threshold-based classification so the project narrative stays coherent.
8. Regenerate plots and remove stale embedded notebook outputs before final publication.

## Proposed Repository Structure

```text
.
├── README.md
├── requirements.txt
├── EmissionsAnalytics.ipynb
├── data/
│   └── .gitkeep
├── images/
│   └── .gitkeep
├── notebooks/
│   └── archive/
│       └── EmissionsAnalytics_original.ipynb
└── src/
    └── emissions_ml/
        ├── __init__.py
        └── data.py
```

## Reproducibility Plan

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Fetch dataset id 551 directly from UCI with `ucimlrepo`.
4. Keep raw and generated data out of Git unless a small derived artifact is intentionally documented.
5. Run notebooks from the repository root so imports from `src/` work consistently.
6. Record model metrics and random seeds in the notebook.

## Citation

Gas Turbine CO and NOx Emission Data Set [Dataset]. (2019). UCI Machine Learning Repository. https://doi.org/10.24432/C5WC95

The dataset is listed by UCI under a Creative Commons Attribution 4.0 International license.
