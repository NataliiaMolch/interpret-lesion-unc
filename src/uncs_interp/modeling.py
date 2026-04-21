"""ElasticNet uncertainty interpretability model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


IDENTIFIER_COLUMNS = {"case_id", "filename", "lesion label", "lesion type"}
QUALITY_COLUMNS = {"IoU", "maxIoU", "IoUadj", "IoUadj_from_uncertainty"}


@dataclass
class TrainConfig:
    """ElasticNet training configuration."""

    train_csv: Path
    model_out: Path
    target_column: str = "ddu"
    include_iou_adj: bool = False
    group_by_case: bool = True
    weighted: bool = True
    n_jobs: int = 1
    cv_splits: int = 5
    feature_percentiles: tuple[int, ...] = (5, 10, 15, 30, 50, 100)
    alphas: tuple[float, ...] = tuple(np.logspace(-4, 1, 7))
    l1_ratios: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0)


def train_elasticnet(config: TrainConfig) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Train and persist an ElasticNet model for lesion uncertainty."""

    table = pd.read_csv(config.train_csv)
    x, y, feature_columns = prepare_xy(
        table,
        target_column=config.target_column,
        include_iou_adj=config.include_iou_adj,
    )
    groups = table["case_id"].astype(str).to_numpy() if config.group_by_case and "case_id" in table else None
    sample_weight = _sample_weight(y) if config.weighted else None

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold(threshold=1e-6)),
            ("scaler", StandardScaler()),
            ("selector", SelectPercentile(f_regression)),
            ("elasticnet", ElasticNet(random_state=0, max_iter=200000, tol=1e-4)),
        ]
    )
    param_grid = {
        "selector__percentile": list(config.feature_percentiles),
        "elasticnet__alpha": list(config.alphas),
        "elasticnet__l1_ratio": list(config.l1_ratios),
    }
    cv = _cv(groups, config.cv_splits)
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring={"R2": "r2", "neg_MSE": "neg_mean_squared_error", "neg_MAE": "neg_mean_absolute_error"},
        refit="R2",
        cv=cv,
        n_jobs=config.n_jobs,
        error_score="raise",
    )
    fit_kwargs = {"elasticnet__sample_weight": sample_weight} if sample_weight is not None else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        grid.fit(x, y, groups=groups, **fit_kwargs)

    pipeline = grid.best_estimator_
    predictions = pipeline.predict(x)
    metrics = pd.DataFrame(
        [
            {
                "CV R2 mean": grid.cv_results_["mean_test_R2"][grid.best_index_],
                "CV R2 std": grid.cv_results_["std_test_R2"][grid.best_index_],
                "Train R2": r2_score(y, predictions),
                "Train MSE": mean_squared_error(y, predictions),
                "Train MAE": mean_absolute_error(y, predictions),
                "best_params": grid.best_params_,
            }
        ]
    )
    importances = feature_importance_table(pipeline, feature_columns)

    artifact = {
        "pipeline": pipeline,
        "feature_columns": feature_columns,
        "target_column": config.target_column,
        "include_iou_adj": config.include_iou_adj,
        "train_csv": str(config.train_csv),
    }
    config.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, config.model_out)
    importances.to_csv(config.model_out.with_suffix(".feature_importance.csv"), index=False)
    metrics.to_csv(config.model_out.with_suffix(".metrics.csv"), index=False)
    return artifact, importances, metrics


def prepare_xy(
    table: pd.DataFrame,
    target_column: str = "ddu",
    include_iou_adj: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Prepare numeric feature matrix and target vector."""

    if target_column not in table.columns:
        raise ValueError(f"Target column '{target_column}' is missing from the table.")
    exclude = set(IDENTIFIER_COLUMNS) | {target_column}
    if not include_iou_adj:
        exclude |= QUALITY_COLUMNS
    feature_columns = [
        column
        for column in table.select_dtypes(include=[np.number]).columns
        if column not in exclude
    ]
    if not feature_columns:
        raise ValueError("No numeric features remain after filtering.")
    valid = table[target_column].notna()
    return table.loc[valid, feature_columns], table.loc[valid, target_column].to_numpy(dtype=float), feature_columns


def feature_importance_table(pipeline: Pipeline, feature_columns: list[str]) -> pd.DataFrame:
    """Return selected ElasticNet coefficients with original feature names."""

    after_variance = np.asarray(feature_columns)[pipeline.named_steps["variance"].get_support()]
    selected = after_variance[pipeline.named_steps["selector"].get_support()]
    coefficients = pipeline.named_steps["elasticnet"].coef_
    return pd.DataFrame({"Feature": selected, "Importance": coefficients}).sort_values(
        by="Importance",
        key=lambda values: values.abs(),
        ascending=False,
    )


def load_model(path: Path) -> dict:
    """Load a persisted model artifact."""

    return joblib.load(path)


def predict_uncertainty(model_path: Path, features_csv: Path, output_csv: Path) -> pd.DataFrame:
    """Predict uncertainty values for a feature table."""

    artifact = load_model(model_path)
    table = pd.read_csv(features_csv)
    x = table[artifact["feature_columns"]]
    result = table[[column for column in ("case_id", "filename", "lesion label") if column in table.columns]].copy()
    result["predicted_uncertainty"] = artifact["pipeline"].predict(x)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def _sample_weight(y: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(y, ord=2)
    if norm == 0:
        return np.ones_like(y)
    weights = y / norm
    if np.any(weights < 0):
        raise ValueError("Weighted training requires non-negative uncertainty values.")
    return weights


def _cv(groups: np.ndarray | None, n_splits: int):
    if groups is None:
        return KFold(n_splits=n_splits, shuffle=True, random_state=0)
    if len(np.unique(groups)) < n_splits:
        n_splits = len(np.unique(groups))
    if n_splits < 2:
        raise ValueError("At least two groups/cases are needed for cross-validation.")
    return GroupKFold(n_splits=n_splits)
