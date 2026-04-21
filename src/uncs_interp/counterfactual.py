"""Counterfactual explanations for ElasticNet uncertainty models."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .modeling import load_model


DEFAULT_IMMUTABLE_PATTERNS = (
    r"^case_id$",
    r"^filename$",
    r"^lesion label$",
    r"^lesion type$",
)


@dataclass
class CounterfactualConfig:
    """Counterfactual report configuration."""

    model_path: Path
    features_csv: Path
    output_csv: Path
    target_uncertainty: float = 0.0
    top_k: int = 10
    immutable_patterns: tuple[str, ...] = DEFAULT_IMMUTABLE_PATTERNS


def explain_counterfactuals(config: CounterfactualConfig) -> pd.DataFrame:
    """Explain how features would need to change to reach target uncertainty."""

    artifact = load_model(config.model_path)
    pipeline = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]
    table = pd.read_csv(config.features_csv)
    x = table[feature_columns].copy()
    predictions = pipeline.predict(x)
    coefficient_table = _original_space_coefficients(pipeline, feature_columns)
    mutable = [
        row
        for row in coefficient_table.to_dict("records")
        if not _is_immutable(row["Feature"], config.immutable_patterns) and abs(row["coefficient"]) > 1e-12
    ]

    rows = []
    for index, prediction in enumerate(predictions):
        desired_delta = config.target_uncertainty - prediction
        identifiers = {
            column: table.iloc[index][column]
            for column in ("case_id", "filename", "lesion label")
            if column in table.columns
        }
        for item in mutable:
            feature = item["Feature"]
            coefficient = item["coefficient"]
            current_value = float(x.iloc[index][feature])
            required_change = desired_delta / coefficient
            rows.append(
                {
                    **identifiers,
                    "predicted_uncertainty": float(prediction),
                    "target_uncertainty": config.target_uncertainty,
                    "feature": feature,
                    "current_value": current_value,
                    "required_value": current_value + required_change,
                    "required_change": required_change,
                    "direction": "increase" if required_change > 0 else "decrease",
                    "absolute_change": abs(required_change),
                    "model_coefficient": coefficient,
                    "note": "One-feature counterfactual; changes are descriptive, not clinically actionable.",
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = (
            result.sort_values(["case_id", "lesion label", "absolute_change"])
            .groupby([column for column in ("case_id", "lesion label") if column in result.columns], dropna=False)
            .head(config.top_k)
            .reset_index(drop=True)
        )
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(config.output_csv, index=False)
    return result


def _original_space_coefficients(pipeline, feature_columns: list[str]) -> pd.DataFrame:
    """Map ElasticNet coefficients back to the pre-scaled feature names."""

    after_variance = np.asarray(feature_columns)[pipeline.named_steps["variance"].get_support()]
    selected_mask = pipeline.named_steps["selector"].get_support()
    selected = after_variance[selected_mask]
    scaled_coefficients = pipeline.named_steps["elasticnet"].coef_
    scale_after_variance = pipeline.named_steps["scaler"].scale_[selected_mask]
    original_coefficients = scaled_coefficients / scale_after_variance
    return pd.DataFrame({"Feature": selected, "coefficient": original_coefficients})


def _is_immutable(feature: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, feature) for pattern in patterns)
