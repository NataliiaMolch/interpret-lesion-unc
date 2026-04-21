"""Lesion novelty features based on PCA and distances to a reference set."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


IDENTIFIERS = {"case_id", "filename", "lesion label", "lesion type"}


def add_novelty_features(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    target_column: str = "ddu",
) -> pd.DataFrame:
    """Add Mahalanobis and nearest-reference distances to ``target``.

    The reference set is typically the training split. Only numeric feature
    columns are used, excluding identifiers and the uncertainty target.
    """

    feature_columns = [
        column
        for column in reference.select_dtypes(include=[np.number]).columns
        if column in target.columns and column not in IDENTIFIERS and column != target_column
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns available for novelty computation.")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_ref = scaler.fit_transform(imputer.fit_transform(reference[feature_columns]))
    x_target = scaler.transform(imputer.transform(target[feature_columns]))

    n_components = min(x_ref.shape[0] - 1, x_ref.shape[1])
    if n_components < 1:
        raise ValueError("At least two reference lesions are needed for PCA novelty features.")
    pca = PCA(n_components=n_components).fit(x_ref)
    z_ref = pca.transform(x_ref)
    z_target = pca.transform(x_target)

    covariance = np.cov(z_ref, rowvar=False)
    covariance = np.atleast_2d(covariance)
    covariance_inv = np.linalg.pinv(covariance)
    mean = z_ref.mean(axis=0)

    result = target.copy()
    mahalanobis = []
    nearest = []
    for row in z_target:
        diff = row - mean
        mahalanobis.append(float(np.sqrt(diff.T @ covariance_inv @ diff)))
        nearest.append(float(np.linalg.norm(z_ref - row.reshape(1, -1), axis=1).min()))
    result["Mahalanobis distance"] = mahalanobis
    result["Smallest distance"] = nearest
    return result
