from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


PARAM_NAMES = ("ymax", "r_opt_mm", "width_mm")


def build_region_feature_matrix(features: dict[str, np.ndarray]) -> np.ndarray:
    """
    Build X matrix from region-level features.

    Expected keys:
      - aridity_index
      - soil_whc
      - heat_stress
    """
    return np.column_stack(
        [
            np.asarray(features["aridity_index"], dtype=float),
            np.asarray(features["soil_whc"], dtype=float),
            np.asarray(features["heat_stress"], dtype=float),
        ]
    )


def fit_linear_multioutput(X: np.ndarray, Y: np.ndarray) -> LinearRegression:
    """
    Fit a multi-output linear regression model mapping X -> Y.

    X: shape (n_samples, n_features)
    Y: shape (n_samples, n_targets)
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if Y.ndim != 2:
        raise ValueError("Y must be 2D")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows")

    model = LinearRegression()
    model.fit(X, Y)
    return model
