"""Has the data the model sees in production drifted from what it was trained on?"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CATEGORICAL, NUMERIC


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population stability index between two samples of one feature.

    Rule of thumb: below 0.1 stable, 0.1 to 0.25 worth a look, above 0.25 the
    distribution has shifted enough that the model's numbers should not be trusted
    without re-checking.
    """
    if pd.api.types.is_numeric_dtype(expected):
        edges = np.unique(np.quantile(expected.dropna(), np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            edges = np.array([expected.min() - 1, expected.mean(), expected.max() + 1])
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
    else:
        categories = sorted(set(expected.astype(str)) | set(actual.astype(str)))
        e = expected.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
        a = actual.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)    # avoid log(0) for empty bins
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, warn: float = 0.1, alert: float = 0.25) -> dict:
    features = {}
    for column in NUMERIC + CATEGORICAL:
        value = psi(reference[column], current[column])
        features[column] = {"psi": round(value, 4), "status": "alert" if value >= alert else "warn" if value >= warn else "ok"}
    worst = max(features, key=lambda c: features[c]["psi"])
    return {"features": features, "worst_feature": worst,
            "status": "alert" if any(f["status"] == "alert" for f in features.values())
            else "warn" if any(f["status"] == "warn" for f in features.values()) else "ok"}
