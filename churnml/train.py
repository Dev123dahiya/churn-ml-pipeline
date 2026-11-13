from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, FEATURES, NUMERIC, TARGET, validate


@dataclass
class TrainResult:
    model: Pipeline
    name: str
    metrics: dict
    threshold: float
    candidates: dict = field(default_factory=dict)   # cross-validated AUC of every model family tried


def _pipeline(estimator) -> Pipeline:
    """Preprocessing lives inside the pipeline, so it is fitted on training folds only.

    Scaling or encoding the full dataset before splitting leaks test-set
    statistics into training. Keeping every step in one object also means the
    exact same transformations run at prediction time.
    """
    pre = ColumnTransformer([("num", StandardScaler(), NUMERIC),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return Pipeline([("pre", pre), ("model", estimator)])


def expected_cost(y_true: np.ndarray, probability: np.ndarray, threshold: float, cost_fn: float, cost_fp: float) -> float:
    """Average cost per customer of acting on predictions at this threshold."""
    flagged = probability >= threshold
    missed = ((y_true == 1) & ~flagged).sum()        # churners we did nothing about
    wasted = ((y_true == 0) & flagged).sum()         # retention offers sent to people who were staying
    return float((missed * cost_fn + wasted * cost_fp) / len(y_true))


def best_threshold(y_true, probability, cost_fn: float = 3000.0, cost_fp: float = 500.0) -> tuple[float, float]:
    """Pick the cut-off that minimises business cost, not the default 0.5.

    Losing a customer costs far more than an unnecessary discount, so the best
    threshold is well below 0.5. With calibrated probabilities the theoretical
    optimum is cost_fp / (cost_fp + cost_fn); the search confirms it on data.
    """
    y_true, probability = np.asarray(y_true), np.asarray(probability)
    grid = np.round(np.arange(0.02, 0.99, 0.01), 2)
    costs = [expected_cost(y_true, probability, t, cost_fn, cost_fp) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), costs[i]


def calibration_error(y_true, probability, bins: int = 10) -> float:
    """Expected calibration error: when the model says 30%, does about 30% churn?"""
    y_true, probability = np.asarray(y_true), np.asarray(probability)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for b in range(bins):
        mask = which == b
        if mask.any():
            error += mask.mean() * abs(probability[mask].mean() - y_true[mask].mean())
    return float(error)


def evaluate(y_true, probability, threshold: float, cost_fn: float, cost_fp: float) -> dict:
    y_true, probability = np.asarray(y_true), np.asarray(probability)
    flagged = probability >= threshold
    tp = int(((y_true == 1) & flagged).sum())
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "calibration_error": calibration_error(y_true, probability),
        "threshold": threshold,
        "precision": tp / max(int(flagged.sum()), 1),
        "recall": tp / max(int((y_true == 1).sum()), 1),
        "flagged_share": float(flagged.mean()),
        "cost_per_customer": expected_cost(y_true, probability, threshold, cost_fn, cost_fp),
        "cost_if_nobody_contacted": expected_cost(y_true, probability, 1.01, cost_fn, cost_fp),
        "cost_at_default_0.5": expected_cost(y_true, probability, 0.5, cost_fn, cost_fp),
    }


def train(frame: pd.DataFrame, seed: int = 0, cost_fn: float = 3000.0, cost_fp: float = 500.0) -> TrainResult:
    problems = validate(frame, require_target=True)
    if problems:
        raise ValueError("training data failed validation: " + "; ".join(problems))
    X, y = frame[FEATURES], frame[TARGET]
    # Three-way split: fit on train, choose the threshold on validation, report on a test set touched once.
    X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_rest, y_rest, test_size=0.25, stratify=y_rest, random_state=seed)

    families = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "gradient_boosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=150, random_state=seed),
    }
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = {name: float(cross_val_score(_pipeline(est), X_train, y_train, cv=folds, scoring="roc_auc").mean()) for name, est in families.items()}
    name = max(scores, key=scores.get)

    # Calibrate so the output can be read as a probability, which the cost-based threshold depends on.
    model = CalibratedClassifierCV(_pipeline(families[name]), method="isotonic", cv=5)
    model.fit(X_train, y_train)
    threshold, _ = best_threshold(y_val, model.predict_proba(X_val)[:, 1], cost_fn, cost_fp)
    metrics = evaluate(y_test, model.predict_proba(X_test)[:, 1], threshold, cost_fn, cost_fp)
    metrics.update({"rows_train": len(X_train), "rows_validation": len(X_val), "rows_test": len(X_test), "churn_rate": float(y.mean()),
                    "cost_false_negative": cost_fn, "cost_false_positive": cost_fp})
    if "true_probability" in frame.attrs:   # only synthetic data knows this: the score of a perfect model
        truth = np.asarray(frame.attrs["true_probability"])[X_test.index]
        metrics["roc_auc_ceiling"] = float(roc_auc_score(y_test, truth))
    return TrainResult(model, name, metrics, threshold, scores)
