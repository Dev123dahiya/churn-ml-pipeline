"""Synthetic subscription data with a known generating process, plus input validation."""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC = ["tenure_months", "monthly_charge", "support_tickets_90d", "late_payments_12m", "logins_30d"]
CATEGORICAL = ["contract", "payment_method", "plan"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "churned"
ALLOWED = {"contract": ["monthly", "annual", "two_year"], "payment_method": ["card", "upi", "bank_transfer", "wallet"],
           "plan": ["basic", "standard", "premium"]}
RANGES = {"tenure_months": (0, 240), "monthly_charge": (0, 10_000), "support_tickets_90d": (0, 200),
          "late_payments_12m": (0, 12), "logins_30d": (0, 2_000)}


def generate(n: int = 6000, seed: int = 0, shift: float = 0.0) -> pd.DataFrame:
    """Customers whose churn probability follows a known logistic formula.

    Because the true probabilities are known, the best achievable score is known
    too, so a model's result can be judged against a ceiling instead of in a vacuum.
    ``shift`` > 0 simulates a later period: prices rise and more customers are on monthly contracts.
    """
    rng = np.random.default_rng(seed)
    contract = rng.choice(ALLOWED["contract"], n, p=[0.55 + 0.25 * shift, 0.30 - 0.15 * shift, 0.15 - 0.10 * shift])
    plan = rng.choice(ALLOWED["plan"], n, p=[0.4, 0.4, 0.2])
    payment = rng.choice(ALLOWED["payment_method"], n, p=[0.35, 0.35, 0.2, 0.1])
    tenure = np.clip(rng.gamma(2.0, 12.0, n), 0, 240).round()
    base_price = pd.Series(plan).map({"basic": 299, "standard": 599, "premium": 999}).to_numpy()
    charge = (base_price * (1 + 0.35 * shift) * rng.normal(1.0, 0.08, n)).round(2)
    tickets = rng.poisson(1.2, n)
    late = np.minimum(rng.poisson(0.6, n), 12)
    logins = rng.poisson(np.where(contract == "monthly", 14, 22), n)

    logit = (-2.6 - 0.035 * tenure + 0.0012 * charge + 0.38 * tickets + 0.55 * late - 0.045 * logins
             + np.select([contract == "monthly", contract == "annual"], [1.1, 0.2], -0.6)
             + np.where(payment == "wallet", 0.35, 0.0))
    probability = 1 / (1 + np.exp(-logit))
    frame = pd.DataFrame({"tenure_months": tenure, "monthly_charge": charge, "support_tickets_90d": tickets, "late_payments_12m": late,
                          "logins_30d": logins, "contract": contract, "payment_method": payment, "plan": plan,
                          TARGET: (rng.random(n) < probability).astype(int)})
    frame.attrs["true_probability"] = probability
    return frame


def validate(frame: pd.DataFrame, require_target: bool = False) -> list[str]:
    """Return a list of problems. An empty list means the frame is safe to train on or score."""
    problems = []
    missing = [c for c in FEATURES + ([TARGET] if require_target else []) if c not in frame.columns]
    if missing:
        return [f"missing columns: {missing}"]
    for column, (low, high) in RANGES.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            problems.append(f"{column}: {int(values.isna().sum())} missing or non-numeric values")
        elif ((values < low) | (values > high)).any():
            problems.append(f"{column}: {int(((values < low) | (values > high)).sum())} values outside [{low}, {high}]")
    for column, allowed in ALLOWED.items():
        unknown = sorted(set(frame[column].dropna().astype(str)) - set(allowed))
        if frame[column].isna().any():
            problems.append(f"{column}: {int(frame[column].isna().sum())} missing values")
        if unknown:
            problems.append(f"{column}: unknown categories {unknown}")
    if require_target and not set(frame[TARGET].unique()) <= {0, 1}:
        problems.append("churned: must be 0 or 1")
    return problems
