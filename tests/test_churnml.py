import numpy as np
import pandas as pd
import pytest

from churnml import FEATURES, Registry, best_threshold, drift_report, generate, psi, train, validate
from churnml.train import _pipeline, calibration_error, expected_cost


@pytest.fixture(scope="module")
def data():
    return generate(3000, seed=1)


@pytest.fixture(scope="module")
def result(data):
    return train(data, seed=1)


# ---- data -----------------------------------------------------------------
def test_generator_is_reproducible_and_valid(data):
    pd.testing.assert_frame_equal(data, generate(3000, seed=1))
    assert validate(data, require_target=True) == []
    assert 0.10 < data["churned"].mean() < 0.25
    assert not data.equals(generate(3000, seed=2))


def test_validation_names_each_problem(data):
    bad = data.head(50).copy()
    bad.loc[bad.index[0], "monthly_charge"] = -5
    bad.loc[bad.index[1], "tenure_months"] = np.nan
    bad.loc[bad.index[2], "contract"] = "weekly"
    bad.loc[bad.index[3], "plan"] = None
    problems = validate(bad)
    assert any("monthly_charge: 1 values outside" in p for p in problems)
    assert any("tenure_months: 1 missing" in p for p in problems)
    assert any("unknown categories ['weekly']" in p for p in problems)
    assert any("plan: 1 missing" in p for p in problems)
    assert validate(data.drop(columns=["plan"])) == ["missing columns: ['plan']"]
    with pytest.raises(ValueError, match="failed validation"):
        train(bad.assign(churned=0))


# ---- training -------------------------------------------------------------
def test_model_learns_the_signal_and_approaches_the_known_ceiling(result):
    m = result.metrics
    assert m["roc_auc"] > 0.72
    assert m["roc_auc"] > m["roc_auc_ceiling"] - 0.03       # close to what a perfect model could do on this data
    assert set(result.candidates) == {"logistic_regression", "gradient_boosting"}
    assert result.name == max(result.candidates, key=result.candidates.get)
    assert m["rows_train"] + m["rows_validation"] + m["rows_test"] == 3000


def test_training_is_reproducible(data, result):
    again = train(data, seed=1)
    assert again.metrics == result.metrics and again.threshold == result.threshold


def test_probabilities_are_calibrated(result):
    assert result.metrics["calibration_error"] < 0.05
    perfect = np.array([0.2] * 500 + [0.8] * 500)
    outcomes = np.array([1] * 100 + [0] * 400 + [1] * 400 + [0] * 100)
    assert calibration_error(outcomes, perfect) == pytest.approx(0.0, abs=1e-9)
    assert calibration_error(outcomes, np.full(1000, 0.9)) == pytest.approx(0.4)   # says 90%, reality is 50%


def test_preprocessing_is_fitted_on_training_data_only(data):
    """The scaler's statistics must come from the rows passed to fit, not the whole dataset."""
    from sklearn.linear_model import LogisticRegression
    train_rows = data.iloc[:1000]
    pipe = _pipeline(LogisticRegression(max_iter=500)).fit(train_rows[FEATURES], train_rows["churned"])
    scaler = pipe.named_steps["pre"].named_transformers_["num"]
    assert scaler.mean_[1] == pytest.approx(train_rows["monthly_charge"].mean())
    assert scaler.mean_[1] != pytest.approx(data["monthly_charge"].mean(), abs=1e-6)
    unseen = train_rows.head(3).assign(plan="enterprise")          # a category never seen in training does not crash scoring
    assert pipe.predict_proba(unseen[FEATURES]).shape == (3, 2)


def test_threshold_follows_the_costs():
    rng = np.random.default_rng(0)
    p = rng.random(20_000)
    y = (rng.random(20_000) < p).astype(int)                       # perfectly calibrated scores
    t, cost = best_threshold(y, p, cost_fn=3000, cost_fp=500)
    assert abs(t - 500 / 3500) < 0.05                              # theory: cost_fp / (cost_fp + cost_fn) = 0.143
    assert cost <= expected_cost(y, p, 0.5, 3000, 500)
    assert best_threshold(y, p, cost_fn=500, cost_fp=500)[0] == pytest.approx(0.5, abs=0.05)
    assert best_threshold(y, p, cost_fn=500, cost_fp=3000)[0] > 0.8  # offers are expensive: only contact near-certain churners


def test_chosen_threshold_beats_default_and_doing_nothing(result):
    m = result.metrics
    assert m["cost_per_customer"] < m["cost_at_default_0.5"] < m["cost_if_nobody_contacted"]
    assert m["threshold"] < 0.5 and 0 < m["flagged_share"] < 1


# ---- registry -------------------------------------------------------------
def test_registry_versions_metadata_and_scoring(tmp_path, data, result):
    reg = Registry(str(tmp_path))
    with pytest.raises(RuntimeError, match="no production model"):
        reg.predict(data.head(3))
    v1 = reg.register(result, data, notes="first")
    meta = reg.meta(v1)
    assert v1 == 1 and meta["training_rows"] == 3000 and meta["features"] == FEATURES and len(meta["training_data_fingerprint"]) == 16
    assert reg.register(result, data) == 2 and reg.versions() == [1, 2]
    assert reg.meta(2)["training_data_fingerprint"] == meta["training_data_fingerprint"]       # same data, same fingerprint
    assert reg.register(result, data.head(100)) == 3 and reg.meta(3)["training_data_fingerprint"] != meta["training_data_fingerprint"]

    assert reg.promote(v1) == (True, "promoted v1")
    scores = reg.predict(data.head(20))
    assert list(scores.columns) == ["churn_probability", "contact", "model_version"] and scores["churn_probability"].between(0, 1).all()
    assert (scores["contact"] == (scores["churn_probability"] >= result.threshold)).all() and (scores["model_version"] == 1).all()
    with pytest.raises(ValueError, match="unknown categories"):
        reg.predict(data.head(5).assign(contract="lifetime"))


def test_promotion_gate_and_rollback(tmp_path, data, result):
    reg = Registry(str(tmp_path))
    good = reg.register(result, data)
    reg.promote(good)

    worse = train(data, seed=1)
    worse.metrics = {**result.metrics, "roc_auc": result.metrics["roc_auc"] - 0.05}
    v_worse = reg.register(worse, data)
    promoted, reason = reg.promote(v_worse)
    assert not promoted and "does not beat production v1" in reason and reg.production == good

    uncalibrated = train(data, seed=1)
    uncalibrated.metrics = {**result.metrics, "roc_auc": 0.99, "calibration_error": 0.2}
    promoted, reason = reg.promote(reg.register(uncalibrated, data))
    assert not promoted and "calibration error" in reason

    better = train(data, seed=1)
    better.metrics = {**result.metrics, "roc_auc": result.metrics["roc_auc"] + 0.02}
    v_better = reg.register(better, data)
    assert reg.promote(v_better, min_auc_gain=0.01)[0] and reg.production == v_better
    assert reg.rollback() == good and reg.production == good
    assert reg.rollback() is None


# ---- monitoring -----------------------------------------------------------
def test_psi_is_zero_for_identical_data_and_grows_with_shift():
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0, 1, 5000))
    assert psi(base, base) == pytest.approx(0.0, abs=1e-9)
    assert psi(base, pd.Series(rng.normal(0, 1, 5000))) < 0.02              # same distribution, different sample
    small, large = psi(base, base + 0.3), psi(base, base + 1.5)
    assert 0.02 < small < large and large > 0.25
    cats = pd.Series(["a"] * 700 + ["b"] * 300)
    assert psi(cats, pd.Series(["a"] * 690 + ["b"] * 310)) < 0.01
    assert psi(cats, pd.Series(["a"] * 300 + ["b"] * 600 + ["c"] * 100)) > 0.25   # includes a brand-new category


def test_drift_report_flags_the_features_that_moved(data):
    reference = data[FEATURES]
    assert drift_report(reference, generate(2000, seed=7)[FEATURES])["status"] == "ok"
    report = drift_report(reference, generate(2000, seed=7, shift=1.0)[FEATURES])
    assert report["status"] == "alert" and report["worst_feature"] == "monthly_charge"
    assert report["features"]["contract"]["status"] == "alert"
    assert report["features"]["tenure_months"]["status"] == "ok" and report["features"]["plan"]["status"] == "ok"
