"""Demo: python -m churnml   (synthetic data; trains, registers, promotes, scores and checks for drift)"""
from __future__ import annotations

import tempfile

from . import Registry, drift_report, generate, train


def main() -> None:
    registry = Registry(tempfile.mkdtemp())
    data = generate(6000, seed=0)
    result = train(data, seed=0)
    m = result.metrics
    print(f"Data: {len(data):,} synthetic customers, churn rate {m['churn_rate']:.1%}")
    print(f"Cross-validated ROC AUC by model family: " + ", ".join(f"{k} {v:.4f}" for k, v in result.candidates.items()))
    print(f"Selected: {result.name}\n")
    print("Held-out test set:")
    print(f"  ROC AUC            {m['roc_auc']:.4f}   (a model that knew the true probabilities scores {m['roc_auc_ceiling']:.4f})")
    print(f"  PR AUC             {m['pr_auc']:.4f}")
    print(f"  Brier score        {m['brier']:.4f}")
    print(f"  Calibration error  {m['calibration_error']:.4f}")
    print(f"\nThreshold chosen on validation data to minimise cost (missed churner = {m['cost_false_negative']:.0f}, wasted offer = {m['cost_false_positive']:.0f}):")
    print(f"  threshold {m['threshold']:.2f}: contact {m['flagged_share']:.1%} of customers, precision {m['precision']:.2f}, recall {m['recall']:.2f}")
    print(f"  cost per customer: {m['cost_per_customer']:.0f} at the chosen threshold, {m['cost_at_default_0.5']:.0f} at the default 0.5, "
          f"{m['cost_if_nobody_contacted']:.0f} if nobody is contacted")

    v1 = registry.register(result, data, notes="first model")
    print(f"\nRegistry: registered v{v1}; promote -> {registry.promote(v1)}")
    weak = train(generate(300, seed=5), seed=5)
    v2 = registry.register(weak, generate(300, seed=5), notes="trained on only 300 rows")
    print(f"Registry: registered v{v2} (trained on 300 rows); promote -> {registry.promote(v2)}")
    print(f"Production is v{registry.production}")

    scores = registry.predict(generate(5, seed=42))
    print("\nScoring 5 new customers:")
    print(scores.round(3).to_string(index=False))

    print("\nDrift check against the training reference:")
    for label, frame in [("same distribution", generate(2000, seed=9)), ("after a price rise and a shift to monthly contracts", generate(2000, seed=9, shift=1.0))]:
        report = drift_report(registry.reference(v1), frame)
        flagged = {c: f["psi"] for c, f in report["features"].items() if f["status"] != "ok"}
        print(f"  {label}: {report['status']}  {flagged if flagged else ''}")


if __name__ == "__main__":
    main()
