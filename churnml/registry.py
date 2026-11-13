"""A file-based model registry: immutable versions, metadata, and a gate on promotion."""
from __future__ import annotations

import hashlib
import json
import os
import time

import joblib
import pandas as pd

from .data import FEATURES, validate


class Registry:
    def __init__(self, root: str) -> None:
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, version: int, name: str) -> str:
        return os.path.join(self.root, f"v{version:04d}", name)

    def versions(self) -> list[int]:
        return sorted(int(d[1:]) for d in os.listdir(self.root) if d.startswith("v") and d[1:].isdigit())

    def register(self, result, training_frame: pd.DataFrame, notes: str = "") -> int:
        """Save a model with everything needed to explain or reproduce it. Versions are never overwritten."""
        version = (self.versions() or [0])[-1] + 1
        os.makedirs(os.path.dirname(self._path(version, "x")))
        joblib.dump(result.model, self._path(version, "model.joblib"))
        fingerprint = hashlib.sha256(pd.util.hash_pandas_object(training_frame[FEATURES], index=False).values.tobytes()).hexdigest()[:16]
        meta = {"version": version, "model_family": result.name, "threshold": result.threshold, "metrics": result.metrics,
                "cv_roc_auc": result.candidates, "features": FEATURES, "training_rows": len(training_frame),
                "training_data_fingerprint": fingerprint, "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "notes": notes}
        with open(self._path(version, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        # Reference rows for drift monitoring: what "normal" inputs looked like at training time.
        training_frame[FEATURES].sample(min(len(training_frame), 2000), random_state=0).to_csv(self._path(version, "reference.csv"), index=False)
        return version

    def meta(self, version: int) -> dict:
        with open(self._path(version, "meta.json")) as f:
            return json.load(f)

    def reference(self, version: int) -> pd.DataFrame:
        return pd.read_csv(self._path(version, "reference.csv"))

    @property
    def production(self) -> int | None:
        path = os.path.join(self.root, "production.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)["version"]

    def promote(self, version: int, min_auc_gain: float = 0.0, max_calibration_error: float = 0.05) -> tuple[bool, str]:
        """Make a version the production model, if it earns it.

        A candidate must be well calibrated and at least as good as the current
        production model on ROC AUC. Returns (promoted, reason) so the decision
        is recorded either way.
        """
        candidate = self.meta(version)["metrics"]
        if candidate["calibration_error"] > max_calibration_error:
            return False, f"calibration error {candidate['calibration_error']:.3f} exceeds {max_calibration_error}"
        current = self.production
        if current is not None:
            baseline = self.meta(current)["metrics"]["roc_auc"]
            if candidate["roc_auc"] < baseline + min_auc_gain:
                return False, f"ROC AUC {candidate['roc_auc']:.4f} does not beat production v{current} ({baseline:.4f}) by {min_auc_gain}"
        tmp = os.path.join(self.root, "production.json.tmp")
        with open(tmp, "w") as f:
            json.dump({"version": version, "previous": current}, f)
        os.replace(tmp, os.path.join(self.root, "production.json"))   # atomic: readers never see a half-written pointer
        return True, f"promoted v{version}" + (f" (was v{current})" if current else "")

    def rollback(self) -> int | None:
        path = os.path.join(self.root, "production.json")
        with open(path) as f:
            previous = json.load(f)["previous"]
        if previous is None:
            return None
        with open(path, "w") as f:
            json.dump({"version": previous, "previous": None}, f)
        return previous

    def predict(self, frame: pd.DataFrame, version: int | None = None) -> pd.DataFrame:
        """Score customers with the production model (or a named version). Bad input is rejected, not guessed at."""
        version = version or self.production
        if version is None:
            raise RuntimeError("no production model; register and promote one first")
        problems = validate(frame)
        if problems:
            raise ValueError("input failed validation: " + "; ".join(problems))
        model = joblib.load(self._path(version, "model.joblib"))
        probability = model.predict_proba(frame[FEATURES])[:, 1]
        threshold = self.meta(version)["threshold"]
        return pd.DataFrame({"churn_probability": probability, "contact": probability >= threshold, "model_version": version}, index=frame.index)
