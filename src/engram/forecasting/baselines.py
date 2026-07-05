"""Baseline ladder forecasters for the Phase 0 forecast harness."""

from __future__ import annotations

import warnings
from collections import Counter
from typing import Any

import numpy as np

from engram.models.track_b import DelinquencyBucket

_ALL_BUCKETS = [bucket.value for bucket in DelinquencyBucket]
_BUCKET_INDEX = {bucket: float(index) for index, bucket in enumerate(_ALL_BUCKETS)}
_NUMERIC_FEATURES = ("current_upb", "credit_score", "interest_rate", "months_observed")
_CATEGORICAL_FEATURES = ("bucket", "prev_bucket", "state")


class _TabularFeatureMixin:
    """Encode canonical forecast row feature dictionaries into numeric vectors."""

    def _fit_encoder(self, rows: list[dict[str, Any]]) -> None:
        categories: dict[str, list[str]] = {}
        for name in _CATEGORICAL_FEATURES:
            values = {
                str(row.get("features", {}).get(name))
                for row in rows
                if row.get("features", {}).get(name) is not None
            }
            categories[name] = sorted(values)
        self._categories = categories

    def _matrix(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray([self._vector(row.get("features", {})) for row in rows], dtype=float)

    def _vector(self, features: dict[str, Any]) -> list[float]:
        values: list[float] = []
        for name in _NUMERIC_FEATURES:
            raw = features.get(name)
            value = float(raw) if isinstance(raw, int | float) else 0.0
            if name == "current_upb":
                value /= 100_000.0
            elif name == "credit_score":
                value /= 850.0
            elif name == "interest_rate":
                value /= 100.0
            values.append(value)

        bucket = features.get("bucket")
        values.append(_BUCKET_INDEX.get(str(bucket), -1.0))

        categories = getattr(self, "_categories", {})
        for name in _CATEGORICAL_FEATURES:
            raw = features.get(name)
            encoded = str(raw) if raw is not None else ""
            known = categories.get(name, [])
            values.extend(1.0 if encoded == value else 0.0 for value in known)
        return values

    def _label_vector(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray([row["label"]["next_bucket"] for row in rows], dtype=str)

    @staticmethod
    def _normalize_probabilities(raw: dict[str, float]) -> dict[str, float]:
        probabilities = {bucket: max(float(raw.get(bucket, 0.0)), 0.0) for bucket in _ALL_BUCKETS}
        total = sum(probabilities.values())
        if total <= 0.0:
            return {bucket: 1.0 / len(_ALL_BUCKETS) for bucket in _ALL_BUCKETS}
        return {bucket: probability / total for bucket, probability in probabilities.items()}

    def _fallback_distribution(self) -> dict[str, float]:
        counts = getattr(self, "_label_counts", Counter())
        total = sum(counts.values())
        if total <= 0:
            return {bucket: 1.0 / len(_ALL_BUCKETS) for bucket in _ALL_BUCKETS}
        return self._normalize_probabilities({bucket: counts.get(bucket, 0) / total for bucket in _ALL_BUCKETS})


class HazardForecaster(_TabularFeatureMixin):
    """Discrete-time multinomial logistic hazard forecaster."""

    name = "hazard_logistic"

    def __init__(self, *, random_state: int = 42, max_iter: int = 2000) -> None:
        self.random_state = random_state
        self.max_iter = max_iter
        self._model: Any | None = None
        self._class_labels: list[str] = []
        self._label_counts: Counter[str] = Counter()
        self._categories: dict[str, list[str]] = {}

    def fit(self, train_rows: list[dict[str, Any]]) -> None:
        """Fit a multinomial logistic model over canonical forecast rows."""
        self._fit_encoder(train_rows)
        labels = self._label_vector(train_rows)
        self._label_counts = Counter(str(label) for label in labels)
        if len(self._label_counts) < 2:
            self._model = None
            return

        x_train = self._matrix(train_rows)
        self._class_labels = sorted(self._label_counts)
        label_to_index = {label: index for index, label in enumerate(self._class_labels)}
        y_train = np.asarray([label_to_index[str(label)] for label in labels], dtype=int)
        from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

        self._model = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight="balanced",
        )
        self._model.fit(x_train, y_train)

    def predict_proba(self, features: dict[str, Any]) -> dict[str, float]:
        """Return full delinquency-bucket probabilities."""
        if self._model is None:
            return self._fallback_distribution()
        x_value = np.asarray([self._vector(features)], dtype=float)
        probabilities = self._model.predict_proba(x_value)[0]
        raw = {
            self._class_labels[int(label_index)]: float(probability)
            for label_index, probability in zip(
                self._model.classes_,
                probabilities,
                strict=True,
            )
        }
        return self._normalize_probabilities(raw)


class GBMForecaster(_TabularFeatureMixin):
    """LightGBM multiclass baseline forecaster."""

    name = "gbm_lightgbm"

    def __init__(
        self,
        *,
        random_state: int = 42,
        n_estimators: int = 25,
        learning_rate: float = 0.15,
    ) -> None:
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self._model: Any | None = None
        self._class_labels: list[str] = []
        self._label_counts: Counter[str] = Counter()
        self._categories: dict[str, list[str]] = {}

    def fit(self, train_rows: list[dict[str, Any]]) -> None:
        """Fit a deterministic LightGBM multiclass model."""
        self._fit_encoder(train_rows)
        labels = self._label_vector(train_rows)
        self._label_counts = Counter(str(label) for label in labels)
        if len(self._label_counts) < 2:
            self._model = None
            return

        x_train = self._matrix(train_rows)
        self._class_labels = sorted(self._label_counts)
        label_to_index = {label: index for index, label in enumerate(self._class_labels)}
        y_train = np.asarray([label_to_index[str(label)] for label in labels], dtype=int)
        from lightgbm import LGBMClassifier

        self._model = LGBMClassifier(
            objective="multiclass",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=7,
            min_data_in_leaf=1,
            min_data_in_bin=1,
            deterministic=True,
            force_col_wise=True,
            random_state=self.random_state,
            verbosity=-1,
        )
        self._model.fit(x_train, y_train)

    def predict_proba(self, features: dict[str, Any]) -> dict[str, float]:
        """Return full delinquency-bucket probabilities."""
        if self._model is None:
            return self._fallback_distribution()
        x_value = np.asarray([self._vector(features)], dtype=float)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            probabilities = np.asarray(self._model.predict_proba(x_value)[0], dtype=float)
        raw = {
            self._class_labels[int(label_index)]: float(probability)
            for label_index, probability in zip(
                self._model.classes_,
                probabilities,
                strict=True,
            )
        }
        return self._normalize_probabilities(raw)
