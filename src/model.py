"""
Machine Learning Matching Model Module.

Responsible for pairwise match classification, threshold tuning optimized for F0.5,
and candidate ranking.

Team Role Assigned: Member 3 (ML matching, Feature engineering, Threshold tuning, F0.5 validation).
"""

import pickle
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.utils import calculate_f05, setup_logger

logger = setup_logger("model")


class EntityMatchingModel:
    """Wrapper class for pairwise entity matching classification."""

    def __init__(self, model_type: str = "logistic_regression", threshold: float = 0.5):
        """Initializes the entity matching model.

        Args:
            model_type: Identifier of the classifier architecture to use.
            threshold: Probability threshold for classifying a pair as a match.
        """
        self.model_type = model_type
        self.threshold = threshold
        self.classifier = None
        self._initialize_classifier()

    def _initialize_classifier(self) -> None:
        """Instantiates the underlying classifier based on model_type."""
        # TODO [Member 3]: Explore and compare classifier choices:
        #   - LogisticRegression (interpretable baseline)
        #   - RandomForestClassifier
        #   - LightGBM / XGBoost / CatBoost
        if self.model_type == "logistic_regression":
            self.classifier = LogisticRegression(class_weight="balanced", max_iter=1000)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def train(self, X_train: pd.DataFrame, y_train: np.ndarray) -> None:
        """Trains the pairwise classification model.

        Args:
            X_train: Feature matrix of candidate pairs.
            y_train: Binary labels (1 for true match, 0 for non-match).
        """
        # TODO [Member 3]: Implement training routine, feature scaling if needed, and logging.
        logger.info(f"Training {self.model_type} on {len(X_train)} candidate pairs...")
        self.classifier.fit(X_train, y_train)
        logger.info("Training complete.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts match probability for candidate pairs.

        Args:
            X: Feature matrix of candidate pairs.

        Returns:
            np.ndarray: Array of match probabilities [0.0, 1.0].
        """
        if self.classifier is None:
            raise RuntimeError("Model has not been trained or loaded yet.")
        # Return probability of positive class (label 1)
        return self.classifier.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Predicts binary match decisions using the decision threshold.

        Args:
            X: Feature matrix of candidate pairs.
            threshold: Custom threshold (defaults to self.threshold).

        Returns:
            np.ndarray: Binary array (1 for predicted match, 0 otherwise).
        """
        thresh = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

    def tune_threshold(
        self,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        threshold_candidates: Optional[np.ndarray] = None,
    ) -> float:
        """Finds the decision threshold maximizing the F0.5 score on validation data.

        Because F0.5 penalizes false positives twice as heavily as false negatives
        (beta=0.5), optimal thresholds are typically higher than 0.5.

        Args:
            X_val: Validation feature matrix.
            y_val: Validation true binary labels.
            threshold_candidates: Array of candidate thresholds to test.

        Returns:
            float: Optimal threshold maximizing F0.5.
        """
        # TODO [Member 3]: Sweep thresholds to maximize precision-weighted F0.5 score.
        if threshold_candidates is None:
            threshold_candidates = np.linspace(0.1, 0.95, 86)

        best_threshold = 0.5
        best_f05 = -1.0

        probs = self.predict_proba(X_val)

        for thresh in threshold_candidates:
            preds = (probs >= thresh).astype(int)
            tp = np.sum((preds == 1) & (y_val == 1))
            fp = np.sum((preds == 1) & (y_val == 0))
            fn = np.sum((preds == 0) & (y_val == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f05 = calculate_f05(precision, recall)

            if f05 > best_f05:
                best_f05 = f05
                best_threshold = float(thresh)

        logger.info(f"Optimal threshold found: {best_threshold:.4f} with F0.5: {best_f05:.4f}")
        self.threshold = best_threshold
        return best_threshold

    def save(self, filepath: str) -> None:
        """Serializes model artifact and state to disk."""
        with open(filepath, "wb") as f:
            pickle.dump({"classifier": self.classifier, "threshold": self.threshold}, f)
        logger.info(f"Saved model checkpoint to {filepath}")

    def load(self, filepath: str) -> None:
        """Loads serialized model artifact from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.classifier = data["classifier"]
        self.threshold = data["threshold"]
        logger.info(f"Loaded model from {filepath} (threshold={self.threshold})")
