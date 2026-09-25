"""
Machine Learning Matching Model Module.

Responsible for pairwise match classification, negative sampling from candidates,
and threshold optimization for the precision-dominant F0.5 metric.

50/50 Technical Ownership:
- Lead: Moksh (Model Experiment A: Baseline Logistic Regression & Threshold Experiment A)
- Lead: Dhwaj [Team Leader] (Model Experiment B: Tabular Classifiers & Threshold Experiment B)

MODEL LICENSING & RESTRICTIONS:
The challenge prohibits using oversized LLMs or models violating licensing and parameter limits.
Focus on strong tabular classifiers and data-driven thresholding.
"""

import pickle
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.utils import calculate_f05, setup_logger

logger = setup_logger("model")


def create_training_pairs(
    ground_truth: Dict[str, Set[str]],
    candidates_map: Dict[str, List[str]],
    negatives_per_positive: int = 5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Constructs balanced training pairs from ground truth and generated candidates.

    Positive pairs:
        Source 1 entity matched to its true ground truth Source 2 / Source 3 entity.

    Negative pairs:
        Carefully sampled from candidate pairs (hard negatives produced by blocking)
        rather than randomly generating millions of trivial negatives across all sources.

    Avoids train/val leakage and combinatorial explosion.

    Args:
        ground_truth: Map of source1_id -> set of true matching IDs.
        candidates_map: Map of source1_id -> list of candidate IDs from blocking.
        negatives_per_positive: Ratio of negative candidate pairs sampled per positive match.
        random_seed: Seed for reproducible negative sampling.

    Returns:
        pd.DataFrame: DataFrame with columns ['source1_entity_id', 'candidate_entity_id', 'label'].
    """
    rng = np.random.default_rng(random_seed)
    pairs: List[Tuple[str, str, int]] = []

    for s1_id, candidates in candidates_map.items():
        true_matches = ground_truth.get(s1_id, set())

        # 1. Add all positive matches present in candidate set
        for cand_id in candidates:
            if cand_id in true_matches:
                pairs.append((s1_id, cand_id, 1))

        # 2. Sample hard negatives from candidate pool (candidates not in true matches)
        neg_candidates = [c for c in candidates if c not in true_matches]
        if neg_candidates:
            # If entity is singleton (0 true matches), sample 1-2 negative candidates
            num_samples = (
                len(true_matches) * negatives_per_positive
                if len(true_matches) > 0
                else 2
            )
            sampled_negatives = rng.choice(
                neg_candidates,
                size=min(num_samples, len(neg_candidates)),
                replace=False,
            )
            for cand_id in sampled_negatives:
                pairs.append((s1_id, cand_id, 0))

    df_pairs = pd.DataFrame(
        pairs, columns=["source1_entity_id", "candidate_entity_id", "label"]
    )
    return df_pairs


class EntityMatchingModel:
    """Pairwise classification model with F0.5-aware threshold tuning."""

    def __init__(self, model_type: str = "logistic_regression", threshold: float = 0.5):
        self.model_type = model_type
        self.threshold = threshold
        self.classifier = None
        self._init_model()

    def _init_model(self) -> None:
        """Instantiates the underlying classifier."""
        # Experiment Track A (Moksh): Interpretable baseline
        if self.model_type == "logistic_regression":
            self.classifier = LogisticRegression(class_weight="balanced", max_iter=1000)
        # Experiment Track B (Dhwaj): Tabular gradient boosting / tree ensembles
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def train(self, X_train: pd.DataFrame, y_train: np.ndarray) -> None:
        """Trains the pairwise classification model."""
        logger.info(f"Training {self.model_type} on {len(X_train)} labeled pairs...")
        self.classifier.fit(X_train, y_train)
        logger.info("Training complete.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Computes match probabilities."""
        if self.classifier is None:
            raise RuntimeError("Model is not initialized or trained.")
        return self.classifier.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Makes binary decisions using the decision threshold."""
        thresh = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

    def tune_threshold(
        self,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        candidate_thresholds: Optional[List[float]] = None,
    ) -> float:
        """Sweeps decision thresholds to maximize the precision-heavy F0.5 score.

        Sweeps through candidate thresholds (e.g. 0.50, 0.55, 0.60, ..., 0.95).
        F0.5 places twice as much weight on precision as recall.
        """
        if candidate_thresholds is None:
            candidate_thresholds = [
                0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
            ]

        probs = self.predict_proba(X_val)
        best_threshold = 0.5
        best_f05 = -1.0

        for thresh in candidate_thresholds:
            preds = (probs >= thresh).astype(int)
            tp = int(np.sum((preds == 1) & (y_val == 1)))
            fp = int(np.sum((preds == 1) & (y_val == 0)))
            fn = int(np.sum((preds == 0) & (y_val == 1)))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f05 = calculate_f05(prec, rec)

            logger.info(f"Threshold: {thresh:.2f} -> Prec: {prec:.4f}, Rec: {rec:.4f}, F0.5: {f05:.4f}")

            if f05 > best_f05:
                best_f05 = f05
                best_threshold = thresh

        logger.info(f"Best validation threshold: {best_threshold:.2f} (F0.5 = {best_f05:.4f})")
        self.threshold = best_threshold
        return best_threshold

    def save(self, filepath: str) -> None:
        """Serializes model artifact and state."""
        with open(filepath, "wb") as f:
            pickle.dump({"classifier": self.classifier, "threshold": self.threshold}, f)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Loads serialized model artifact."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.classifier = data["classifier"]
        self.threshold = data["threshold"]
        logger.info(f"Model loaded from {filepath}")
