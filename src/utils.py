"""
Utility functions for Amazon ML Challenge 2026: Business Entity Resolution.

Provides shared helper functions for configuration parsing, TSV data loading,
output formatting conforming to submission specs, logging, and evaluation metrics (F0.5).
"""

import logging
import os
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple
import pandas as pd
import yaml


def setup_logger(name: str = "entity_resolution", level: str = "INFO") -> logging.Logger:
    """Configures and returns a standard logger with a clean format.

    Args:
        name: Name of the logger instance.
        level: Logging level (e.g., 'INFO', 'DEBUG', 'WARNING').

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Loads YAML configuration file into a dictionary.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration key-value mappings.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If parsing fails.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_source_tsv(file_path: str) -> pd.DataFrame:
    """Reads a source TSV file using explicit tab separation.

    Expected columns in source files:
        - entity_id: Unique record ID (prefixed S1-, S2-, S3-)
        - business_name: Business name string
        - business_address: Business address string
        - country: Country identifier (e.g., US, India, France)

    Args:
        file_path: Path to the TSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame with consistent string types and NaN fill.
    """
    df = pd.read_csv(
        file_path,
        sep="\t",
        dtype={
            "entity_id": str,
            "business_name": str,
            "business_address": str,
            "country": str,
        },
        keep_default_na=False,
    )
    return df


def load_ground_truth(file_path: str) -> Dict[str, Set[str]]:
    """Loads training ground-truth matching TSV file.

    Format expected:
        source1_entity_id	matched_entity_ids
        S1-00001	S2-00047,S3-00812
        S1-00002

    Args:
        file_path: Path to train_ground_truth.tsv.

    Returns:
        Dict[str, Set[str]]: Mapping from source1_entity_id to a set of matched IDs.
    """
    df = pd.read_csv(
        file_path,
        sep="\t",
        dtype={"source1_entity_id": str, "matched_entity_ids": str},
        keep_default_na=False,
    )
    ground_truth = {}
    for _, row in df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        matched_str = str(row["matched_entity_ids"]).strip()
        if matched_str:
            ground_truth[s1_id] = set(
                m.strip() for m in matched_str.split(",") if m.strip()
            )
        else:
            ground_truth[s1_id] = set()
    return ground_truth


def save_submission_tsv(
    source1_ids: Sequence[str],
    id_mapping: Mapping[str, Sequence[str]],
    output_path: str,
    id_col_name: str = "matched_entity_ids",
) -> None:
    """Writes a submission or candidate TSV file strictly conforming to challenge rules.

    Rules enforced:
        - Exactly one row per Source 1 entity.
        - Tab-separated ('\\t').
        - Comma-separated target IDs with no extra spaces or quotation marks.
        - Empty string for entities with zero matches / candidates.
        - No duplicate IDs within any single row.

    Args:
        source1_ids: Ordered sequence of all Source 1 entity IDs that must be included.
        id_mapping: Mapping from Source 1 ID to list/set of matched or candidate IDs.
        output_path: Target path for the output TSV.
        id_col_name: Column name for target IDs ('matched_entity_ids' or 'candidate_entity_ids').
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    rows: List[Tuple[str, str]] = []

    for s1_id in source1_ids:
        raw_targets = id_mapping.get(s1_id, [])
        # Deduplicate while preserving order
        deduped_targets = list(dict.fromkeys(raw_targets))
        joined_targets = ",".join(deduped_targets)
        rows.append((s1_id, joined_targets))

    df_out = pd.DataFrame(rows, columns=["source1_entity_id", id_col_name])
    df_out.to_csv(output_path, sep="\t", index=False)


def calculate_f05(precision: float, recall: float, eps: float = 1e-9) -> float:
    """Computes the precision-weighted F0.5 score:
    
    Formula:
        F_beta = (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)
        With beta = 0.5:
        F_0.5 = (1.25 * precision * recall) / (0.25 * precision + recall)

    Args:
        precision: Macro or micro precision score in [0.0, 1.0].
        recall: Macro or micro recall score in [0.0, 1.0].
        eps: Small epsilon to prevent division by zero.

    Returns:
        float: Computed F0.5 score.
    """
    beta_sq = 0.5**2  # 0.25
    numerator = (1 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall
    if denominator < eps:
        return 0.0
    return numerator / denominator


def evaluate_entity_resolution(
    ground_truth: Mapping[str, Set[str]],
    predictions: Mapping[str, Set[str]],
) -> Dict[str, float]:
    """Evaluates ER predictions against ground truth computing Pairwise Precision, Recall, and F0.5.

    Args:
        ground_truth: Mapping from source1_id to set of true matching IDs.
        predictions: Mapping from source1_id to set of predicted matching IDs.

    Returns:
        Dict[str, float]: Evaluation metrics: precision, recall, f05, tp, fp, fn.
    """
    tp = 0
    fp = 0
    fn = 0

    all_keys = set(ground_truth.keys()).union(set(predictions.keys()))
    for s1_id in all_keys:
        true_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())

        tp += len(pred_set.intersection(true_set))
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f05 = calculate_f05(precision, recall)

    return {
        "precision": precision,
        "recall": recall,
        "f05": f05,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }
