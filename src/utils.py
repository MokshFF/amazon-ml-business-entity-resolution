"""
Shared Utility Functions Module.

Provides memory-efficient data loading, submission TSV generation,
entity-level F0.5 evaluation, and taxonomy-based error analysis.

50/50 Joint Responsibility: Moksh & Dhwaj [Team Leader].
"""

import logging
import os
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple
import pandas as pd
import yaml


def setup_logger(name: str = "er_pipeline", level: str = "INFO") -> logging.Logger:
    """Configures clean logging format for pipeline execution."""
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
    """Loads configuration YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_source_tsv(
    file_path: str,
    usecols: Optional[List[str]] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Memory-conscious loader for source TSV files.

    Args:
        file_path: Path to TSV file.
        usecols: Selective columns to load to minimize RAM usage.
        nrows: Optional row limit for fast local prototyping.

    Returns:
        pd.DataFrame: Loaded DataFrame with memory-efficient dtypes.
    """
    dtypes = {
        "entity_id": "category",
        "business_name": "string",
        "business_address": "string",
        "country": "category",
    }
    if usecols:
        dtypes = {k: v for k, v in dtypes.items() if k in usecols}

    return pd.read_csv(
        file_path,
        sep="\t",
        usecols=usecols,
        nrows=nrows,
        dtype=dtypes,
        keep_default_na=False,
    )


def load_source_chunks(
    file_path: str,
    chunk_size: int = 50000,
    usecols: Optional[List[str]] = None,
) -> Iterator[pd.DataFrame]:
    """Iterates through large TSV datasets in memory-manageable chunks."""
    dtypes = {
        "entity_id": "string",
        "business_name": "string",
        "business_address": "string",
        "country": "string",
    }
    if usecols:
        dtypes = {k: v for k, v in dtypes.items() if k in usecols}

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        chunksize=chunk_size,
        usecols=usecols,
        dtype=dtypes,
        keep_default_na=False,
    ):
        yield chunk


def load_ground_truth(file_path: str) -> Dict[str, Set[str]]:
    """Loads ground-truth matching TSV file into memory-efficient set mapping."""
    df = pd.read_csv(
        file_path,
        sep="\t",
        dtype={"source1_entity_id": str, "matched_entity_ids": str},
        keep_default_na=False,
    )
    gt: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        matched = str(row["matched_entity_ids"]).strip()
        if matched:
            gt[s1] = set(m.strip() for m in matched.split(",") if m.strip())
        else:
            gt[s1] = set()
    return gt


def save_submission_tsv(
    source1_ids: Sequence[str],
    id_mapping: Mapping[str, Sequence[str]],
    output_path: str,
    id_col_name: str = "matched_entity_ids",
) -> None:
    """Formats and writes submission TSV conforming strictly to competition rules:
    - Exactly one row per Source 1 entity
    - Tab-separated ('\\t')
    - Comma-separated match IDs with zero quoting
    - Empty string for entities with zero matches (singletons)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    rows: List[Tuple[str, str]] = []

    for s1_id in source1_ids:
        raw_ids = id_mapping.get(s1_id, [])
        # Deduplicate while preserving order
        deduped = list(dict.fromkeys(raw_ids))
        rows.append((s1_id, ",".join(deduped)))

    df_out = pd.DataFrame(rows, columns=["source1_entity_id", id_col_name])
    df_out.to_csv(output_path, sep="\t", index=False)


def calculate_f05(precision: float, recall: float, eps: float = 1e-9) -> float:
    """Computes F0.5 score: beta=0.5 weights precision twice as heavily as recall."""
    beta_sq = 0.25
    num = (1 + beta_sq) * precision * recall
    denom = (beta_sq * precision) + recall
    if denom < eps:
        return 0.0
    return num / denom


def evaluate_entity_resolution(
    ground_truth: Mapping[str, Set[str]],
    predictions: Mapping[str, Set[str]],
) -> Dict[str, float]:
    """Entity-level evaluation conforming to the official challenge specification.

    Computes:
        - True Positives (TP): Correctly identified matching pairs
        - False Positives (FP): Non-matching pairs incorrectly predicted as matches (false merges)
        - False Negatives (FN): True matches missed by model
        - Singleton Accuracy: Precision on true singletons (entities with 0 matches)
        - Precision, Recall, and F0.5 score
    """
    tp = 0
    fp = 0
    fn = 0
    correct_singletons = 0
    total_singletons = 0

    all_s1_ids = set(ground_truth.keys()).union(set(predictions.keys()))

    for s1_id in all_s1_ids:
        true_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())

        # Check singleton performance
        if len(true_set) == 0:
            total_singletons += 1
            if len(pred_set) == 0:
                correct_singletons += 1

        tp += len(pred_set.intersection(true_set))
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f05 = calculate_f05(precision, recall)
    singleton_acc = (
        correct_singletons / total_singletons if total_singletons > 0 else 1.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f05": f05,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "singleton_accuracy": singleton_acc,
        "total_singletons": float(total_singletons),
    }


def categorize_error(
    s1_row: pd.Series,
    cand_row: pd.Series,
    is_false_positive: bool,
) -> str:
    """Taxonomy-based error classifier to guide targeted pipeline improvements.

    Categories:
        - 'typo': Minor character differences in names
        - 'abbreviation': Shortened or expanded tokens
        - 'legal_suffix': Inconsistent legal designations
        - 'address_variation': Discrepancies in street, area, or landmark
        - 'missing_address': Incomplete address fields
        - 'transliteration': Phonetic or regional spelling differences
        - 'similar_business_names': Different businesses sharing popular tokens
        - 'false_merge': Non-matching entities merged incorrectly
        - 'missed_match': True pair failed to match
        - 'singleton_error': Singleton incorrectly assigned a match
    """
    # TODO [Moksh & Dhwaj]: Implement rule-based error categorization based on validation outputs
    if is_false_positive:
        return "false_merge"
    return "missed_match"
