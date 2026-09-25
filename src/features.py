"""
Feature Engineering Module.

Responsible for computing fine-grained pairwise similarity features between
Source 1 entities and generated candidate entities (from Source 2 and Source 3).

Team Role Assigned: Member 3 (ML matching, Feature engineering, Threshold tuning, F0.5 validation).
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz


def compute_string_similarities(str_a: str, str_b: str, prefix: str = "") -> Dict[str, float]:
    """Computes a suite of string similarity metrics between two strings.

    Args:
        str_a: First string (e.g., from Source 1).
        str_b: Second string (e.g., from candidate).
        prefix: Prefix for feature keys (e.g., 'name_' or 'address_').

    Returns:
        Dict[str, float]: Dictionary of normalized similarity values [0.0, 1.0].
    """
    if not str_a or not str_b:
        return {
            f"{prefix}ratio": 0.0,
            f"{prefix}token_sort_ratio": 0.0,
            f"{prefix}token_set_ratio": 0.0,
            f"{prefix}jaro_winkler": 0.0,
            f"{prefix}len_diff_ratio": 1.0,
        }

    # Normalized Levenshtein ratio
    ratio = fuzz.ratio(str_a, str_b) / 100.0
    # Token sort ratio (handles word order transpositions)
    token_sort = fuzz.token_sort_ratio(str_a, str_b) / 100.0
    # Token set ratio (handles subset/superset tokens)
    token_set = fuzz.token_set_ratio(str_a, str_b) / 100.0
    # Jaro-Winkler similarity (prioritizes prefix matches)
    jaro_winkler = distance.JaroWinkler.similarity(str_a, str_b)

    # Relative length difference
    len_max = max(len(str_a), len(str_b))
    len_diff = abs(len(str_a) - len(str_b)) / len_max if len_max > 0 else 0.0

    return {
        f"{prefix}ratio": float(ratio),
        f"{prefix}token_sort_ratio": float(token_sort),
        f"{prefix}token_set_ratio": float(token_set),
        f"{prefix}jaro_winkler": float(jaro_winkler),
        f"{prefix}len_diff_ratio": float(len_diff),
    }


class PairwiseFeatureExtractor:
    """Extracts similarity feature vectors for candidate pairs."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initializes feature extractor.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}

    def extract_pair_features(
        self, s1_row: pd.Series, cand_row: pd.Series
    ) -> Dict[str, float]:
        """Extracts similarity features between one Source 1 record and one candidate record.

        Args:
            s1_row: Source 1 entity row.
            cand_row: Candidate entity row (Source 2 or Source 3).

        Returns:
            Dict[str, float]: Feature dictionary.
        """
        features: Dict[str, float] = {}

        # Business Name Similarities
        name_s1 = str(s1_row.get("name_clean", s1_row.get("business_name", "")))
        name_cand = str(cand_row.get("name_clean", cand_row.get("business_name", "")))
        features.update(compute_string_similarities(name_s1, name_cand, prefix="name_"))

        # Address Similarities
        addr_s1 = str(s1_row.get("address_clean", s1_row.get("business_address", "")))
        addr_cand = str(cand_row.get("address_clean", cand_row.get("business_address", "")))
        features.update(compute_string_similarities(addr_s1, addr_cand, prefix="addr_"))

        # Country Compatibility
        country_s1 = str(s1_row.get("country", "")).strip().upper()
        country_cand = str(cand_row.get("country", "")).strip().upper()
        features["country_exact_match"] = 1.0 if country_s1 == country_cand else 0.0

        # TODO [Member 3]: Add advanced features:
        #   - TF-IDF Cosine similarity on address tokens & business names
        #   - Substring containment & acronym matches (e.g. IBM vs International Business Machines)
        #   - Numeric token overlap (street numbers, PIN/postal codes)

        return features

    def extract_batch_features(
        self, pairs_df: pd.DataFrame, records_lookup: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """Computes feature matrix for a DataFrame of candidate pairs.

        Args:
            pairs_df: DataFrame containing ['source1_entity_id', 'candidate_entity_id'].
            records_lookup: Lookup map from entity_id to preprocessed Series.

        Returns:
            pd.DataFrame: Feature matrix aligned with candidate pairs.
        """
        # TODO [Member 3]: Optimize batch feature computation (e.g., vectorized or parallelized chunking).
        feature_list: List[Dict[str, float]] = []
        for _, row in pairs_df.iterrows():
            s1 = records_lookup.get(row["source1_entity_id"])
            cand = records_lookup.get(row["candidate_entity_id"])
            if s1 is not None and cand is not None:
                feature_list.append(self.extract_pair_features(s1, cand))
            else:
                feature_list.append({})

        return pd.DataFrame(feature_list)
