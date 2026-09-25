"""
Feature Engineering Module.

Responsible for computing similarity features between Source 1 records and candidate records.

50/50 Technical Ownership:
- Lead: Moksh (Name similarity features: Levenshtein, Jaccard, token overlap, TF-IDF cosine, character n-grams)
- Lead: Dhwaj [Team Leader] (Address & country features: Address similarity, postal match, country match, missing-value indicators)

MEMORY EFFICIENCY:
- Use sparse matrices for TF-IDF vectors where applicable.
- Extract features in batches rather than loading full cross-product matrices into RAM.
- Do not blindly implement every feature: validate each feature's contribution via feature ablation.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz


def extract_name_features(name_a: str, name_b: str) -> Dict[str, float]:
    """Computes name-based similarity features.

    Lead: Moksh
    Features explored:
        - Levenshtein ratio
        - Token sort ratio & Token set ratio (word transpositions & subsets)
        - Jaro-Winkler similarity (prefix sensitivity)
        - Character n-gram overlap
        - Name length ratio & difference
    """
    if not name_a or not name_b:
        return {
            "name_ratio": 0.0,
            "name_token_sort_ratio": 0.0,
            "name_token_set_ratio": 0.0,
            "name_jaro_winkler": 0.0,
            "name_len_diff_ratio": 1.0,
            "name_token_jaccard": 0.0,
        }

    tokens_a = set(name_a.split())
    tokens_b = set(name_b.split())
    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)
    jaccard = len(intersection) / len(union) if union else 0.0

    len_max = max(len(name_a), len(name_b))
    len_diff = abs(len(name_a) - len(name_b)) / len_max if len_max > 0 else 0.0

    return {
        "name_ratio": fuzz.ratio(name_a, name_b) / 100.0,
        "name_token_sort_ratio": fuzz.token_sort_ratio(name_a, name_b) / 100.0,
        "name_token_set_ratio": fuzz.token_set_ratio(name_a, name_b) / 100.0,
        "name_jaro_winkler": float(distance.JaroWinkler.similarity(name_a, name_b)),
        "name_len_diff_ratio": float(len_diff),
        "name_token_jaccard": float(jaccard),
    }


def extract_address_and_context_features(
    addr_a: str, addr_b: str, country_a: str, country_b: str, cand_id: str
) -> Dict[str, float]:
    """Computes address similarity, country compatibility, and context features.

    Lead: Dhwaj [Team Leader]
    Features explored:
        - Address token set ratio & Levenshtein
        - Token Jaccard overlap on address components
        - Address length difference
        - Country exact match indicator (1.0 = match, 0.0 = mismatch)
        - Source indicator (S2 vs S3 candidate)
        - Missing address indicator
    """
    is_addr_missing = 1.0 if (not addr_a or not addr_b) else 0.0

    if not addr_a or not addr_b:
        addr_features = {
            "addr_ratio": 0.0,
            "addr_token_set_ratio": 0.0,
            "addr_token_jaccard": 0.0,
            "addr_len_diff_ratio": 1.0,
        }
    else:
        tokens_a = set(addr_a.split())
        tokens_b = set(addr_b.split())
        intersection = tokens_a.intersection(tokens_b)
        union = tokens_a.union(tokens_b)
        jaccard = len(intersection) / len(union) if union else 0.0
        len_max = max(len(addr_a), len(addr_b))
        len_diff = abs(len(addr_a) - len(addr_b)) / len_max if len_max > 0 else 0.0

        addr_features = {
            "addr_ratio": fuzz.ratio(addr_a, addr_b) / 100.0,
            "addr_token_set_ratio": fuzz.token_set_ratio(addr_a, addr_b) / 100.0,
            "addr_token_jaccard": float(jaccard),
            "addr_len_diff_ratio": float(len_diff),
        }

    # Country & context features
    c_a = str(country_a).strip().upper()
    c_b = str(country_b).strip().upper()
    country_match = 1.0 if (c_a and c_b and c_a == c_b) else 0.0
    source_is_s3 = 1.0 if cand_id.startswith("S3-") else 0.0

    context_features = {
        "country_exact_match": country_match,
        "is_source3": source_is_s3,
        "is_addr_missing": is_addr_missing,
    }

    return {**addr_features, **context_features}


class UnifiedFeatureExtractor:
    """Combines Moksh's name features and Dhwaj's address/context features into a single pipeline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def extract_pair(
        self, s1_row: pd.Series, cand_row: pd.Series
    ) -> Dict[str, float]:
        """Extracts complete feature vector for a candidate pair."""
        name_a = str(s1_row.get("name_clean", s1_row.get("business_name", "")))
        name_b = str(cand_row.get("name_clean", cand_row.get("business_name", "")))
        name_feats = extract_name_features(name_a, name_b)

        addr_a = str(s1_row.get("address_clean", s1_row.get("business_address", "")))
        addr_b = str(cand_row.get("address_clean", cand_row.get("business_address", "")))
        country_a = str(s1_row.get("country", ""))
        country_b = str(cand_row.get("country", ""))
        cand_id = str(cand_row.get("entity_id", ""))
        addr_feats = extract_address_and_context_features(
            addr_a, addr_b, country_a, country_b, cand_id
        )

        return {**name_feats, **addr_feats}

    def extract_batch(
        self,
        candidate_pairs_df: pd.DataFrame,
        s1_lookup: Dict[str, pd.Series],
        cand_lookup: Dict[str, pd.Series],
    ) -> pd.DataFrame:
        """Memory-conscious batch feature extraction for candidate pairs."""
        rows: List[Dict[str, float]] = []
        for _, pair in candidate_pairs_df.iterrows():
            s1 = s1_lookup.get(pair["source1_entity_id"])
            cand = cand_lookup.get(pair["candidate_entity_id"])
            if s1 is not None and cand is not None:
                rows.append(self.extract_pair(s1, cand))
            else:
                rows.append({})
        return pd.DataFrame(rows)
