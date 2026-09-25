"""
Candidate Generation and Blocking Module.

Responsible for reducing the O(N^2) pairwise search space between Source 1
and (Source 2 + Source 3) records to a high-recall candidate set.

Team Role Assigned: Member 2 (Blocking, Candidate generation, candidate_pairs.tsv).
"""

from typing import Dict, List, Mapping, Optional, Sequence, Set
import pandas as pd
from src.utils import save_submission_tsv


class CandidateGenerator:
    """Interface and pipeline for blocking and candidate generation."""

    def __init__(self, max_candidates: int = 50, country_blocking: bool = True):
        """Initializes candidate generator with blocking configuration.

        Args:
            max_candidates: Maximum candidates allowed per Source 1 entity.
            country_blocking: Whether to strictly block on country matching.
        """
        self.max_candidates = max_candidates
        self.country_blocking = country_blocking

    def build_index(self, s2_records: pd.DataFrame, s3_records: pd.DataFrame) -> None:
        """Builds candidate search indices over combined Source 2 and Source 3 records.

        Args:
            s2_records: Preprocessed DataFrame for Source 2 records.
            s3_records: Preprocessed DataFrame for Source 3 records.
        """
        # TODO [Member 2]: Combine S2 and S3 records and build indexing data structures:
        #   - Token-level inverted indices (e.g., TF-IDF top terms or token sets)
        #   - Character n-gram blocking keys
        #   - Phonetic/Soundex blocking keys on core business tokens
        #   - Country-based partitions (must handle US, India, France dynamically)
        pass

    def generate_candidates_for_entity(self, s1_row: pd.Series) -> List[str]:
        """Generates candidate IDs (from S2 / S3) for a single Source 1 record.

        Args:
            s1_row: A single row representing a preprocessed Source 1 record.

        Returns:
            List[str]: List of candidate entity IDs (prefixed with 'S2-' or 'S3-').
        """
        # TODO [Member 2]: Query blocking index and rank/filter candidates up to self.max_candidates.
        return []

    def generate_all_candidates(
        self,
        s1_records: pd.DataFrame,
        s2_records: pd.DataFrame,
        s3_records: pd.DataFrame,
    ) -> Dict[str, List[str]]:
        """Executes candidate generation across all Source 1 entities.

        Args:
            s1_records: Preprocessed Source 1 records DataFrame.
            s2_records: Preprocessed Source 2 records DataFrame.
            s3_records: Preprocessed Source 3 records DataFrame.

        Returns:
            Dict[str, List[str]]: Map of source1_entity_id -> list of candidate entity IDs.
        """
        self.build_index(s2_records, s3_records)

        candidates_map: Dict[str, List[str]] = {}
        # Ensure every Source 1 entity has an entry (even if empty)
        for _, s1_row in s1_records.iterrows():
            s1_id = str(s1_row["entity_id"]).strip()
            candidates = self.generate_candidates_for_entity(s1_row)
            # Deduplicate candidate IDs while preserving order
            candidates_map[s1_id] = list(dict.fromkeys(candidates))

        return candidates_map


def export_candidate_pairs(
    source1_ids: Sequence[str],
    candidates_map: Mapping[str, Sequence[str]],
    output_path: str,
) -> None:
    """Exports candidate pairs to the official candidate_pairs.tsv format.

    Args:
        source1_ids: All Source 1 entity IDs in the dataset split.
        candidates_map: Mapping from source1_id to candidate IDs.
        output_path: Path where candidate_pairs.tsv will be written.
    """
    save_submission_tsv(
        source1_ids=source1_ids,
        id_mapping=candidates_map,
        output_path=output_path,
        id_col_name="candidate_entity_ids",
    )


def compute_blocking_metrics(
    ground_truth: Mapping[str, Set[str]],
    candidates_map: Mapping[str, Sequence[str]],
) -> Dict[str, float]:
    """Computes blocking recall (recall ceiling) and candidate reduction ratio.

    Args:
        ground_truth: Mapping from source1_id to true matching IDs.
        candidates_map: Mapping from source1_id to candidate IDs from blocking.

    Returns:
        Dict[str, float]: Dictionary with 'blocking_recall', 'avg_candidates_per_entity'.
    """
    total_true_matches = sum(len(matches) for matches in ground_truth.values())
    captured_matches = 0
    total_candidates = 0

    for s1_id, true_matches in ground_truth.items():
        candidates = set(candidates_map.get(s1_id, []))
        total_candidates += len(candidates)
        captured_matches += len(true_matches.intersection(candidates))

    recall_ceiling = (
        captured_matches / total_true_matches if total_true_matches > 0 else 0.0
    )
    avg_candidates = (
        total_candidates / len(ground_truth) if len(ground_truth) > 0 else 0.0
    )

    return {
        "blocking_recall_ceiling": recall_ceiling,
        "avg_candidates_per_entity": avg_candidates,
        "total_true_matches": float(total_true_matches),
        "captured_true_matches": float(captured_matches),
    }
