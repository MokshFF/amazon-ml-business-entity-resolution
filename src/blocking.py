"""
Candidate Generation and Blocking Module.

Responsible for reducing the O(N^2) pairwise comparison space to a high-recall candidate pool.
Blocking defines the recall ceiling: any true match missed during blocking cannot be recovered later.

50/50 Technical Ownership:
- Lead: Moksh (Blocking Strategy A: Name token indexing & character prefix blocking)
- Lead: Dhwaj [Team Leader] (Blocking Strategy B: Country-based partitions & address token indexing)

MEMORY EFFICIENCY:
- Avoid quadratic cross-joins (all-vs-all).
- Use inverted indexes and sparse hash mappings.
- Bound max candidates per entity to control downstream feature extraction costs.
"""

from typing import Dict, List, Mapping, Optional, Sequence, Set
import pandas as pd
from src.utils import save_submission_tsv, setup_logger

logger = setup_logger("blocking")


class BlockingStrategyA:
    """Strategy A — Name-Centric Blocking.
    Lead: Moksh
    Explores:
        - Token-level inverted index over distinctive business name tokens
        - Character n-gram and prefix blocking keys (e.g., first 3-4 chars)
    """

    def __init__(self, max_candidates: int = 50):
        self.max_candidates = max_candidates
        self.index: Dict[str, List[str]] = {}

    def build_index(self, records: pd.DataFrame) -> None:
        """Builds name-based inverted index from candidate records."""
        # TODO [Moksh]: Build token/prefix inverted index on candidate names
        pass

    def retrieve_candidates(self, s1_row: pd.Series) -> List[str]:
        """Retrieves candidate entity IDs for a single Source 1 record."""
        # TODO [Moksh]: Query index using S1 name tokens, rank and truncate to max_candidates
        return []


class BlockingStrategyB:
    """Strategy B — Country & Address Partitioning.
    Lead: Dhwaj [Team Leader]
    Explores:
        - Exact country partitioning (dynamic: US, India, France, etc.)
        - Address token indexing (postal codes, street numbers, primary location tokens)
    """

    def __init__(self, max_candidates: int = 50):
        self.max_candidates = max_candidates
        self.index: Dict[str, List[str]] = {}

    def build_index(self, records: pd.DataFrame) -> None:
        """Builds country-partitioned address index from candidate records."""
        # TODO [Dhwaj]: Build address/country partitioned indexing structure
        pass

    def retrieve_candidates(self, s1_row: pd.Series) -> List[str]:
        """Retrieves candidate entity IDs matching country and address criteria."""
        # TODO [Dhwaj]: Query index with country constraint and address tokens
        return []


class MultiStrategyBlocker:
    """Unified blocking interface combining Strategy A (Moksh) and Strategy B (Dhwaj).
    Ensures maximum candidate recall while controlling reduction ratio.
    """

    def __init__(self, max_candidates_per_entity: int = 50):
        self.max_candidates_per_entity = max_candidates_per_entity
        self.strategy_a = BlockingStrategyA(max_candidates=max_candidates_per_entity)
        self.strategy_b = BlockingStrategyB(max_candidates=max_candidates_per_entity)

    def fit_indices(self, s2_records: pd.DataFrame, s3_records: pd.DataFrame) -> None:
        """Constructs blocking indexes over combined Source 2 and Source 3 candidate records."""
        logger.info("Building multi-strategy candidate blocking indexes...")
        combined_cands = pd.concat([s2_records, s3_records], ignore_index=True)
        self.strategy_a.build_index(combined_cands)
        self.strategy_b.build_index(combined_cands)
        logger.info("Candidate indexing complete.")

    def get_candidates_for_entity(self, s1_row: pd.Series) -> List[str]:
        """Retrieves and merges candidate IDs from both strategies for a single S1 entity."""
        cands_a = self.strategy_a.retrieve_candidates(s1_row)
        cands_b = self.strategy_b.retrieve_candidates(s1_row)

        # Union while preserving deterministic order and deduplicating
        merged = list(dict.fromkeys(cands_a + cands_b))
        return merged[: self.max_candidates_per_entity]

    def generate_candidate_map(
        self,
        s1_records: pd.DataFrame,
        s2_records: pd.DataFrame,
        s3_records: pd.DataFrame,
    ) -> Dict[str, List[str]]:
        """Generates candidate map for all Source 1 records."""
        self.fit_indices(s2_records, s3_records)

        candidates_map: Dict[str, List[str]] = {}
        for _, s1_row in s1_records.iterrows():
            s1_id = str(s1_row["entity_id"]).strip()
            cands = self.get_candidates_for_entity(s1_row)
            candidates_map[s1_id] = cands

        return candidates_map


def export_candidate_pairs(
    source1_ids: Sequence[str],
    candidates_map: Mapping[str, Sequence[str]],
    output_path: str,
) -> None:
    """Exports candidate pairs to candidate_pairs.tsv.
    IMPORTANT: This must represent the exact candidate set fed into the final ML matching model.
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
    """Computes blocking recall ceiling and candidate reduction ratio."""
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
