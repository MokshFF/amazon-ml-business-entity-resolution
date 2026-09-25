"""
End-to-End Test Inference and Submission Generation Module.

Joint Ownership & Integration: Moksh & Dhwaj [Team Leader].

Execution Flow:
1. Load test records (S1, S2, S3) with memory-safe dtypes
2. Preprocess & normalize records without aggressive over-cleaning
3. Multi-strategy candidate generation -> outputs candidate_pairs.tsv
4. Unified feature engineering on candidate pairs (name + address/country features)
5. ML scoring & F0.5 precision-weighted thresholding
6. Final match selection -> outputs matching_results.tsv
"""

import argparse
import os
from typing import Dict, List
import pandas as pd

from src.blocking import MultiStrategyBlocker, export_candidate_pairs
from src.features import UnifiedFeatureExtractor
from src.model import EntityMatchingModel
from src.preprocessing import preprocess_records
from src.utils import load_config, load_source_tsv, save_submission_tsv, setup_logger

logger = setup_logger("predict")


def run_pipeline(config_path: str = "configs/config.yaml") -> None:
    """Executes the test prediction pipeline."""
    logger.info("Initializing Entity Resolution inference pipeline...")
    config = load_config(config_path)

    test_paths = config["paths"]["test"]
    output_paths = config["paths"]["output"]

    # Verify input test files exist
    for source_key in ["source1", "source2", "source3"]:
        path = test_paths[source_key]
        if not os.path.exists(path):
            logger.warning(
                f"Test file '{path}' not found. Download or place dataset in '{config['paths']['dataset_dir']}'."
            )
            return

    # 1. Load test data
    logger.info("Loading test sources...")
    df_s1 = load_source_tsv(test_paths["source1"])
    df_s2 = load_source_tsv(test_paths["source2"])
    df_s3 = load_source_tsv(test_paths["source3"])

    # 2. Preprocess records
    logger.info("Running preprocessing and normalization...")
    df_s1_clean = preprocess_records(df_s1)
    df_s2_clean = preprocess_records(df_s2)
    df_s3_clean = preprocess_records(df_s3)

    # 3. Candidate Generation / Blocking
    logger.info("Generating candidate pairs via MultiStrategyBlocker...")
    blocker = MultiStrategyBlocker(
        max_candidates_per_entity=config["blocking"]["max_candidates_per_entity"]
    )
    candidates_map = blocker.generate_candidate_map(
        df_s1_clean, df_s2_clean, df_s3_clean
    )

    # Export candidate_pairs.tsv
    cand_pairs_path = output_paths["candidate_pairs"]
    logger.info(f"Writing candidate pairs to {cand_pairs_path}...")
    export_candidate_pairs(
        source1_ids=df_s1["entity_id"].tolist(),
        candidates_map=candidates_map,
        output_path=cand_pairs_path,
    )

    # 4. Feature Extraction & Scoring (Placeholder)
    # TODO [Moksh & Dhwaj]: Batch feature extraction using UnifiedFeatureExtractor
    matching_results: Dict[str, List[str]] = {}

    for s1_id in df_s1["entity_id"]:
        # Default initialization: singleton (empty match list)
        matching_results[s1_id] = []

    # 5. Export matching_results.tsv
    matching_results_path = output_paths["matching_results"]
    logger.info(f"Writing final matching results to {matching_results_path}...")
    save_submission_tsv(
        source1_ids=df_s1["entity_id"].tolist(),
        id_mapping=matching_results,
        output_path=matching_results_path,
        id_col_name="matched_entity_ids",
    )

    logger.info("Inference completed successfully.")
    logger.info(f"Generated outputs:\n  1. {cand_pairs_path}\n  2. {matching_results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Entity Resolution inference pipeline.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
