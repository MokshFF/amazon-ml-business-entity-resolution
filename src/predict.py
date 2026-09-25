"""
End-to-End Inference and Submission Generation Module.

Coordinates the full prediction pipeline for the test set:
1. Load test records (Source 1, Source 2, Source 3)
2. Preprocess and normalize text fields
3. Candidate Generation / Blocking -> writes candidate_pairs.tsv
4. Similarity Feature Engineering on candidate pairs
5. ML Model Scoring and Threshold Decision
6. Output Generation -> writes matching_results.tsv
7. Format Validation check

Team Role Assigned: Member 4 (Integration, Test inference, Output generation, Validation).
"""

import argparse
import os
import sys
from typing import Dict, List, Optional
import pandas as pd

from src.blocking import CandidateGenerator, export_candidate_pairs
from src.features import PairwiseFeatureExtractor
from src.model import EntityMatchingModel
from src.preprocessing import preprocess_records
from src.utils import load_config, load_source_tsv, save_submission_tsv, setup_logger

logger = setup_logger("predict")


def run_inference_pipeline(config_path: str = "configs/config.yaml") -> None:
    """Executes the end-to-end test inference pipeline.

    Args:
        config_path: Path to configuration YAML file.
    """
    logger.info("Initializing Entity Resolution test inference pipeline...")
    config = load_config(config_path)

    # 1. Load test datasets
    test_paths = config["paths"]["test"]
    output_paths = config["paths"]["output"]

    logger.info(f"Loading Source 1 test records from: {test_paths['source1']}")
    logger.info(f"Loading Source 2 test records from: {test_paths['source2']}")
    logger.info(f"Loading Source 3 test records from: {test_paths['source3']}")

    # TODO [Member 4]: Verify file existence before reading
    if not (
        os.path.exists(test_paths["source1"])
        and os.path.exists(test_paths["source2"])
        and os.path.exists(test_paths["source3"])
    ):
        logger.warning(
            "Test data files not found at specified paths. Ensure dataset is downloaded locally."
        )
        return

    df_s1 = load_source_tsv(test_paths["source1"])
    df_s2 = load_source_tsv(test_paths["source2"])
    df_s3 = load_source_tsv(test_paths["source3"])

    # 2. Preprocess and normalize records
    logger.info("Preprocessing and normalizing test records across sources...")
    df_s1_clean = preprocess_records(df_s1)
    df_s2_clean = preprocess_records(df_s2)
    df_s3_clean = preprocess_records(df_s3)

    # 3. Candidate Generation / Blocking
    logger.info("Running blocking to generate candidate pairs...")
    candidate_generator = CandidateGenerator(
        max_candidates=config["blocking"]["max_candidates_per_entity"]
    )
    candidates_map = candidate_generator.generate_all_candidates(
        df_s1_clean, df_s2_clean, df_s3_clean
    )

    # Export candidate_pairs.tsv
    cand_pairs_path = output_paths["candidate_pairs"]
    logger.info(f"Exporting blocking candidates to: {cand_pairs_path}")
    export_candidate_pairs(
        source1_ids=df_s1["entity_id"].tolist(),
        candidates_map=candidates_map,
        output_path=cand_pairs_path,
    )

    # 4. Feature Extraction & Model Inference
    # TODO [Member 4 / Member 3]: Connect trained model checkpoint and batch scoring
    matching_results: Dict[str, List[str]] = {}

    # Placeholder: Initialize default empty match list (singletons) for all Source 1 entities
    for s1_id in df_s1["entity_id"]:
        # TODO [Member 4]: Populate with ML model predicted matches above tuned threshold
        matching_results[s1_id] = []

    # 5. Output Generation -> matching_results.tsv
    matching_results_path = output_paths["matching_results"]
    logger.info(f"Writing final entity matches to: {matching_results_path}")
    save_submission_tsv(
        source1_ids=df_s1["entity_id"].tolist(),
        id_mapping=matching_results,
        output_path=matching_results_path,
        id_col_name="matched_entity_ids",
    )

    logger.info("Inference completed successfully.")
    logger.info(f"Output files generated: {cand_pairs_path} and {matching_results_path}")


def main() -> None:
    """CLI entry point for running prediction pipeline."""
    parser = argparse.ArgumentParser(
        description="Run Entity Resolution inference pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()
    run_inference_pipeline(args.config)


if __name__ == "__main__":
    main()
