"""
Preprocessing and Normalization Module.

Responsible for standardizing noisy business entity records across Source 1, 2, and 3.

50/50 Technical Ownership:
- Lead: Moksh (Name preprocessing, legal suffixes, abbreviations, noise profiling)
- Lead: Dhwaj [Team Leader] (Address preprocessing, landmark tokens, country handling)

DESIGN PRINCIPLE:
Do NOT assume aggressive normalization rules before inspecting real data distributions during EDA.
Avoid overly aggressive stripping that could cause false merges between distinct businesses.
Designed to process large records efficiently via chunking.
"""

import re
from typing import Iterator, Optional
import pandas as pd


def normalize_text(text: Optional[str]) -> str:
    """Shared baseline normalization: lowercasing, strip leading/trailing spaces,
    and collapse multiple whitespaces.

    Args:
        text: Input string or None.

    Returns:
        str: Cleaned baseline string.
    """
    if text is None or pd.isna(text):
        return ""
    text = str(text).lower().strip()
    return re.sub(r"\s+", " ", text)


def normalize_name(name: Optional[str]) -> str:
    """Normalizes business entity names.

    Lead: Moksh
    Focus:
        - Profiling legal suffix variations ('Corp', 'Corporation', 'Inc', 'Pvt Ltd', 'LLC')
        - Symbol normalization ('&' vs 'and')
        - Typo patterns and punctuation handling

    NOTE: Avoid overly aggressive normalization that could merge distinct businesses.
    Actual rules must be backed by validation findings from `eda_moksh.ipynb`.

    Args:
        name: Raw business name.

    Returns:
        str: Normalized business name.
    """
    cleaned = normalize_text(name)
    if not cleaned:
        return ""

    # TODO [Moksh]: Implement data-driven name cleaning rules after EDA:
    # - Standardize legal suffixes based on frequency analysis
    # - Handle punctuation without stripping distinguishing acronyms

    return cleaned


def normalize_address(address: Optional[str]) -> str:
    """Normalizes business entity addresses.

    Lead: Dhwaj [Team Leader]
    Focus:
        - Road / street abbreviation expansion or standardization ('rd' vs 'road', 'st' vs 'street')
        - Landmark reference patterns ('near ...', 'opp ...')
        - Postal code extraction and component tokenization

    NOTE: Avoid stripping address components that distinguish multiple branches of the same chain.
    Actual rules must be backed by validation findings from `eda_dhwaj.ipynb`.

    Args:
        address: Raw business address string.

    Returns:
        str: Normalized address string.
    """
    cleaned = normalize_text(address)
    if not cleaned:
        return ""

    # TODO [Dhwaj]: Implement data-driven address cleaning rules after EDA:
    # - Street suffix harmonization
    # - Postal / PIN code identification
    # - Landmark noise mitigation

    return cleaned


def normalize_country(country: Optional[str]) -> str:
    """Normalizes country label strings.

    Lead: Dhwaj [Team Leader]
    CRITICAL RULE:
        The training set contains 'US' and 'India'. The test set contains 'France'
        and possibly other open-set country labels. Never filter or hardcode to only US/India!

    Args:
        country: Raw country string.

    Returns:
        str: Cleaned uppercase country identifier.
    """
    if country is None or pd.isna(country):
        return "UNKNOWN"
    return str(country).strip().upper()


def preprocess_records(df: pd.DataFrame) -> pd.DataFrame:
    """Applies preprocessing transformations to a DataFrame of business records.

    Expected columns:
        ['entity_id', 'business_name', 'business_address', 'country']

    Produces normalized columns:
        ['name_clean', 'address_clean', 'country_clean']

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with normalized columns.
    """
    df_clean = df.copy()
    df_clean["name_clean"] = df_clean["business_name"].apply(normalize_name)
    df_clean["address_clean"] = df_clean["business_address"].apply(normalize_address)
    df_clean["country_clean"] = df_clean["country"].apply(normalize_country)
    return df_clean


def preprocess_chunks(
    chunk_iterator: Iterator[pd.DataFrame],
) -> Iterator[pd.DataFrame]:
    """Memory-efficient generator that yields preprocessed chunks for large files.

    Args:
        chunk_iterator: Iterator yielding DataFrame chunks.

    Yields:
        pd.DataFrame: Cleaned DataFrame chunk.
    """
    for chunk in chunk_iterator:
        yield preprocess_records(chunk)
