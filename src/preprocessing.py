"""
Preprocessing and Normalization Module.

Responsible for cleaning, standardizing, and normalizing noisy business entity records
from Source 1, Source 2, and Source 3 before blocking and feature generation.

Team Role Assigned: Member 1 (EDA, Data profiling, Preprocessing/normalization).
"""

import re
import string
from typing import Optional
import pandas as pd


def normalize_text_basic(text: Optional[str]) -> str:
    """Performs baseline text normalization: lowercasing, whitespace stripping,
    and Unicode punctuation normalization.

    Args:
        text: Input string or None.

    Returns:
        str: Cleaned string.
    """
    if text is None or pd.isna(text):
        return ""
    text = str(text).lower().strip()
    # Replace multiple whitespace characters with a single space
    text = re.sub(r"\s+", " ", text)
    return text


def clean_business_name(name: Optional[str]) -> str:
    """Normalizes a business entity name string.

    Handles noise patterns such as legal suffixes (e.g., Corp, Pvt Ltd, Inc),
    punctuation variations (& vs and), casing, and special characters.

    Args:
        name: Raw business name.

    Returns:
        str: Normalized business name.
    """
    text = normalize_text_basic(name)
    if not text:
        return ""

    # TODO [Member 1]: Expand and standardize legal entity suffixes:
    #   - 'corporation', 'corp', 'incorporated', 'inc' -> 'inc'
    #   - 'private limited', 'pvt ltd', 'ltd', 'limited' -> 'ltd'
    #   - '&' -> 'and'
    # TODO [Member 1]: Profile character distributions across sources during EDA.

    return text


def clean_business_address(address: Optional[str]) -> str:
    """Normalizes a business address string.

    Handles noise patterns such as street abbreviations (Rd vs Road, St vs Street),
    transliteration noise, landmark references, and punctuation.

    Args:
        address: Raw business address string.

    Returns:
        str: Normalized address string.
    """
    text = normalize_text_basic(address)
    if not text:
        return ""

    # TODO [Member 1]: Implement address token normalization:
    #   - 'rd', 'rd.' -> 'road'
    #   - 'st', 'st.' -> 'street'
    #   - 'ave', 'ave.' -> 'avenue'
    #   - 'fl', 'flr', 'floor'
    # TODO [Member 1]: Handle Indian/US/French postal code variations and landmark phrases.

    return text


def clean_country(country: Optional[str]) -> str:
    """Standardizes country label strings.

    CRITICAL NOTE:
        Training data covers 'US' and 'India'. The test set additionally contains 'France'
        and possibly other open-set labels. Do NOT hard-code or filter to only US/India!

    Args:
        country: Raw country string.

    Returns:
        str: Normalized uppercase country label.
    """
    if country is None or pd.isna(country):
        return "UNKNOWN"
    return str(country).strip().upper()


def preprocess_records(df: pd.DataFrame) -> pd.DataFrame:
    """Applies preprocessing transformations to a business records DataFrame.

    Expected columns in input DataFrame:
        - entity_id
        - business_name
        - business_address
        - country

    Adds normalized columns:
        - name_clean
        - address_clean
        - country_clean

    Args:
        df: Input DataFrame containing raw source records.

    Returns:
        pd.DataFrame: DataFrame with normalized feature columns added.
    """
    df_clean = df.copy()

    # Apply cleaning pipelines
    df_clean["name_clean"] = df_clean["business_name"].apply(clean_business_name)
    df_clean["address_clean"] = df_clean["business_address"].apply(clean_business_address)
    df_clean["country_clean"] = df_clean["country"].apply(clean_country)

    # TODO [Member 1]: Integrate additional profiling-driven normalization steps.

    return df_clean
