"""Preprocessing module for text data and data validation"""

import pandas as pd
import re
from pathlib import Path
from typing import Tuple, Optional
import logging

from .config import (
    INPUT_CSV,
    MIN_COMMENT_LENGTH,
    MAX_COMMENT_LENGTH,
    TURKISH_STOPWORDS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(input_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load customer reviews from CSV file.
    
    Args:
        input_path: Path to input CSV file. If None, uses config INPUT_CSV.
    
    Returns:
        DataFrame with reviews loaded
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If CSV structure is invalid
    """
    path = input_path or INPUT_CSV
    
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} reviews from {path}")
        return df
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {str(e)}")


def select_comment_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select appropriate comment column.
    Prioritizes 'cleaned_comment' if available, otherwise uses 'comment'.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with standardized 'comment' column
    
    Raises:
        ValueError: If neither column exists
    """
    if "cleaned_comment" in df.columns:
        logger.info("Using 'cleaned_comment' column")
        df = df.copy()
        df["comment"] = df["cleaned_comment"]
        df["original_comment"] = df.get("comment", df["cleaned_comment"])
    elif "comment" in df.columns:
        logger.info("Using 'comment' column")
        df = df.copy()
        df["original_comment"] = df["comment"]
    else:
        raise ValueError("DataFrame must contain 'comment' or 'cleaned_comment' column")
    
    return df


def clean_text(text: str) -> str:
    """
    Clean Turkish text: remove URLs, emails, special chars, normalize whitespace.
    
    Args:
        text: Raw text to clean
    
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove numbers and special characters (keep Turkish characters)
    text = re.sub(r'[^\w\s\-çğıöşüÇĞİÖŞÜ]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def remove_stopwords(text: str, stopwords: list = None) -> str:
    """
    Remove Turkish stopwords from text.
    
    Args:
        text: Input text
        stopwords: List of stopwords. If None, uses config TURKISH_STOPWORDS.
    
    Returns:
        Text with stopwords removed
    """
    if stopwords is None:
        stopwords = TURKISH_STOPWORDS
    
    words = text.split()
    filtered_words = [w for w in words if w not in stopwords and len(w) > 1]
    return " ".join(filtered_words)


def filter_comments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out empty, too short, and too long comments.
    
    Args:
        df: DataFrame with 'comment' column
    
    Returns:
        Filtered DataFrame
    """
    initial_count = len(df)
    
    # Remove NaN values
    df = df.dropna(subset=["comment"])
    
    # Convert to string
    df = df.copy()
    df["comment"] = df["comment"].astype(str)
    
    # Remove empty strings
    df = df[df["comment"].str.strip() != ""]
    
    # Remove comments that are too short
    df = df[df["comment"].str.len() >= MIN_COMMENT_LENGTH]
    
    # Remove comments that are too long
    df = df[df["comment"].str.len() <= MAX_COMMENT_LENGTH]
    
    removed_count = initial_count - len(df)
    logger.info(f"Filtered out {removed_count} comments (kept {len(df)})")
    
    return df


def preprocess_reviews(df: pd.DataFrame, remove_stopwords_flag: bool = True) -> pd.DataFrame:
    """
    Apply full preprocessing pipeline to reviews.
    
    Args:
        df: Input DataFrame
        remove_stopwords_flag: Whether to remove stopwords
    
    Returns:
        Preprocessed DataFrame
    """
    logger.info("Starting preprocessing pipeline...")
    
    # Select appropriate comment column
    df = select_comment_column(df)
    
    # Filter comments
    df = filter_comments(df)
    
    # Clean text
    logger.info("Cleaning text...")
    df["cleaned_comment"] = df["comment"].apply(clean_text)
    
    # Remove stopwords if flag is set
    if remove_stopwords_flag:
        logger.info("Removing Turkish stopwords...")
        df["cleaned_comment"] = df["cleaned_comment"].apply(
            lambda x: remove_stopwords(x, TURKISH_STOPWORDS)
        )
    
    # Filter again after stopword removal (might create empty comments)
    df = df[df["cleaned_comment"].str.strip() != ""]
    df = df[df["cleaned_comment"].str.len() >= MIN_COMMENT_LENGTH]
    
    logger.info(f"Preprocessing complete. Final dataset: {len(df)} comments")
    
    return df


def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Validate dataset before model training.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        True if valid, raises exception otherwise
    
    Raises:
        ValueError: If dataset is invalid
    """
    if len(df) == 0:
        raise ValueError("Dataset is empty")
    
    if len(df) < 20:
        logger.warning(f"Dataset has only {len(df)} comments. Minimum recommended: 20")
    
    if "cleaned_comment" not in df.columns:
        raise ValueError("DataFrame must have 'cleaned_comment' column")
    
    null_count = df["cleaned_comment"].isna().sum()
    if null_count > 0:
        raise ValueError(f"Dataset has {null_count} null values in 'cleaned_comment'")
    
    logger.info("Dataset validation passed")
    return True


def get_preprocessing_stats(original_df: pd.DataFrame, processed_df: pd.DataFrame) -> dict:
    """
    Calculate preprocessing statistics.
    
    Args:
        original_df: Original DataFrame
        processed_df: Processed DataFrame
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        "original_count": len(original_df),
        "final_count": len(processed_df),
        "removed_count": len(original_df) - len(processed_df),
        "removal_percentage": round((len(original_df) - len(processed_df)) / len(original_df) * 100, 2),
        "avg_cleaned_length": round(processed_df["cleaned_comment"].str.len().mean(), 2),
    }
    return stats
