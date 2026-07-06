"""Export results to CSV and JSON formats"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .config import (
    OUTPUT_REVIEWS_CSV,
    OUTPUT_SUMMARY_CSV,
    OUTPUT_SUMMARY_JSON,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_reviews_export(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare reviews DataFrame for export.
    
    Args:
        df: DataFrame with all review information
    
    Returns:
        DataFrame with selected columns for export
    """
    export_columns = [
        "original_comment",
        "cleaned_comment",
        "topic_id",
        "topic_name",
        "topic_keywords",
        "sentiment",
        "sentiment_score",
    ]
    
    # Check which columns exist and select available ones
    available_columns = [col for col in export_columns if col in df.columns]
    
    export_df = df[available_columns].copy()
    
    logger.info(f"Prepared {len(export_df)} reviews for export")
    return export_df


def prepare_summary_export(
    sentiment_stats: pd.DataFrame,
    representative_comments: Dict[int, List[str]],
) -> pd.DataFrame:
    """
    Prepare topic summary DataFrame for export.
    
    Args:
        sentiment_stats: Topic sentiment statistics
        representative_comments: Dictionary of representative comments per topic
    
    Returns:
        DataFrame with topic summaries
    """
    summary_data = []
    
    for idx, row in sentiment_stats.iterrows():
        topic_id = row["topic_id"]
        
        # Get representative comments (join as string for CSV)
        rep_comments = representative_comments.get(topic_id, [])
        rep_comments_str = " | ".join(rep_comments[:3]) if rep_comments else ""
        
        summary_data.append({
            "topic_id": topic_id,
            "topic_name": row["topic_name"],
            "topic_size": row["total_comments"],
            "keywords": row.get("keywords", ""),
            "representative_comments": rep_comments_str,
            "positive_count": row["positive_count"],
            "negative_count": row["negative_count"],
            "neutral_count": row["neutral_count"],
            "positive_ratio": row["positive_ratio"],
            "negative_ratio": row["negative_ratio"],
            "neutral_ratio": row["neutral_ratio"],
            "avg_sentiment_score": row["avg_sentiment_score"],
        })
    
    summary_df = pd.DataFrame(summary_data)
    logger.info(f"Prepared summary for {len(summary_df)} topics")
    
    return summary_df


def export_reviews_csv(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Export reviews to CSV.
    
    Args:
        df: Reviews DataFrame
        output_path: Output file path. If None, uses config OUTPUT_REVIEWS_CSV.
    
    Returns:
        Path to exported file
    """
    path = output_path or OUTPUT_REVIEWS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    
    export_df = prepare_reviews_export(df)
    export_df.to_csv(path, index=False, encoding="utf-8")
    
    logger.info(f"Exported {len(export_df)} reviews to {path}")
    return path


def export_summary_csv(
    sentiment_stats: pd.DataFrame,
    representative_comments: Dict[int, List[str]],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Export topic summary to CSV.
    
    Args:
        sentiment_stats: Topic sentiment statistics
        representative_comments: Dictionary of representative comments per topic
        output_path: Output file path. If None, uses config OUTPUT_SUMMARY_CSV.
    
    Returns:
        Path to exported file
    """
    path = output_path or OUTPUT_SUMMARY_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    
    summary_df = prepare_summary_export(sentiment_stats, representative_comments)
    summary_df.to_csv(path, index=False, encoding="utf-8")
    
    logger.info(f"Exported topic summary to {path}")
    return path


def export_summary_json(
    sentiment_stats: pd.DataFrame,
    representative_comments: Dict[int, List[str]],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Export topic summary to JSON.
    
    Args:
        sentiment_stats: Topic sentiment statistics
        representative_comments: Dictionary of representative comments per topic
        output_path: Output file path. If None, uses config OUTPUT_SUMMARY_JSON.
    
    Returns:
        Path to exported file
    """
    path = output_path or OUTPUT_SUMMARY_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare JSON structure
    json_data = {
        "metadata": {
            "total_topics": int(len(sentiment_stats)),
            "total_reviews": int(sentiment_stats["total_comments"].sum()),
        },
        "topics": [],
    }
    
    for idx, row in sentiment_stats.iterrows():
        topic_id = row["topic_id"]
        
        # Get representative comments
        rep_comments = representative_comments.get(topic_id, [])
        
        topic_data = {
            "topic_id": int(topic_id),
            "topic_name": str(row["topic_name"]),
            "topic_size": int(row["total_comments"]),
            "keywords": str(row.get("keywords", "")),
            "representative_comments": rep_comments[:3] if rep_comments else [],
            "sentiment_analysis": {
                "positive": {
                    "count": int(row["positive_count"]),
                    "ratio": float(row["positive_ratio"]),
                },
                "negative": {
                    "count": int(row["negative_count"]),
                    "ratio": float(row["negative_ratio"]),
                },
                "neutral": {
                    "count": int(row["neutral_count"]),
                    "ratio": float(row["neutral_ratio"]),
                },
                "avg_score": float(row["avg_sentiment_score"]),
            },
        }
        json_data["topics"].append(topic_data)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Exported topic summary to JSON: {path}")
    return path


def export_all_results(
    reviews_df: pd.DataFrame,
    sentiment_stats: pd.DataFrame,
    representative_comments: Dict[int, List[str]],
) -> Dict[str, Path]:
    """
    Export all results to CSV and JSON files.
    
    Args:
        reviews_df: Reviews with topics and sentiments
        sentiment_stats: Topic sentiment statistics
        representative_comments: Dictionary of representative comments per topic
    
    Returns:
        Dictionary with paths to exported files
    """
    logger.info("Exporting all results...")
    
    paths = {
        "reviews_csv": export_reviews_csv(reviews_df),
        "summary_csv": export_summary_csv(sentiment_stats, representative_comments),
        "summary_json": export_summary_json(sentiment_stats, representative_comments),
    }
    
    logger.info("All results exported successfully")
    logger.info(f"  - Reviews CSV: {paths['reviews_csv']}")
    logger.info(f"  - Summary CSV: {paths['summary_csv']}")
    logger.info(f"  - Summary JSON: {paths['summary_json']}")
    
    return paths


def validate_exports() -> bool:
    """
    Validate that all expected output files exist.
    
    Returns:
        True if all files exist, False otherwise
    """
    files_to_check = [
        OUTPUT_REVIEWS_CSV,
        OUTPUT_SUMMARY_CSV,
        OUTPUT_SUMMARY_JSON,
    ]
    
    all_exist = all(f.exists() for f in files_to_check)
    
    if all_exist:
        logger.info("All output files validated successfully")
    else:
        missing = [f for f in files_to_check if not f.exists()]
        logger.warning(f"Missing output files: {missing}")
    
    return all_exist
