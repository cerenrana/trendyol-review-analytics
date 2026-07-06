"""Sentiment analysis module for Turkish reviews"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple
from transformers import pipeline

from .config import (
    SENTIMENT_MODEL,
    SENTIMENT_MODEL_MAX_LENGTH,
    SENTIMENT_BATCH_SIZE,
    SENTIMENT_LABELS,
    DEVICE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sentiment_model(model_name: str = SENTIMENT_MODEL):
    """
    Load Turkish sentiment analysis model.
    
    Args:
        model_name: HuggingFace model identifier
    
    Returns:
        Sentiment analysis pipeline
    """
    logger.info(f"Loading sentiment model: {model_name}")
    
    try:
        sentiment_pipeline = pipeline(
            "text-classification",
            model=model_name,
            device=0 if DEVICE == "cuda" else -1,
            truncation=True,
            max_length=SENTIMENT_MODEL_MAX_LENGTH,
        )
        logger.info("Sentiment model loaded successfully")
        return sentiment_pipeline
    except Exception as e:
        logger.error(f"Failed to load sentiment model: {str(e)}")
        raise


def analyze_sentiment_batch(
    texts: List[str],
    pipeline_model,
    batch_size: int = SENTIMENT_BATCH_SIZE,
) -> List[Dict]:
    """
    Analyze sentiment for a batch of texts.
    
    Args:
        texts: List of texts to analyze
        pipeline_model: Loaded sentiment pipeline
        batch_size: Batch size for processing
    
    Returns:
        List of sentiment results with labels and scores
    """
    logger.info(f"Analyzing sentiment for {len(texts)} texts...")
    
    results = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        try:
            batch_results = pipeline_model(batch, batch_size=batch_size)
            results.extend(batch_results)
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            # Add neutral sentiment for failed items
            results.extend([{"label": "neutral", "score": 0.5}] * len(batch))
    
    logger.info("Sentiment analysis complete")
    return results


def standardize_sentiment_labels(label: str) -> str:
    """
    Standardize sentiment labels to positive/negative/neutral.
    
    Args:
        label: Raw label from model
    
    Returns:
        Standardized label
    """
    label = label.lower().strip()
    
    if label in ["positive", "pozitif", "olumlu"]:
        return "positive"
    elif label in ["negative", "negatif", "olumsuz"]:
        return "negative"
    else:
        return "neutral"


def add_sentiment_to_documents(
    df: pd.DataFrame,
    sentiment_results: List[Dict],
) -> pd.DataFrame:
    """
    Add sentiment information to documents DataFrame.
    
    Args:
        df: DataFrame with comments
        sentiment_results: Sentiment analysis results
    
    Returns:
        DataFrame with sentiment columns added
    """
    df = df.copy()
    
    # Extract sentiment labels and scores
    sentiments = [standardize_sentiment_labels(r["label"]) for r in sentiment_results]
    scores = [r["score"] for r in sentiment_results]
    
    df["sentiment"] = sentiments
    df["sentiment_score"] = scores
    
    logger.info(f"Added sentiment information to {len(df)} documents")
    
    return df


def calculate_topic_sentiment_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate sentiment statistics for each topic.
    
    Args:
        df: DataFrame with topic_id and sentiment columns
    
    Returns:
        DataFrame with sentiment statistics per topic
    """
    logger.info("Calculating topic-level sentiment statistics...")
    
    topic_stats = []
    
    for topic_id in df["topic_id"].unique():
        topic_df = df[df["topic_id"] == topic_id]
        
        # Count sentiments
        positive_count = (topic_df["sentiment"] == "positive").sum()
        negative_count = (topic_df["sentiment"] == "negative").sum()
        neutral_count = (topic_df["sentiment"] == "neutral").sum()
        total_count = len(topic_df)
        
        # Calculate ratios
        positive_ratio = positive_count / total_count if total_count > 0 else 0
        negative_ratio = negative_count / total_count if total_count > 0 else 0
        neutral_ratio = neutral_count / total_count if total_count > 0 else 0
        
        # Average sentiment score
        avg_score = topic_df["sentiment_score"].mean()
        
        topic_stats.append({
            "topic_id": topic_id,
            "topic_name": topic_df["topic_name"].iloc[0],
            "total_comments": total_count,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "positive_ratio": round(positive_ratio, 4),
            "negative_ratio": round(negative_ratio, 4),
            "neutral_ratio": round(neutral_ratio, 4),
            "avg_sentiment_score": round(avg_score, 4),
        })
    
    stats_df = pd.DataFrame(topic_stats)
    # Sort by negative ratio (most problematic topics first)
    stats_df = stats_df.sort_values("negative_ratio", ascending=False)
    
    logger.info(f"Calculated stats for {len(stats_df)} topics")
    
    return stats_df


def get_topic_sentiment_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get sentiment distribution for each topic (for visualization).
    
    Args:
        df: DataFrame with topic_id and sentiment columns
    
    Returns:
        DataFrame with sentiment distribution data
    """
    logger.info("Calculating sentiment distribution per topic...")
    
    dist_data = []
    
    for topic_id in sorted(df["topic_id"].unique()):
        topic_df = df[df["topic_id"] == topic_id]
        topic_name = topic_df["topic_name"].iloc[0]
        
        sentiment_counts = topic_df["sentiment"].value_counts().to_dict()
        
        dist_data.append({
            "topic_id": topic_id,
            "topic_name": topic_name,
            "positive": sentiment_counts.get("positive", 0),
            "negative": sentiment_counts.get("negative", 0),
            "neutral": sentiment_counts.get("neutral", 0),
            "total": len(topic_df),
        })
    
    dist_df = pd.DataFrame(dist_data)
    return dist_df


def identify_problematic_topics(
    sentiment_stats: pd.DataFrame,
    negative_threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Identify topics with high negative sentiment ratios.
    
    Args:
        sentiment_stats: Topic sentiment statistics
        negative_threshold: Ratio threshold for problematic topics
    
    Returns:
        DataFrame with problematic topics
    """
    problematic = sentiment_stats[sentiment_stats["negative_ratio"] >= negative_threshold]
    logger.info(f"Found {len(problematic)} problematic topics (negative_ratio >= {negative_threshold})")
    
    return problematic.sort_values("negative_ratio", ascending=False)
