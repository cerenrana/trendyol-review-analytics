"""Run Turkish BERT sentiment analysis for raw Trendyol reviews."""

import argparse
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
from transformers import pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.topic_modeling.config import (
    DEVICE,
    SENTIMENT_BATCH_SIZE,
    SENTIMENT_LABELS,
    SENTIMENT_MODEL,
    SENTIMENT_MODEL_MAX_LENGTH,
)

DEFAULT_INPUT_PATH = BASE_DIR / "data" / "raw" / "reviews.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "processed" / "reviews_with_sentiment.csv"


def normalize_label(label: str) -> str:
    """Normalize model-specific labels to POSITIVE/NEGATIVE/NEUTRAL."""
    normalized = str(label).strip().lower()

    if normalized.startswith("label_"):
        label_id = int(normalized.replace("label_", ""))
        normalized = SENTIMENT_LABELS.get(label_id, "neutral")

    label_map: Dict[str, str] = {
        "positive": "POSITIVE",
        "pozitif": "POSITIVE",
        "olumlu": "POSITIVE",
        "negative": "NEGATIVE",
        "negatif": "NEGATIVE",
        "olumsuz": "NEGATIVE",
        "neutral": "NEUTRAL",
        "notr": "NEUTRAL",
        "nötr": "NEUTRAL",
    }
    return label_map.get(normalized, "NEUTRAL")


def load_reviews(input_path: Path) -> pd.DataFrame:
    """Load raw reviews and validate the expected text column."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if "comment" not in df.columns:
        raise ValueError("Input CSV must contain a 'comment' column.")

    df = df.copy()
    df["comment"] = df["comment"].fillna("").astype(str)
    return df


def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment label and confidence score columns to reviews."""
    sentiment_pipeline = pipeline(
        "text-classification",
        model=SENTIMENT_MODEL,
        device=0 if DEVICE == "cuda" else -1,
        truncation=True,
        max_length=SENTIMENT_MODEL_MAX_LENGTH,
    )

    comments = df["comment"].tolist()
    results = sentiment_pipeline(comments, batch_size=SENTIMENT_BATCH_SIZE)

    output_df = df.copy()
    output_df["sentiment_label"] = [
        normalize_label(result.get("label", "neutral")) for result in results
    ]
    output_df["sentiment_score"] = [
        float(result.get("score", 0.0)) for result in results
    ]
    return output_df


def save_results(df: pd.DataFrame, output_path: Path) -> None:
    """Save sentiment results as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved sentiment dataset to: {output_path}")
    print(f"Total reviews: {len(df)}")
    print("Sentiment distribution:")
    print(df["sentiment_label"].value_counts())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze sentiment for raw Trendyol review comments."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input CSV path. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading reviews from: {args.input}")
    reviews_df = load_reviews(args.input)
    result_df = analyze_sentiment(reviews_df)
    save_results(result_df, args.output)


if __name__ == "__main__":
    main()
