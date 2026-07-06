"""Main pipeline orchestration for topic modeling and sentiment analysis"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .preprocessing import (
    load_data,
    preprocess_reviews,
    get_preprocessing_stats,
)
from .modeling import (
    generate_embeddings,
    create_topic_model,
    extract_topic_info,
    get_representative_comments,
    assign_topics_to_documents,
)
from .sentiment import (
    load_sentiment_model,
    analyze_sentiment_batch,
    add_sentiment_to_documents,
    calculate_topic_sentiment_stats,
    get_topic_sentiment_distribution,
    identify_problematic_topics,
)
from .export import (
    export_all_results,
    validate_exports,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline(
    input_csv: Optional[Path] = None,
    remove_stopwords: bool = True,
    negative_threshold: float = 0.3,
) -> Dict[str, Path]:
    """
    Run complete topic modeling and sentiment analysis pipeline.
    
    Pipeline steps:
    1. Load and preprocess reviews
    2. Generate embeddings
    3. Create topic model with UMAP + HDBSCAN + BERTopic
    4. Analyze sentiment for each review
    5. Calculate topic-level sentiment statistics
    6. Export results
    
    Args:
        input_csv: Path to input CSV file. If None, uses config default.
        remove_stopwords: Whether to remove Turkish stopwords during preprocessing.
        negative_threshold: Threshold for identifying problematic topics.
    
    Returns:
        Dictionary with paths to all output files
    
    Raises:
        Exception: If any step fails
    """
    try:
        logger.info("=" * 60)
        logger.info("Starting Topic Modeling and Sentiment Analysis Pipeline")
        logger.info("=" * 60)
        
        # ===== STEP 1: Load and Preprocess =====
        logger.info("\n[1/6] PREPROCESSING")
        logger.info("-" * 60)
        
        raw_df = load_data(input_csv)
        preprocessed_df = preprocess_reviews(raw_df, remove_stopwords_flag=remove_stopwords)
        
        # Get preprocessing statistics
        prep_stats = get_preprocessing_stats(raw_df, preprocessed_df)
        logger.info(f"Preprocessing stats:")
        for key, value in prep_stats.items():
            logger.info(f"  {key}: {value}")
        
        # ===== STEP 2: Generate Embeddings =====
        logger.info("\n[2/6] EMBEDDING GENERATION")
        logger.info("-" * 60)
        
        comments_list = preprocessed_df["cleaned_comment"].tolist()
        embeddings = generate_embeddings(comments_list)
        
        # ===== STEP 3: Topic Modeling =====
        logger.info("\n[3/6] TOPIC MODELING")
        logger.info("-" * 60)
        
        topic_model, results_df = create_topic_model(comments_list, embeddings)
        
        # Extract topic information
        topic_info_df = extract_topic_info(topic_model)
        logger.info(f"\nFound {len(topic_info_df)} topics:")
        for idx, topic in topic_info_df.iterrows():
            logger.info(f"  {topic['topic_name']} (size: {topic['topic_size']})")
        
        # Get representative comments
        topics_array = results_df["topic"].values
        representative_comments = get_representative_comments(
            topic_model,
            comments_list,
            topics_array,
            n_per_topic=3,
        )
        
        # Assign topics to documents
        preprocessed_df = assign_topics_to_documents(
            preprocessed_df,
            topics_array,
            topic_info_df,
        )
        
        # ===== STEP 4: Sentiment Analysis =====
        logger.info("\n[4/6] SENTIMENT ANALYSIS")
        logger.info("-" * 60)
        
        sentiment_model = load_sentiment_model()
        sentiment_results = analyze_sentiment_batch(
            preprocessed_df["cleaned_comment"].tolist(),
            sentiment_model,
        )
        
        preprocessed_df = add_sentiment_to_documents(preprocessed_df, sentiment_results)
        
        # Show sentiment distribution
        sentiment_dist = preprocessed_df["sentiment"].value_counts()
        logger.info(f"\nSentiment distribution:")
        for sentiment, count in sentiment_dist.items():
            logger.info(f"  {sentiment}: {count}")
        
        # ===== STEP 5: Topic Sentiment Statistics =====
        logger.info("\n[5/6] TOPIC SENTIMENT STATISTICS")
        logger.info("-" * 60)
        
        sentiment_stats = calculate_topic_sentiment_stats(preprocessed_df)
        
        logger.info(f"\nTop 5 Most Problematic Topics (by negative ratio):")
        top_problematic = sentiment_stats.head(5)
        for idx, row in top_problematic.iterrows():
            logger.info(
                f"  {row['topic_name']}: {row['negative_ratio']:.2%} negative "
                f"({row['negative_count']}/{row['total_comments']})"
            )
        
        # Identify problematic topics
        problematic_topics = identify_problematic_topics(
            sentiment_stats,
            negative_threshold=negative_threshold,
        )
        logger.info(f"\nIdentified {len(problematic_topics)} problematic topics "
                   f"(negative_ratio >= {negative_threshold})")
        
        # ===== STEP 6: Export Results =====
        logger.info("\n[6/6] EXPORTING RESULTS")
        logger.info("-" * 60)
        
        export_paths = export_all_results(
            preprocessed_df,
            sentiment_stats,
            representative_comments,
        )
        
        # Validate exports
        if validate_exports():
            logger.info("\n✓ All results exported successfully")
        else:
            logger.warning("\n⚠ Some export files might be missing")
        
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline Completed Successfully!")
        logger.info("=" * 60)
        
        # Print summary
        logger.info("\nPipeline Summary:")
        logger.info(f"  - Input reviews: {len(raw_df)}")
        logger.info(f"  - Processed reviews: {len(preprocessed_df)}")
        logger.info(f"  - Topics discovered: {len(sentiment_stats)}")
        logger.info(f"  - Positive reviews: {(preprocessed_df['sentiment'] == 'positive').sum()}")
        logger.info(f"  - Negative reviews: {(preprocessed_df['sentiment'] == 'negative').sum()}")
        logger.info(f"  - Neutral reviews: {(preprocessed_df['sentiment'] == 'neutral').sum()}")
        
        return export_paths
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed with error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        # Run pipeline with default configuration
        export_paths = run_pipeline()
        
        logger.info("\nOutput files:")
        for key, path in export_paths.items():
            logger.info(f"  {key}: {path}")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)
