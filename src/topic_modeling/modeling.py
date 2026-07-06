"""Topic modeling module using BERTopic with UMAP and HDBSCAN"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, List, Dict, Optional
from pathlib import Path

from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic

from .config import (
    EMBEDDING_MODEL,
    UMAP_N_NEIGHBORS,
    UMAP_N_COMPONENTS,
    UMAP_MIN_DIST,
    UMAP_METRIC,
    UMAP_RANDOM_STATE,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    HDBSCAN_METRIC,
    HDBSCAN_CLUSTER_SELECTION_METHOD,
    BERTOPIC_TOP_N_WORDS,
    BERTOPIC_MIN_TOPIC_SIZE,
    DEVICE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_embeddings(
    texts: List[str],
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Generate embeddings using SentenceTransformer.
    
    Args:
        texts: List of texts to embed
        model_name: SentenceTransformer model name
        batch_size: Batch size for encoding
    
    Returns:
        Embedding matrix (n_samples, embedding_dim)
    """
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device=DEVICE)
    
    logger.info(f"Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    
    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def reduce_dimensions(embeddings: np.ndarray) -> np.ndarray:
    """
    Reduce embedding dimensionality using UMAP.
    
    Args:
        embeddings: Input embeddings (n_samples, embedding_dim)
    
    Returns:
        Reduced embeddings (n_samples, n_components)
    """
    logger.info("Reducing dimensionality with UMAP...")
    umap_model = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
    )
    
    reduced_embeddings = umap_model.fit_transform(embeddings)
    logger.info(f"Reduced embeddings shape: {reduced_embeddings.shape}")
    
    return reduced_embeddings


def cluster_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Cluster embeddings using HDBSCAN.
    
    Args:
        embeddings: Input embeddings (n_samples, embedding_dim)
    
    Returns:
        Cluster labels (-1 for noise)
    """
    logger.info("Clustering with HDBSCAN...")
    clusterer = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
    )
    
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    logger.info(f"Found {n_clusters} clusters, {n_noise} noise points")
    
    return labels


def create_topic_model(
    documents: List[str],
    embeddings: np.ndarray,
) -> Tuple[BERTopic, pd.DataFrame]:
    """
    Create BERTopic model with preprocessed embeddings.
    
    Args:
        documents: List of original documents
        embeddings: Precomputed embeddings
    
    Returns:
        Tuple of (BERTopic model, results DataFrame)
    """
    logger.info("Creating BERTopic model...")
    
    # Create UMAP model for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
    )
    
    # Create HDBSCAN model for clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
    )
    
    # Initialize BERTopic
    topic_model = BERTopic(
        embedding_model=EMBEDDING_MODEL,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        top_n_words=BERTOPIC_TOP_N_WORDS,
        min_topic_size=BERTOPIC_MIN_TOPIC_SIZE,
        language="turkish",
        verbose=True,
    )
    
    logger.info("Fitting BERTopic model...")
    topics, probs = topic_model.fit_transform(
        documents,
        embeddings=embeddings,
    )
    
    logger.info(f"Number of topics: {len(set(topics)) - (1 if -1 in topics else 0)}")
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        "document": documents,
        "topic": topics,
        "probability": probs,
    })
    
    return topic_model, results_df


def extract_topic_info(topic_model: BERTopic) -> pd.DataFrame:
    """
    Extract detailed information about each topic.
    
    Args:
        topic_model: Fitted BERTopic model
    
    Returns:
        DataFrame with topic information
    """
    logger.info("Extracting topic information...")
    
    topic_info_list = []
    topics = topic_model.get_topic_info()
    
    for idx, row in topics.iterrows():
        topic_id = row["Topic"]
        if topic_id == -1:
            continue  # Skip noise cluster
        
        # Get topic keywords
        words = topic_model.get_topic(topic_id)
        keywords = [word for word, _ in words]
        
        topic_info_list.append({
            "topic_id": topic_id,
            "topic_name": f"Topic {topic_id}: {', '.join(keywords[:3])}",
            "topic_size": row["Count"],
            "keywords": ", ".join(keywords[:BERTOPIC_TOP_N_WORDS]),
        })
    
    topic_info_df = pd.DataFrame(topic_info_list)
    logger.info(f"Extracted info for {len(topic_info_df)} topics")
    
    return topic_info_df


def get_representative_comments(
    topic_model: BERTopic,
    documents: List[str],
    topics: np.ndarray,
    n_per_topic: int = 3,
) -> Dict[int, List[str]]:
    """
    Get representative comments for each topic.
    
    Args:
        topic_model: Fitted BERTopic model
        documents: List of original documents
        topics: Topic assignments for each document
        n_per_topic: Number of representative documents per topic
    
    Returns:
        Dictionary mapping topic_id to list of representative comments
    """
    logger.info(f"Extracting {n_per_topic} representative comments per topic...")
    
    representative_comments = {}
    
    # Get unique topics (excluding -1 which is noise)
    unique_topics = [t for t in set(topics) if t != -1]
    
    for topic_id in unique_topics:
        # Get indices where this topic appears
        topic_indices = np.where(topics == topic_id)[0]
        
        # Get the first n_per_topic documents for this topic
        representative_indices = topic_indices[:n_per_topic]
        representative_docs = [documents[idx] for idx in representative_indices]
        representative_comments[topic_id] = representative_docs
    
    return representative_comments


def assign_topics_to_documents(
    df: pd.DataFrame,
    topics: np.ndarray,
    topic_info_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign topics and topic names to documents.
    
    Args:
        df: DataFrame with cleaned comments
        topics: Topic assignments
        topic_info_df: Topic information DataFrame
    
    Returns:
        DataFrame with topic columns added
    """
    df = df.copy()
    df["topic_id"] = topics
    
    # Map topic names
    topic_name_map = dict(zip(topic_info_df["topic_id"], topic_info_df["topic_name"]))
    topic_keywords_map = dict(zip(topic_info_df["topic_id"], topic_info_df["keywords"]))
    
    df["topic_name"] = df["topic_id"].map(topic_name_map)
    df["topic_keywords"] = df["topic_id"].map(topic_keywords_map)
    
    # Filter out noise points (topic_id == -1)
    df = df[df["topic_id"] != -1]
    
    logger.info(f"Assigned topics to {len(df)} documents")
    
    return df
