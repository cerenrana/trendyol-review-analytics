"""Configuration for Topic Modeling Pipeline"""

from pathlib import Path
from typing import List

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Input/Output file paths
INPUT_CSV = PROCESSED_DATA_DIR / "processed_reviews.csv"
OUTPUT_REVIEWS_CSV = PROCESSED_DATA_DIR / "topic_model_reviews.csv"
OUTPUT_SUMMARY_CSV = PROCESSED_DATA_DIR / "topic_summary.csv"
OUTPUT_SUMMARY_JSON = PROCESSED_DATA_DIR / "topic_summary.json"

# Model configurations
# Embedding model: Multilingual support for Turkish, lightweight
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Sentiment model: Turkish-specific BERT sentiment classifier
SENTIMENT_MODEL = "savasy/bert-base-turkish-sentiment-cased"
SENTIMENT_MODEL_MAX_LENGTH = 512

# UMAP parameters for dimensionality reduction
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 5
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42

# HDBSCAN parameters for clustering
HDBSCAN_MIN_CLUSTER_SIZE = 15
HDBSCAN_MIN_SAMPLES = 5
HDBSCAN_METRIC = "euclidean"
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"

# BERTopic parameters
BERTOPIC_LANGUAGE = "turkish"
BERTOPIC_TOP_N_WORDS = 10
BERTOPIC_MIN_TOPIC_SIZE = 10

# Text preprocessing
MIN_COMMENT_LENGTH = 5  # Minimum character length
MAX_COMMENT_LENGTH = 2000

# Turkish stopwords (common Turkish words to exclude)
TURKISH_STOPWORDS: List[str] = [
    "ve", "bir", "bu", "için", "olmak", "var", "yok", "de", "da", "ben", "sen",
    "o", "biz", "siz", "onlar", "ben", "ama", "ancak", "fakat", "lakin", "ise",
    "ile", "dari", "yor", "yet", "dir", "dır", "dur", "dür", "miş", "muş",
    "mi", "mı", "mu", "mü", "ne", "nedir", "nasıl", "niye", "niçin", "ne zaman",
    "kaç", "kim", "kimi", "kimse", "hiç", "her", "herkes", "hepsi", "tüm",
    "bütün", "birçok", "birkaç", "bazı", "çok", "az", "daha", "en", "pek",
    "sadece", "ancak", "yalnız", "elbette", "tabii", "doğal", "evet", "hayır",
    "oluyor", "oluş", "olacak", "oldu", "olmuş", "olan", "olmaz", "olmadı",
    "yap", "yapı", "yapar", "yaptu", "yapmak", "yapm", "yaptı", "yapıyor",
    "git", "giti", "gidip", "gitmiş", "gitmek", "gid", "gidi", "gidin",
    "al", "aldı", "alıyor", "alır", "almak", "alındı", "aldığı", "alan",
    "ver", "verdi", "veriyor", "verir", "vermek", "verildi", "verilen",
    "bak", "bakı", "bakıyor", "bakar", "bakmak", "bakıl", "bakıldı", "bakılan",
    "gör", "gördü", "görmek", "görüyor", "görür", "görünmek", "göründü",
    "kat", "katı", "katıyor", "katılı", "katılan", "katılmak",
    "söy", "söyledi", "söylemek", "söylenmi", "söylenmiş", "söyleniyor",
    "koş", "koştu", "koşmak", "koşuyor", "koşan", "koşulan",
    "kalk", "kalktu", "kalkmak", "kalkıyor", "kalkılan", "kalkıldı",
    "ek", "ekle", "ekledi", "eklemek", "eklenen", "ekleniyor",
]

# Device configuration (auto-detect GPU if available)
DEVICE = "cpu"  # Change to "cuda" if GPU available (not on Mac)

# Processing batch size for sentiment analysis
SENTIMENT_BATCH_SIZE = 32

# Sentiment label mapping (for Turkish BERT model)
SENTIMENT_LABELS = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# Dashboard configuration
STREAMLIT_THEME = "light"
DEFAULT_TOP_N_TOPICS = 10
DEFAULT_TOP_N_KEYWORDS = 8
