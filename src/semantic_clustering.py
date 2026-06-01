import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "processed_reviews.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "clustered_reviews.csv"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

MIN_COMMENTS_FOR_CLUSTERING = 5
MAX_K = 5

ASPECTS = [
    "renk_pigment",
    "koku_tat",
    "nemlendirme",
    "yapı_doku",
    "paketleme",
    "eksik_yanlış_ürün",
    "fiyat_performans",
    "kargo_teslimat"
]

SENTIMENTS = ["negative", "positive"]


def find_best_k(embeddings, max_k_limit=5):
    n = len(embeddings)

    if n < MIN_COMMENTS_FOR_CLUSTERING:
        return 1

    max_k = min(max_k_limit, n - 1)

    best_k = 2
    best_score = -1

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        score = silhouette_score(embeddings, labels)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k


TURKISH_STOPWORDS = [
    "ürün", "ürünü", "ürünün", "ürünler", "urun", "urunu",
    "rengi", "rengini", "renk", "kokusu", "koku",
    "beğendim", "begendim", "beğenmedim", "begenmedim",
    "güzel", "guzel", "çok", "cok", "biraz", "aldım", "aldim",
    "geldi", "var", "yok", "ama", "fakat", "için", "icin",
    "ben", "bana", "bence", "daha", "gibi", "olan", "oldu",
    "iyi", "kötü", "kotu", "değil", "degil"
]


def get_cluster_keywords(kw_model, comments):
    combined_text = " ".join(comments)

    keywords = kw_model.extract_keywords(
        combined_text,
        keyphrase_ngram_range=(1, 2),
        stop_words=TURKISH_STOPWORDS,
        top_n=5
    )

    return ", ".join(kw for kw, _ in keywords)


def get_representative_comments(comments, embeddings, labels, cluster_id, center):
    cluster_indexes = [i for i, label in enumerate(labels) if label == cluster_id]

    cluster_embeddings = embeddings[cluster_indexes]
    similarities = cosine_similarity([center], cluster_embeddings)[0]

    top_positions = similarities.argsort()[-3:][::-1]

    representatives = [
        comments[cluster_indexes[pos]]
        for pos in top_positions
    ]

    return " | ".join(representatives)


print(f"Loading processed reviews from: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)

if "cleaned_comment" not in df.columns:
    df["cleaned_comment"] = df["comment"].fillna("").astype(str)

df["cleaned_comment"] = df["cleaned_comment"].fillna("").astype(str)
df = df[df["cleaned_comment"].str.len() > 5].copy()

model = SentenceTransformer(MODEL_NAME)
kw_model = KeyBERT(model=MODEL_NAME)

all_clustered_dfs = []

for aspect in ASPECTS:
    aspect_sentiment_col = f"{aspect}_sentiment"

    if aspect_sentiment_col not in df.columns:
        print(f"Skipping {aspect}: column not found.")
        continue

    for sentiment in SENTIMENTS:
        subset = df[df[aspect_sentiment_col] == sentiment].copy()

        if subset.empty:
            print(f"Skipping {aspect} / {sentiment}: no comments.")
            continue

        comments = subset["cleaned_comment"].tolist()

        print(f"\nProcessing aspect={aspect}, sentiment={sentiment}, comments={len(comments)}")

        embeddings = model.encode(comments, show_progress_bar=True)

        best_k = find_best_k(embeddings, MAX_K)
        print(f"Best k for {aspect} / {sentiment}: {best_k}")

        if best_k == 1:
            labels = [0] * len(comments)
            subset["cluster"] = labels
            subset["cluster_id"] = f"{aspect}_{sentiment}_0"

            keywords = get_cluster_keywords(kw_model, comments)

            subset["cluster_keywords"] = keywords
            subset["representative_comments"] = " | ".join(comments[:3])

        else:
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)

            subset["cluster"] = labels
            subset["cluster_id"] = subset["cluster"].apply(
                lambda x: f"{aspect}_{sentiment}_{x}"
            )

            cluster_keywords_map = {}
            representative_map = {}

            for cluster_id in sorted(set(labels)):
                cluster_comments = [
                    comments[i]
                    for i, label in enumerate(labels)
                    if label == cluster_id
                ]

                cluster_keywords_map[cluster_id] = get_cluster_keywords(
                    kw_model,
                    cluster_comments
                )

                representative_map[cluster_id] = get_representative_comments(
                    comments,
                    embeddings,
                    labels,
                    cluster_id,
                    kmeans.cluster_centers_[cluster_id]
                )

                print(
                    f"Cluster {aspect}_{sentiment}_{cluster_id}: "
                    f"{cluster_keywords_map[cluster_id]}"
                )

            subset["cluster_keywords"] = subset["cluster"].map(cluster_keywords_map)
            subset["representative_comments"] = subset["cluster"].map(representative_map)

        subset["cluster_aspect"] = aspect
        subset["cluster_sentiment"] = sentiment

        all_clustered_dfs.append(subset)

if not all_clustered_dfs:
    raise ValueError("No clustered data generated.")

result_df = pd.concat(all_clustered_dfs, ignore_index=True)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(f"\nSaved clustered reviews to: {OUTPUT_PATH}")