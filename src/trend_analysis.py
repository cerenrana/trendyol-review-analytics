import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "clustered_reviews.csv"
DASHBOARD_DIR = BASE_DIR / "data" / "processed" / "dashboard"

DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)


ASPECT_COLUMNS = [
    "renk_pigment_sentiment",
    "koku_tat_sentiment",
    "nemlendirme_sentiment",
    "yapı_doku_sentiment",
    "paketleme_sentiment",
    "eksik_yanlış_ürün_sentiment",
    "fiyat_performans_sentiment",
    "kargo_teslimat_sentiment"
]


def load_data(path: Path) -> pd.DataFrame:
    print(f"Loading clustered reviews from: {path}")
    return pd.read_csv(path)


def detect_timestamp_column(df: pd.DataFrame) -> str:
    possible_columns = ["created_at", "last_modified_date", "review_date"]

    for col in possible_columns:
        if col in df.columns:
            return col

    raise KeyError("No timestamp column found.")


def prepare_dates(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    if pd.api.types.is_numeric_dtype(df[timestamp_col]):
        df["review_date"] = pd.to_datetime(df[timestamp_col], unit="ms", errors="coerce")
    else:
        df["review_date"] = pd.to_datetime(df[timestamp_col], errors="coerce")

    df = df[df["review_date"].notna()].copy()
    df["review_day"] = df["review_date"].dt.date

    return df


def save_rating_trends(df: pd.DataFrame):
    rating_trend = (
        df.groupby("review_day")["rating"]
        .mean()
        .reset_index()
    )

    rating_trend.to_csv(DASHBOARD_DIR / "daily_avg_rating.csv", index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(rating_trend["review_day"], rating_trend["rating"], marker="o")
    plt.title("Daily Average Rating")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(DASHBOARD_DIR / "daily_avg_rating.png")
    plt.close()

    return rating_trend


def save_review_volume(df: pd.DataFrame):
    review_volume = (
        df.groupby("review_day")
        .size()
        .reset_index(name="review_count")
    )

    review_volume.to_csv(DASHBOARD_DIR / "daily_review_volume.csv", index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(review_volume["review_day"], review_volume["review_count"], marker="o")
    plt.title("Daily Review Volume")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(DASHBOARD_DIR / "daily_review_volume.png")
    plt.close()

    return review_volume


def save_aspect_negative_trends(df: pd.DataFrame):
    negative_df = df[df["rating"] <= 2].copy()
    trend_frames = []

    existing_aspect_columns = [col for col in ASPECT_COLUMNS if col in df.columns]

    for col in existing_aspect_columns:
        aspect_name = col.replace("_sentiment", "")

        aspect_negative = negative_df[negative_df[col] == "negative"]

        daily_counts = (
            aspect_negative
            .groupby("review_day")
            .size()
            .reset_index(name="count")
        )

        daily_counts["aspect"] = aspect_name
        trend_frames.append(daily_counts)

        daily_counts.to_csv(
            DASHBOARD_DIR / f"aspect_negative_trend_{aspect_name}.csv",
            index=False
        )

        print(f"Saved aspect trend for {aspect_name}")

    if not trend_frames:
        print("No aspect sentiment columns found.")
        return None

    aspect_trend_df = pd.concat(trend_frames, ignore_index=True)
    aspect_trend_df.to_csv(DASHBOARD_DIR / "aspect_negative_trends.csv", index=False)

    aspect_summary = (
        aspect_trend_df.groupby("aspect")["count"]
        .sum()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 5))
    aspect_summary.plot(kind="bar")
    plt.title("Negative Aspect Mentions")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(DASHBOARD_DIR / "aspect_negative_summary.png")
    plt.close()

    return aspect_trend_df


def save_cluster_outputs(df: pd.DataFrame):
    if "cluster_id" not in df.columns:
        print("No cluster_id column found.")
        return

    cluster_distribution = (
        df.groupby("cluster_id")
        .size()
        .reset_index(name="review_count")
    )

    cluster_distribution.to_csv(
        DASHBOARD_DIR / "cluster_distribution.csv",
        index=False
    )

    if "cluster_keywords" in df.columns:
        cluster_keywords = (
            df[["cluster_id", "cluster_keywords"]]
            .drop_duplicates()
        )

        cluster_keywords.to_csv(
            DASHBOARD_DIR / "cluster_keywords.csv",
            index=False
        )


def save_html_summary():
    html_path = DASHBOARD_DIR / "dashboard.html"

    with open(html_path, "w", encoding="utf-8") as html_file:
        html_file.write("""
        <html>
        <head>
            <meta charset="utf-8">
            <title>Review Dashboard</title>
        </head>
        <body>
            <h1>Review Dashboard</h1>

            <h2>Time Series</h2>
            <ul>
                <li><a href="daily_avg_rating.csv">Daily avg rating</a></li>
                <li><a href="daily_review_volume.csv">Daily review volume</a></li>
            </ul>

            <h2>Graphs</h2>
            <img src="daily_avg_rating.png" style="max-width:100%;">
            <img src="daily_review_volume.png" style="max-width:100%;">

            <h2>Clusters</h2>
            <ul>
                <li><a href="cluster_distribution.csv">Cluster distribution</a></li>
                <li><a href="cluster_keywords.csv">Cluster keywords</a></li>
            </ul>

            <h2>Aspect Negative Trends</h2>
            <ul>
                <li><a href="aspect_negative_trends.csv">Aspect negative trends</a></li>
            </ul>

            <img src="aspect_negative_summary.png" style="max-width:100%;">
        </body>
        </html>
        """)

    print(f"Saved dashboard HTML to: {html_path}")


def main():
    df = load_data(INPUT_PATH)

    timestamp_col = detect_timestamp_column(df)
    print(f"Using timestamp column: {timestamp_col}")

    df = prepare_dates(df, timestamp_col)

    save_rating_trends(df)
    save_review_volume(df)
    save_aspect_negative_trends(df)
    save_cluster_outputs(df)
    save_html_summary()

    print(f"Saved dashboard files to: {DASHBOARD_DIR}")


if __name__ == "__main__":
    main()