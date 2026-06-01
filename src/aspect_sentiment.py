import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "reviews_with_sentiment.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "processed_reviews.csv"

print(f"Loading sentiment dataset from: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)

df["comment"] = df["comment"].fillna("").astype(str)

aspects = {
    "renk_pigment": [
        "renk", "rengi", "rengini", "pigment", "kırmızı", "pembe",
        "kahverengi", "tonu açık", "tonu koyu"
    ],

    "koku_tat": [
        "koku", "kokusu", "kokuyor", "tadı", "tat", "çilek"
    ],

    "nemlendirme": [
        "nem", "nemlendir", "kurut", "kuruluk", "dudak", "onarıyor",
        "çatlak", "yumuşat"
    ],

    "yapı_doku": [
        "yapı", "yapısı", "doku", "dokusu", "yapış", "gloss",
        "parlak", "parlat", "kalın", "ince", "hissiyat"
    ],

    "paketleme": [
        "paket", "paketleme", "paketlemesi", "kutu", "kutusu",
        "akmış", "akmis", "kırık", "kirik", "ezilmiş", "ezilmis",
        "hasarlı", "hasarli", "ambalaj"
    ],

    "eksik_yanlış_ürün": [
        "eksik", "boş", "bos", "yarısı", "yarisi", "gelmedi",
        "yanlış", "yanlis", "tester", "farklı ürün", "baska ürün",
        "başka ürün"
    ],

    "fiyat_performans": [
        "fiyat", "performans", "indirim", "indirimli",
        "uygun", "pahalı", "pahali", "değer", "deger"
    ],

    "kargo_teslimat": [
        "kargo", "teslimat", "kargolama", "geç geldi", "gec geldi",
        "hızlı geldi", "hizli geldi", "geç teslim", "gec teslim"
    ]
}

positive_words = [
    "güzel", "guzel", "beğendim", "begendim", "bayıldım", "bayildim",
    "harika", "mükemmel", "mukemmel", "iyi", "tatlı", "tatli",
    "sevdim", "başarılı", "basarili", "öneririm", "oneririm",
    "memnun", "süper", "super", "efsane", "hoş", "hos",
    "kaliteli", "hakediyor", "stokladım", "kullanışlı", "kullanisli",
    "nemlendiriyor", "onarıyor", "parlatması", "alin", "aldırın"
]

negative_words = [
    "kötü", "kotu", "beğenmedim", "begenmedim", "boş", "bos",
    "yanlış", "yanlis", "gelmedi", "kırık", "kirik", "akmış", "akmis",
    "eksik", "iade", "pişman", "pisman", "berbat", "asla",
    "tester", "yarısı", "yarisi", "sevmedim", "sıkıntı", "sikinti"
]

negation_positive_patterns = [
    "yapış yapış değil",
    "yapis yapis degil",
    "kötü değil",
    "kotu degil",
    "rahatsız etmiyor",
    "rahatsiz etmiyor",
    "ağır değil",
    "agir degil"
]

contrast_words = [" ama ", " fakat ", " ancak ", " lakin "]


def simple_sentiment(text):
    text = str(text).lower()
    for pattern in negation_positive_patterns:
        if pattern in text:
            return "positive"

    pos = sum(1 for w in positive_words if w in text)
    neg = sum(1 for w in negative_words if w in text)

    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def contrast_sentiment(text):
    text = str(text).lower()
    for conj in contrast_words:
        if conj in text:
            left, right = text.split(conj, 1)
            left_sent = simple_sentiment(left)
            right_sent = simple_sentiment(right)
            if right_sent != "neutral":
                return right_sent
            if left_sent != "neutral":
                return left_sent
            return "neutral"
    return simple_sentiment(text)


def detect_aspects(comment):
    comment = str(comment).lower()
    return [aspect for aspect, keywords in aspects.items() if any(keyword in comment for keyword in keywords)]


def aspect_sentiment(comment, aspect_keywords):
    comment = str(comment).lower()
    if not any(keyword in comment for keyword in aspect_keywords):
        return None
    return contrast_sentiment(comment)


def summarize_aspect_sentiments(row):
    parts = []
    for aspect in aspects.keys():
        sentiment = row.get(f"{aspect}_sentiment")
        if pd.notna(sentiment):
            parts.append(f"{aspect}:{sentiment}")
    return "; ".join(parts) if parts else None


df["detected_aspects"] = df["comment"].apply(detect_aspects)
df["overall_rule_sentiment"] = df["comment"].apply(contrast_sentiment)

for aspect, keywords in aspects.items():
    df[f"{aspect}_sentiment"] = df["comment"].apply(
        lambda x: aspect_sentiment(x, keywords)
    )

df["aspect_sentiment"] = df.apply(summarize_aspect_sentiments, axis=1)

print("Detected aspect counts:")
aspect_counts = df["detected_aspects"].explode().value_counts()
print(aspect_counts)

print("Aspect sentiment summary:")
for aspect in aspects.keys():
    col = f"{aspect}_sentiment"
    print(f"{aspect}: {df[col].value_counts(dropna=True).to_dict()}")

print("Negative review aspect distribution:")
negative_df = df[df["rating"] <= 2]
print(negative_df["detected_aspects"].explode().value_counts())

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"Saved processed review dataset to: {OUTPUT_PATH}")
