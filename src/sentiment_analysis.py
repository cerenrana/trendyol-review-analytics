import requests
import pandas as pd
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://apigw.trendyol.com/discovery-storefront-trproductgw-service/api/review-read/product-reviews/images"

params = {
    "channelId": 1,
    "contentId": 870669244,
    "merchantId": 891275,
    "page": 0
}

rows = []

while True:
    print(f"Page: {params['page']}")
    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        raise RuntimeError(f"Review API returned status {response.status_code}")

    data = response.json()
    reviews = data.get("content", [])

    if not reviews:
        print("No more reviews.")
        break

    for r in reviews:
        rows.append({
            "review_id": r.get("reviewId"),
            "product_id": r.get("contentId", params.get("contentId")),
            "rating": r.get("rate"),
            "comment": r.get("comment"),
            "created_at": r.get("lastModifiedDate")
        })

    params["page"] += 1
    time.sleep(1)


df = pd.DataFrame(rows)
df = df.drop_duplicates(subset=["review_id"])
output_path = RAW_DIR / "reviews.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"Saved raw review dataset to: {output_path}")

