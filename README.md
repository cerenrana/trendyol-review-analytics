# Trendyol Review Analytics

Trendyol ürün yorumları için hazırlanmış Streamlit tabanlı müşteri yorum analizi projesi. Mevcut uygulama; yorum toplama, aspect bazlı sentiment analizi, semantik clustering ve basit trend çıktıları üzerine kuruludur.

## Özellikler

- Trendyol yorumlarını API üzerinden CSV olarak toplama
- Rating, yorum metni ve tarih bilgilerini işleme
- Kargo, paketleme, fiyat-performans, nemlendirme gibi aspect'leri tespit etme
- Aspect bazlı kural tabanlı sentiment analizi
- SentenceTransformer, KMeans ve KeyBERT ile semantik clustering
- Streamlit dashboard ile yorumları, aspect dağılımlarını, sentiment sonuçlarını ve cluster içeriklerini inceleme

## Proje Yapısı

```text
.
├── app.py                    # Streamlit dashboard
├── requirements.txt          # Python bağımlılıkları
├── data/
│   ├── raw/
│   │   └── reviews.csv       # Ham Trendyol yorumları
│   └── processed/
│       ├── reviews_with_sentiment.csv
│       ├── processed_reviews.csv
│       └── clustered_reviews.csv
└── src/
    ├── review_scraper.py     # Trendyol yorumlarını çeker
    ├── sentiment_analysis.py # Mevcut durumda yorum çekme script'i
    ├── aspect_sentiment.py   # Aspect ve kural tabanlı sentiment üretir
    ├── semantic_clustering.py# Aspect/sentiment bazlı cluster üretir
    └── trend_analysis.py     # Günlük trend dosyaları üretir
```

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Kullanım

### 1. Yorumları Çekme

```bash
python src/review_scraper.py
```

Bu komut Trendyol API'den yorumları çeker ve şu dosyaya kaydeder:

```text
data/raw/reviews.csv
```

`src/review_scraper.py` içindeki `contentId` ve `merchantId` değerleri belirli bir ürün için ayarlanmıştır. Farklı ürün analiz etmek için bu değerleri güncelleyin.

### 2. Aspect Sentiment Analizi

```bash
python src/aspect_sentiment.py
```

Bu script şu girdiyi bekler:

```text
data/processed/reviews_with_sentiment.csv
```

Çıktı:

```text
data/processed/processed_reviews.csv
```

### 3. Semantic Clustering

```bash
python src/semantic_clustering.py
```

Bu script `processed_reviews.csv` üzerinden aspect ve sentiment kırılımında cluster üretir.

Çıktı:

```text
data/processed/clustered_reviews.csv
```

### 4. Trend Çıktıları

```bash
python src/trend_analysis.py
```

Trend çıktıları şu klasöre yazılır:

```text
data/processed/dashboard/
```

### 5. Dashboard

```bash
streamlit run app.py
```

Streamlit varsayılan olarak şu adreste açılır:

```text
http://localhost:8501
```

Dashboard sekmeleri:

- Aspect Analysis
- Sentiment Overview
- Detailed Reviews
- Clustering

## Veri Dosyaları

`data/raw/reviews.csv` ham yorumları içerir.

`data/processed/reviews_with_sentiment.csv` yorumların genel sentiment skorlarıyla birlikte tutulduğu ara dosyadır.

`data/processed/processed_reviews.csv` aspect tespiti ve aspect sentiment kolonlarını içerir.

`data/processed/clustered_reviews.csv` semantic clustering sonuçlarını içerir.

## Notlar

- CSV dosyaları `.gitignore` ile dışlanmıştır; veri dosyalarının repoya eklenmemesi tercih edilir.
- İlk embedding modeli çalıştırması model indirme ve cache oluşturma nedeniyle yavaş olabilir.
- `semantic_clustering.py` içinde KeyBERT kullanıldığı için `keybert` bağımlılığı gereklidir.
- `trend_analysis.py` grafik üretmek için `matplotlib` kullanır.
