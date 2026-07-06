# Trendyol Review Analytics

Trendyol ürün yorumlarını toplayan, temizleyen ve NLP teknikleriyle analiz eden bir Python projesi. Proje iki ana analiz hattı içerir:

- Topic modeling ve sentiment analizi: Yorumları BERTopic ile konu kümelerine ayırır, Türkçe BERT sentiment modeliyle pozitif, negatif ve nötr sınıflandırması yapar.
- Legacy aspect analizi: Kargo, paketleme, fiyat-performans, nemlendirme gibi önceden tanımlı aspect'ler için kural tabanlı sentiment ve semantik clustering çıktıları üretir.

Streamlit dashboard üzerinden topic dağılımı, sentiment oranları, problemli topic'ler, yorum arama ve çıktı indirme akışları incelenebilir.

## Proje Yapısı

```text
.
├── app.py                         # Topic modeling ve sentiment dashboard'u
├── app_legacy.py                  # Eski aspect/sentiment dashboard'u
├── requirements.txt               # Python bağımlılıkları
├── data/
│   ├── raw/reviews.csv            # Trendyol'dan çekilen ham yorumlar
│   └── processed/                 # Analiz çıktıları
├── src/
│   ├── review_scraper.py          # Trendyol yorumlarını çeker
│   ├── aspect_sentiment.py        # Kural tabanlı aspect sentiment analizi
│   ├── semantic_clustering.py     # Aspect bazlı semantik clustering
│   ├── trend_analysis.py          # Günlük trend çıktıları üretir
│   └── topic_modeling/
│       ├── config.py              # Model, dosya yolu ve pipeline ayarları
│       ├── preprocessing.py       # Veri yükleme, temizleme, stopword çıkarımı
│       ├── modeling.py            # Embedding, UMAP, HDBSCAN, BERTopic
│       ├── sentiment.py           # Türkçe sentiment modeli
│       ├── export.py              # CSV ve JSON export işlemleri
│       └── run_topic_modeling.py  # Ana pipeline orkestrasyonu
└── QUICK_START.md                 # Hızlı başlangıç notları
```

## Kurulum

Python 3.8 veya üzeri önerilir. İlk model indirmelerinde internet bağlantısı gerekir; embedding ve sentiment modelleri Hugging Face üzerinden indirilir.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Veri Hazırlama

Projede varsayılan topic modeling girdisi:

```text
data/processed/processed_reviews.csv
```

Bu dosyada en az `comment` veya `cleaned_comment` kolonu bulunmalıdır. Mevcut veri akışı şu şekilde çalışır:

```bash
python src/review_scraper.py
python src/sentiment_analysis.py
python src/aspect_sentiment.py
```

`review_scraper.py` içindeki `contentId` ve `merchantId` değerleri belirli bir Trendyol ürünü için ayarlanmıştır. Farklı ürün analiz edilecekse bu parametreler güncellenmelidir.

## Topic Modeling Pipeline

Ana topic modeling ve sentiment pipeline'ını çalıştırmak için:

```bash
python -m src.topic_modeling.run_topic_modeling
```

Pipeline şu adımları uygular:

1. CSV verisini yükler.
2. Yorumları temizler ve isteğe bağlı olarak Türkçe stopword'leri çıkarır.
3. `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` modeliyle embedding üretir.
4. UMAP ve HDBSCAN ile kümeleri çıkarır.
5. BERTopic ile topic isimleri ve anahtar kelimeler üretir.
6. `savasy/bert-base-turkish-sentiment-cased` modeliyle sentiment analizi yapar.
7. Sonuçları CSV ve JSON olarak dışa aktarır.

Beklenen çıktı dosyaları:

```text
data/processed/topic_model_reviews.csv
data/processed/topic_summary.csv
data/processed/topic_summary.json
```

Mevcut örnek çıktılarda yaklaşık 1.066 işlenmiş yorum ve 9 topic bulunmaktadır.

## Dashboard

Topic modeling dashboard'unu başlatmak için:

```bash
streamlit run app.py
```

Streamlit varsayılan olarak şu adreste açılır:

```text
http://localhost:8501
```

Dashboard sekmeleri:

- Genel İstatistikler: Topic bazlı sentiment dağılımı, en negatif topic'ler ve topic boyutları.
- Topic Detayları: Seçilen topic için anahtar kelimeler, oranlar ve temsili yorumlar.
- Yorum Arama: Yorum içeriği ve sentiment'e göre filtreleme.
- Verileri İndir: CSV ve JSON çıktılarını indirme.

Legacy dashboard için:

```bash
streamlit run app_legacy.py
```

## Çıktı Dosyaları

### `topic_model_reviews.csv`

Her yorum için topic ve sentiment bilgisini içerir.

Temel kolonlar:

- `original_comment`
- `cleaned_comment`
- `topic_id`
- `topic_name`
- `topic_keywords`
- `sentiment`
- `sentiment_score`

### `topic_summary.csv`

Topic seviyesinde özet istatistikleri içerir.

Temel kolonlar:

- `topic_id`
- `topic_name`
- `topic_size`
- `keywords`
- `representative_comments`
- `positive_count`
- `negative_count`
- `neutral_count`
- `positive_ratio`
- `negative_ratio`
- `neutral_ratio`
- `avg_sentiment_score`

### `topic_summary.json`

Dashboard veya farklı servislerde kullanılabilecek JSON formatlı topic özetidir.

## Ayarlar

Topic modeling ayarları [src/topic_modeling/config.py](src/topic_modeling/config.py) dosyasından değiştirilebilir.

Sık değiştirilen değerler:

```python
DEVICE = "cpu"
HDBSCAN_MIN_CLUSTER_SIZE = 15
HDBSCAN_MIN_SAMPLES = 5
BERTOPIC_TOP_N_WORDS = 10
SENTIMENT_BATCH_SIZE = 32
DEFAULT_TOP_N_TOPICS = 10
```

Topic sayısı çok azsa `HDBSCAN_MIN_CLUSTER_SIZE` düşürülebilir. Topic sayısı çok fazlaysa bu değer artırılabilir.

## Sorun Giderme

### Analiz dosyaları bulunamadı

Dashboard açıldığında çıktı dosyaları bulunamazsa önce pipeline'ı çalıştırın:

```bash
python -m src.topic_modeling.run_topic_modeling
```

### `ModuleNotFoundError: No module named 'umap'`

Bağımlılıkları tekrar kurun:

```bash
pip install -r requirements.txt
```

### İlk çalıştırma yavaş

İlk çalıştırmada Hugging Face modelleri indirildiği için pipeline yavaş olabilir. Modeller yerel cache'e alındıktan sonra sonraki çalıştırmalar daha hızlı olur.

### CUDA hatası

Mac veya GPU olmayan makinelerde `DEVICE = "cpu"` olarak kalmalıdır.

## Geliştirme Notları

Bu projede iyileştirilebilecek başlıca alanlar:

- `QUICK_START.md` ve `test_data.py` dosyalarında karakter kodlama bozulmaları var; Türkçe metinler temizlenmeli.
- Script'lerde sabit ürün parametreleri ve dosya yolları var; CLI argümanları veya `.env` ayarlarıyla yönetilebilir.
- Topic pipeline için küçük bir test seti ve otomatik smoke test eklenebilir.
- Büyük CSV dosyaları `.gitignore` içinde dışlanmış olsa da mevcut takip durumu düzenli kontrol edilmeli; veri gizliliği için ham yorumların repoya eklenmemesi tercih edilebilir.
- `app.py` içinde bazı kolonların varlığı varsayılıyor; eksik kolon durumları için kullanıcı dostu hata mesajları eklenebilir.
- Topic özetinde `keywords` alanı sentiment istatistiklerine taşınmadığı için bazı export'larda boş kalabilir; topic bilgisiyle merge edilerek düzeltilebilir.

## Lisans ve Veri Kullanımı

Bu proje eğitim ve analiz amaçlıdır. Trendyol verisi kullanılırken platform kullanım koşulları, kişisel veri gizliliği ve rate limit kuralları dikkate alınmalıdır.
