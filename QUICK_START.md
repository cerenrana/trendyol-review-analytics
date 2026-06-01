# 
## 
- **Python:** 3.8+
- **RAM:** En az 4GB (8GB+ nnnnnnerilir)
- **Disk:** 2GB+ (model indirmeleri iin)

 Adm 1: Kurulum (5 dakika)

```bash
cd /Users/cerenranaozdemir/trendyol-review-analytics

            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }kle
pip install -r requirements.txt
```

## 
```bash
python3 -m src.topic_modeling.run_topic_modeling
```

**Beklenen kt:**
```
============================================================
Pipeline Completed Successfully!
============================================================

Pipeline Summary:
  - Input reviews: 1093
  - Processed reviews: 1066
  - Topics discovered: 9
  - Positive reviews: 567
  - Negative reviews: 499
```

## 
```bash
streamlit run app.py
```

 http://localhost:8501

## 
Pipeline baarl olursa u dosyalar oluacak:

```
data/processed/
 topic_model_reviews.csv      # 1066 yorum (her birine topic+sentiment)
 topic_summary.csv            # 9 topic istatistikleri  
 topic_summary.json           # JSON format zzzzzzet
```

## 
`src/topic_modeling/config.py` dosyasnda:

```python
# CPU vs GPU
DEVICE = "cpu"                    # Mac'te "cpu", CUDA GPU'sunda "cuda"

# Daha fazla topic istiyorsan
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }k = daha fazla topic)

# Daha az topic istiyorsan  
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }ksek = daha az topic)

# Stopword'leri skip et
# altrrken: run_pipeline(remove_stopwords=False)
```

## 
| Sekme | erik |
|-------|--------|
| | | | 
 Hata Giderme## 

### "ModuleNotFoundError: No module named 'umap'"
```bash
pip install umap-learn hdbscan bertopic
```

### "Torch not compiled with CUDA enabled"
Sorun yok! Zaten `config.py` MAC'te CPU'ya ayarland.

### "BERTopic model very slow on first run"
Modeller HuggingFace'ten indiriliyordur. Sonraki altrmalar ok hzl olacak.

### "No data in dashboard"
Pipeline' bir kez altrp output dosyalarnn olutuunu kontrol et:
```bash
ls -lh data/processed/topic_*
```

## 
**Pipeline bileenleri:**
- - - - - 
---

            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }m dosyalar ready! Dashboard' amak iin:** `streamlit run app.py`
