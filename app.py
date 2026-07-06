"""Streamlit Dashboard for Topic Modeling and Sentiment Analysis Results"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import logging

from src.topic_modeling.config import (
    OUTPUT_REVIEWS_CSV,
    OUTPUT_SUMMARY_CSV,
    OUTPUT_SUMMARY_JSON,
    DEFAULT_TOP_N_TOPICS,
    DEFAULT_TOP_N_KEYWORDS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Müşteri Yorum Analizi - Topic & Sentiment",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .topic-header {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_reviews_data():
    """Load reviews with topics and sentiments."""
    if not OUTPUT_REVIEWS_CSV.exists():
        return None
    try:
        return pd.read_csv(OUTPUT_REVIEWS_CSV)
    except Exception as e:
        logger.error(f"Error loading reviews: {str(e)}")
        return None


@st.cache_data
def load_summary_data():
    """Load topic summary."""
    if not OUTPUT_SUMMARY_CSV.exists():
        return None
    try:
        return pd.read_csv(OUTPUT_SUMMARY_CSV)
    except Exception as e:
        logger.error(f"Error loading summary: {str(e)}")
        return None


@st.cache_data
def load_json_summary():
    """Load JSON summary for detailed information."""
    if not OUTPUT_SUMMARY_JSON.exists():
        return None
    try:
        with open(OUTPUT_SUMMARY_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON summary: {str(e)}")
        return None


def create_sentiment_distribution_chart(summary_df):
    """Create sentiment distribution chart."""
    # Prepare data for stacked bar chart
    chart_data = summary_df[[
        "topic_name",
        "positive_count",
        "negative_count",
        "neutral_count",
    ]].copy()
    
    fig = go.Figure(data=[
        go.Bar(name="Pozitif", x=chart_data["topic_name"], y=chart_data["positive_count"], marker_color="#2ecc71"),
        go.Bar(name="Negatif", x=chart_data["topic_name"], y=chart_data["negative_count"], marker_color="#e74c3c"),
        go.Bar(name="Nötr", x=chart_data["topic_name"], y=chart_data["neutral_count"], marker_color="#95a5a6"),
    ])
    
    fig.update_layout(
        barmode="stack",
        title="Topic'ler Bazında Sentiment Dağılımı",
        xaxis_title="Topic",
        yaxis_title="Yorum Sayısı",
        hovermode="x unified",
        height=500,
    )
    
    return fig


def create_negative_ratio_chart(summary_df):
    """Create chart showing topics by negative ratio."""
    top_negative = summary_df.nlargest(DEFAULT_TOP_N_TOPICS, "negative_ratio")
    
    fig = px.bar(
        top_negative,
        x="negative_ratio",
        y="topic_name",
        orientation="h",
        title=f"En Negatif {min(DEFAULT_TOP_N_TOPICS, len(top_negative))} Topic",
        labels={"negative_ratio": "Negatif Oran", "topic_name": "Topic"},
        color="negative_ratio",
        color_continuous_scale="Reds",
    )
    
    fig.update_layout(height=400, xaxis_tickformat=".0%")
    return fig


def create_topic_size_chart(summary_df):
    """Create chart showing topic sizes."""
    top_topics = summary_df.nlargest(DEFAULT_TOP_N_TOPICS, "topic_size")
    
    fig = px.pie(
        top_topics,
        values="topic_size",
        names="topic_name",
        title=f"En Büyük {min(DEFAULT_TOP_N_TOPICS, len(top_topics))} Topic",
    )
    
    fig.update_layout(height=500)
    return fig


def display_top_metrics(summary_df, reviews_df):
    """Display top-level metrics."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📊 Toplam Topic", len(summary_df))
    
    with col2:
        st.metric("💬 Toplam Yorum", len(reviews_df))
    
    with col3:
        positive_pct = (reviews_df["sentiment"] == "positive").sum() / len(reviews_df) * 100
        st.metric("😊 Pozitif %", f"{positive_pct:.1f}%")
    
    with col4:
        negative_pct = (reviews_df["sentiment"] == "negative").sum() / len(reviews_df) * 100
        st.metric("😞 Negatif %", f"{negative_pct:.1f}%")
    
    with col5:
        avg_topic_size = len(reviews_df) / len(summary_df)
        st.metric("📈 Ort. Topic Boyutu", f"{avg_topic_size:.0f}")


def display_topic_details(reviews_df, summary_df, json_summary):
    """Display detailed information about selected topic."""
    st.subheader("🔍 Topic Detayları")
    
    selected_topic = st.selectbox(
        "Topic Seç",
        options=summary_df["topic_name"].tolist(),
        key="topic_select",
    )
    
    if selected_topic:
        topic_row = summary_df[summary_df["topic_name"] == selected_topic].iloc[0]
        topic_id = topic_row["topic_id"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Topic ID:** {topic_id}")
            st.write(f"**Topic Adı:** {topic_row['topic_name']}")
            st.write(f"**Toplam Yorum:** {topic_row['topic_size']}")
            st.write(f"**Keywords:** {topic_row['keywords']}")
        
        with col2:
            st.write(f"**Pozitif:** {topic_row['positive_count']} ({topic_row['positive_ratio']:.1%})")
            st.write(f"**Negatif:** {topic_row['negative_count']} ({topic_row['negative_ratio']:.1%})")
            st.write(f"**Nötr:** {topic_row['neutral_count']} ({topic_row['neutral_ratio']:.1%})")
            st.write(f"**Ort. Sentiment Score:** {topic_row['avg_sentiment_score']:.3f}")
        
        # Show representative comments
        st.write("**Temsili Yorumlar:**")
        if topic_row['representative_comments']:
            comments = str(topic_row['representative_comments']).split(" | ")
            for i, comment in enumerate(comments, 1):
                if comment.strip():
                    st.write(f"{i}. {comment}")
        
        # Show topic comments
        topic_comments = reviews_df[reviews_df["topic_id"] == topic_id]
        
        st.write(f"**Toplamda {len(topic_comments)} yorum**")
        
        # Filter by sentiment
        sentiment_filter = st.selectbox(
            "Sentiment Filtresi",
            options=["Tümü", "Pozitif", "Negatif", "Nötr"],
            key=f"sentiment_filter_{topic_id}",
        )
        
        if sentiment_filter != "Tümü":
            sentiment_map = {"Pozitif": "positive", "Negatif": "negative", "Nötr": "neutral"}
            topic_comments = topic_comments[
                topic_comments["sentiment"] == sentiment_map[sentiment_filter]
            ]
        
        # Display comments in expandable sections
        if len(topic_comments) > 0:
            st.write(f"**Filtrelenmiş Yorum Sayısı: {len(topic_comments)}**")
            
            for idx, row in topic_comments.head(10).iterrows():
                with st.expander(
                    f"💬 {row['sentiment'].upper()} - {row['cleaned_comment'][:60]}..."
                ):
                    st.write(f"**Original:** {row['original_comment']}")
                    st.write(f"**Cleaned:** {row['cleaned_comment']}")
                    st.write(f"**Sentiment:** {row['sentiment']} (Score: {row['sentiment_score']:.3f})")


def display_search_and_filter(reviews_df):
    """Display search and filter interface."""
    st.subheader("🔎 Yorum Ara ve Filtrele")
    
    col1, col2 = st.columns(2)
    
    with col1:
        search_term = st.text_input("Yorum İçeriğinde Ara", placeholder="Örn: kalite, fiyat, kargo...")
    
    with col2:
        sentiment_filter = st.multiselect(
            "Sentiment Filtresi",
            options=["positive", "negative", "neutral"],
            default=["positive", "negative", "neutral"],
        )
    
    # Apply filters
    filtered_df = reviews_df.copy()
    
    if search_term:
        filtered_df = filtered_df[
            filtered_df["cleaned_comment"].str.contains(search_term.lower(), na=False)
        ]
    
    if sentiment_filter:
        filtered_df = filtered_df[filtered_df["sentiment"].isin(sentiment_filter)]
    
    # Display results
    st.write(f"**Toplam Sonuç: {len(filtered_df)}**")
    
    if len(filtered_df) > 0:
        # Create display dataframe
        display_df = filtered_df[[
            "original_comment",
            "topic_name",
            "sentiment",
            "sentiment_score",
        ]].copy()
        
        display_df.columns = ["Yorum", "Topic", "Sentiment", "Score"]
        
        st.dataframe(display_df.head(20), use_container_width=True, hide_index=True)
    else:
        st.info("Sonuç bulunamadı.")


def main():
    """Main dashboard function."""
    st.title("📊 Türkçe Müşteri Yorum Analizi - NLP Topic Modeling")
    st.markdown("---")
    
    # Load data
    reviews_df = load_reviews_data()
    summary_df = load_summary_data()
    json_summary = load_json_summary()
    
    # Check if data is available
    if reviews_df is None or summary_df is None:
        st.error("❌ Analiz dosyaları bulunamadı!")
        st.info("""
        Pipeline'ı çalıştırmak için:
        ```bash
        python -m src.topic_modeling.run_topic_modeling
        ```
        """)
        return
    
    st.success(f"✅ {len(summary_df)} topic ve {len(reviews_df)} yorum başarıyla yüklendi!")
    
    # Display top metrics
    display_top_metrics(summary_df, reviews_df)
    
    st.markdown("---")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Genel İstatistikler",
        "🔍 Topic Detayları",
        "🔎 Yorum Arama",
        "📋 Verileri İndir",
    ])
    
    with tab1:
        st.subheader("📊 Sentiment Dağılımı")
        fig1 = create_sentiment_distribution_chart(summary_df)
        st.plotly_chart(fig1, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔴 En Negatif Topic'ler")
            fig2 = create_negative_ratio_chart(summary_df)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            st.subheader("📊 Topic Boyutları")
            fig3 = create_topic_size_chart(summary_df)
            st.plotly_chart(fig3, use_container_width=True)
        
        # Show summary table
        st.subheader("📋 Tüm Topic'ler")
        display_summary = summary_df[[
            "topic_id",
            "topic_name",
            "topic_size",
            "positive_ratio",
            "negative_ratio",
            "neutral_ratio",
            "avg_sentiment_score",
        ]].copy()
        
        display_summary.columns = [
            "ID", "Topic", "Boyut", "Pos %", "Neg %", "Ntr %", "Ort. Score"
        ]
        
        st.dataframe(display_summary, use_container_width=True, hide_index=True)
    
    with tab2:
        display_topic_details(reviews_df, summary_df, json_summary)
    
    with tab3:
        display_search_and_filter(reviews_df)
    
    with tab4:
        st.subheader("📥 Sonuçları İndir")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_summary = summary_df.to_csv(index=False, encoding="utf-8")
            st.download_button(
                label="📊 Topic Özeti (CSV)",
                data=csv_summary,
                file_name="topic_summary.csv",
                mime="text/csv",
            )
        
        with col2:
            csv_reviews = reviews_df.to_csv(index=False, encoding="utf-8")
            st.download_button(
                label="💬 Tüm Yorumlar (CSV)",
                data=csv_reviews,
                file_name="topic_model_reviews.csv",
                mime="text/csv",
            )
        
        with col3:
            if json_summary:
                json_str = json.dumps(json_summary, ensure_ascii=False, indent=2)
                st.download_button(
                    label="📄 Özet (JSON)",
                    data=json_str,
                    file_name="topic_summary.json",
                    mime="application/json",
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Pipeline Bileşenleri:**
    - 🔤 SentenceTransformer (paraphrase-multilingual-MiniLM-L12-v2)
    - 📉 UMAP (Dimensionality Reduction)
    - 🎯 HDBSCAN (Clustering)
    - 🗂️ BERTopic (Topic Modeling)
    - 😊 Türkçe Sentiment Analysis (bert-base-turkish-sentiment-cased)
    """)


if __name__ == "__main__":
    main()
