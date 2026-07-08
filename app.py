"""Streamlit Dashboard for Legacy Review Analysis Projects"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import ast

# Page config
st.set_page_config(
    page_title="Review Analysis - Legacy Projects",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title
st.title("📊 Müşteri Yorum Analizi - Eski Projeler")
st.markdown("**Aspect-Based Sentiment & Trend Analysis**")

# Data paths
DATA_DIR = Path("data/processed")

@st.cache_data
def load_processed_reviews():
    """Load processed reviews with aspect sentiments"""
    path = DATA_DIR / "processed_reviews.csv"
    if path.exists():
        df = pd.read_csv(path)
        # Convert string representation of lists to actual lists
        if "detected_aspects" in df.columns:
            df["detected_aspects"] = df["detected_aspects"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else []
            )
        return df
    return None

@st.cache_data
def load_reviews_with_sentiment():
    """Load reviews with BERT sentiment"""
    path = DATA_DIR / "reviews_with_sentiment.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_clustered_reviews():
    """Load semantic clustered reviews"""
    path = DATA_DIR / "clustered_reviews.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

# Load all data
df_processed = load_processed_reviews()
df_sentiment = load_reviews_with_sentiment()
df_clustered = load_clustered_reviews()

if df_processed is None:
    st.error("❌ Processed reviews veri dosyası bulunamadı!")
    st.stop()

# Display data loading status
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📊 Toplam Yorum", len(df_processed))
with col2:
    if df_sentiment is not None:
        st.metric("🎭 Sentiment Analiz", len(df_sentiment))
with col3:
    if df_clustered is not None:
        st.metric("🎯 Semantic Clusters", df_clustered["cluster"].nunique() if "cluster" in df_clustered.columns else 0)

st.divider()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Aspect Analysis",
    "🎭 Sentiment Overview",
    "🔍 Detailed Reviews",
    "📊 Clustering"
])

# ==================== TAB 1: ASPECT ANALYSIS ====================
with tab1:
    st.subheader("Aspect-Based Sentiment Analysis")
    
    # Get all aspect columns
    aspect_cols = [col for col in df_processed.columns if "_sentiment" in col and col != "overall_rule_sentiment"]
    
    if aspect_cols:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Aspect Sentiment Dağılımı**")
            
            # Create aspect sentiment summary
            aspect_data = []
            for aspect_col in aspect_cols:
                aspect_name = aspect_col.replace("_sentiment", "").replace("_", " ").title()
                counts = df_processed[aspect_col].value_counts()
                
                aspect_data.append({
                    "Aspect": aspect_name,
                    "Positive": int(counts.get("positive", 0)),
                    "Negative": int(counts.get("negative", 0)),
                    "Neutral": int(counts.get("neutral", 0))
                })
            
            aspect_df = pd.DataFrame(aspect_data)
            
            # Melt for visualization
            aspect_melted = aspect_df.melt(
                id_vars="Aspect",
                value_vars=["Positive", "Negative", "Neutral"],
                var_name="Sentiment",
                value_name="Count"
            )
            
            fig = px.bar(
                aspect_melted,
                x="Aspect",
                y="Count",
                color="Sentiment",
                title="Aspect Sentiment Distribution",
                barmode="group",
                color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"}
            )
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Aspect Summary Table**")
            
            # Calculate percentages
            for idx, row in aspect_df.iterrows():
                total = row["Positive"] + row["Negative"] + row["Neutral"]
                if total > 0:
                    aspect_df.at[idx, "Positive %"] = f"{row['Positive']/total*100:.1f}%"
                    aspect_df.at[idx, "Negative %"] = f"{row['Negative']/total*100:.1f}%"
            
            display_df = aspect_df[["Aspect", "Positive", "Positive %", "Negative", "Negative %", "Neutral"]]
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Most detected aspects
        st.write("**En Çok Bahsedilen Aspect'ler**")
        
        if "detected_aspects" in df_processed.columns:
            all_aspects = []
            for aspects_list in df_processed["detected_aspects"]:
                if isinstance(aspects_list, list):
                    all_aspects.extend(aspects_list)
            
            if all_aspects:
                aspect_counts = pd.Series(all_aspects).value_counts().head(10)
                aspect_names = [col.replace("_", " ").title() for col in aspect_counts.index]
                
                fig = px.bar(
                    x=aspect_names,
                    y=aspect_counts.values,
                    title="Top 10 Most Discussed Aspects",
                    labels={"x": "Aspect", "y": "Mentions"},
                    color=aspect_counts.values,
                    color_continuous_scale="Viridis"
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aspect sentiment columns not found in data")

# ==================== TAB 2: SENTIMENT OVERVIEW ====================
with tab2:
    st.subheader("Overall Sentiment Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Genel Sentiment Dağılımı (BERT)**")
        
        if df_sentiment is not None and "sentiment_label" in df_sentiment.columns:
            sentiment_counts = df_sentiment["sentiment_label"].value_counts()
            
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution",
                color_discrete_map={
                    "POSITIVE": "#2ecc71",
                    "NEGATIVE": "#e74c3c",
                    "NEUTRAL": "#95a5a6"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.write("**İstatistikler**")
            for label, count in sentiment_counts.items():
                pct = count / len(df_sentiment) * 100
                st.write(f"- **{label}**: {count} ({pct:.1f}%)")
    
    with col2:
        st.write("**Rating vs Sentiment**")
        
        if "rating" in df_processed.columns and "overall_rule_sentiment" in df_processed.columns:
            rating_sentiment = df_processed.groupby(["rating", "overall_rule_sentiment"]).size().reset_index(name="count")
            
            fig = px.bar(
                rating_sentiment,
                x="rating",
                y="count",
                color="overall_rule_sentiment",
                title="Rating vs Rule-Based Sentiment",
                barmode="group",
                color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment score distribution
    if df_sentiment is not None and "sentiment_score" in df_sentiment.columns:
        st.write("**Sentiment Score Distribution**")
        
        fig = px.histogram(
            df_sentiment,
            x="sentiment_score",
            nbins=30,
            title="BERT Sentiment Score Distribution",
            labels={"sentiment_score": "Sentiment Score"},
            color_discrete_sequence=["#3498db"]
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 3: DETAILED REVIEWS ====================
with tab3:
    st.subheader("Yorumları Detaylı İncele")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if "overall_rule_sentiment" in df_processed.columns:
            sentiment_filter = st.multiselect(
                "Sentiment Filtresi",
                options=df_processed["overall_rule_sentiment"].unique(),
                default=df_processed["overall_rule_sentiment"].unique()
            )
        else:
            sentiment_filter = None
    
    with col2:
        rating_filter = st.slider(
            "Rating Range",
            min_value=1,
            max_value=5,
            value=(1, 5)
        )
    
    with col3:
        search_term = st.text_input("Yorum Ara", placeholder="Örn: kalite, kargo...")
    
    # Apply filters
    filtered_df = df_processed.copy()
    
    if sentiment_filter and "overall_rule_sentiment" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["overall_rule_sentiment"].isin(sentiment_filter)]
    
    if "rating" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["rating"] >= rating_filter[0]) &
            (filtered_df["rating"] <= rating_filter[1])
        ]
    
    if search_term and "comment" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["comment"].str.lower().str.contains(search_term.lower(), na=False)
        ]
    
    # Display results
    st.write(f"**Toplam Sonuç: {len(filtered_df)}**")
    
    if len(filtered_df) > 0:
        # Create display dataframe
        display_cols = []
        if "rating" in filtered_df.columns:
            display_cols.append("rating")
        if "comment" in filtered_df.columns:
            display_cols.append("comment")
        if "overall_rule_sentiment" in filtered_df.columns:
            display_cols.append("overall_rule_sentiment")
        if "detected_aspects" in filtered_df.columns:
            display_cols.append("detected_aspects")
        
        if display_cols:
            display_df = filtered_df[display_cols].head(50).copy()
            display_df.columns = [col.replace("_", " ").title() for col in display_df.columns]
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("Sonuç bulunamadı")

# ==================== TAB 4: CLUSTERING ====================
with tab4:
    st.subheader("Semantic Clustering Analysis")
    
    if df_clustered is not None and "cluster" in df_clustered.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Cluster Dağılımı**")
            
            cluster_counts = df_clustered["cluster"].value_counts().sort_index()
            
            fig = px.bar(
                x=cluster_counts.index,
                y=cluster_counts.values,
                title="Reviews per Cluster",
                labels={"x": "Cluster ID", "y": "Review Count"},
                color=cluster_counts.values,
                color_continuous_scale="Blues"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Cluster Sentiments**")
            
            if "sentiment_label" in df_clustered.columns:
                cluster_sentiment = df_clustered.groupby(["cluster", "sentiment_label"]).size().reset_index(name="count")
                
                fig = px.bar(
                    cluster_sentiment,
                    x="cluster",
                    y="count",
                    color="sentiment_label",
                    title="Sentiment Distribution by Cluster",
                    barmode="stack",
                    color_discrete_map={
                        "POSITIVE": "#2ecc71",
                        "NEGATIVE": "#e74c3c",
                        "NEUTRAL": "#95a5a6"
                    }
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Cluster details
        st.write("**Cluster İçeriği**")
        
        selected_cluster = st.selectbox(
            "Cluster Seç",
            options=sorted(df_clustered["cluster"].unique())
        )
        
        cluster_data = df_clustered[df_clustered["cluster"] == selected_cluster]
        
        st.write(f"**Cluster {selected_cluster} - {len(cluster_data)} yorum**")
        
        if "comment" in cluster_data.columns:
            for idx, row in cluster_data.head(10).iterrows():
                with st.expander(f"Yorum {idx}: {row.get('comment', '')[:60]}..."):
                    st.write(f"**Comment:** {row.get('comment', 'N/A')}")
                    if "sentiment_label" in row:
                        st.write(f"**Sentiment:** {row.get('sentiment_label', 'N/A')}")
                    if "sentiment_score" in row:
                        st.write(f"**Score:** {row.get('sentiment_score', 'N/A'):.3f}")
    else:
        st.info("Clustered reviews data not found. Run semantic_clustering.py first.")

# Footer
st.divider()
st.markdown("""
**Proje Bileşenleri:**
- 📥 Review Scraper (Trendyol API)
- 🎭 BERT Sentiment Analysis (Turkish)
- 🎯 Aspect-Based Sentiment
- 📊 Trend Analysis
- 🔍 Semantic Clustering
""")
