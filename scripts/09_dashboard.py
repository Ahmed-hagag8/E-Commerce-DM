import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text
from pathlib import Path
import os

st.set_page_config(
    page_title="Customer Behavior Analysis And Hybrid Recommendation System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main { background-color: #0e1117; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-card h3 {
        color: #a0aec0;
        font-size: 14px;
        margin-bottom: 5px;
        font-family: 'Inter', sans-serif;
    }
    .metric-card .value {
        color: #63b3ed;
        font-size: 32px;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
    }
    .metric-card .value.green { color: #48bb78; }
    .metric-card .value.orange { color: #ed8936; }
    .metric-card .value.purple { color: #9f7aea; }
    .metric-card .value.red { color: #fc8181; }

    .section-header {
        background: linear-gradient(90deg, #2b6cb0 0%, #4299e1 100%);
        border-radius: 8px;
        padding: 12px 20px;
        margin: 20px 0 15px 0;
        color: white;
        font-size: 18px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e;
        border-radius: 8px;
        padding: 10px 20px;
        color: #a0aec0;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2b6cb0, #4299e1);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_engine():
    return create_engine('mysql+pymysql://root:root@localhost:3306/ecommerce_dm')

@st.cache_data(ttl=300)
def load_data(query):
    engine = get_engine()
    return pd.read_sql(query, engine)

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / 'data' / 'generated'

with st.sidebar:
    st.markdown("# Customer Behavior Analysis And Hybrid Recommendation System")
    st.markdown("---")

    page = st.radio("Navigate", [
        "📊 Overview",
        "👥 Customer Analysis",
        "📦 Product Analysis",
        "💬 Sentiment Analysis",
        "🔗 Association Rules",
        "🤖 Churn Prediction",
        "🎯 Recommendation Engine"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("##### Data Warehouse Stats")
    try:
        engine = get_engine()
        with engine.connect() as conn:
            for table in ['Fact_Orders', 'Fact_Reviews', 'Dim_Customer']:
                count = conn.execute(text(f'SELECT COUNT(*) FROM {table}')).scalar()
                st.metric(table, f"{count:,}")
    except:
        st.warning("Database not connected")

def metric_card(label, value, color=""):
    st.markdown(f"""
    <div class="metric-card">
        <h3>{label}</h3>
        <div class="value {color}">{value}</div>
    </div>
    """, unsafe_allow_html=True)

if page == "📊 Overview":
    st.markdown("# 🏪 Customer Behavior Analysis And Hybrid Recommendation System")

    col1, col2, col3, col4, col5 = st.columns(5)
    try:
        customers = load_data("SELECT COUNT(*) as c FROM Dim_Customer").iloc[0, 0]
        products = load_data("SELECT COUNT(*) as c FROM Dim_Product").iloc[0, 0]
        orders = load_data("SELECT COUNT(*) as c FROM Fact_Orders").iloc[0, 0]
        reviews = load_data("SELECT COUNT(*) as c FROM Fact_Reviews").iloc[0, 0]
        revenue = load_data("SELECT SUM(TotalPrice) as r FROM Fact_Orders").iloc[0, 0]

        with col1: metric_card("Customers", f"{customers:,}", "")
        with col2: metric_card("Products", f"{products:,}", "green")
        with col3: metric_card("Orders", f"{orders:,}", "orange")
        with col4: metric_card("Reviews", f"{reviews:,}", "purple")
        with col5: metric_card("Revenue", f"${revenue/1_000_000:.1f}M" if revenue else "$0", "green")
    except:
        st.error("Could not load metrics from database")

    st.markdown("")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">📈 Revenue Over Time</div>', unsafe_allow_html=True)
        try:
            revenue_df = load_data("""
                SELECT dt.FullDate, SUM(fo.TotalPrice) AS Revenue, COUNT(*) AS Orders
                FROM Fact_Orders fo JOIN Dim_Time dt ON fo.TimeID = dt.TimeID
                GROUP BY dt.FullDate ORDER BY dt.FullDate
            """)
            revenue_df['FullDate'] = pd.to_datetime(revenue_df['FullDate'])
            revenue_df['MonthYear'] = revenue_df['FullDate'].dt.to_period('M').astype(str)
            monthly = revenue_df.groupby('MonthYear').agg({'Revenue': 'sum', 'Orders': 'sum'}).reset_index()

            fig = go.Figure()
            fig.add_trace(go.Bar(x=monthly['MonthYear'], y=monthly['Revenue'],
                                 name='Revenue', marker_color='#4299e1'))
            fig.update_layout(template='plotly_dark', height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

    with col2:
        st.markdown('<div class="section-header">🌍 Sales by Region</div>', unsafe_allow_html=True)
        try:
            region_df = load_data("""
                SELECT dl.Region, SUM(fo.TotalPrice) AS Revenue, COUNT(*) AS Orders
                FROM Fact_Orders fo JOIN Dim_Location dl ON fo.LocationID = dl.LocationID
                GROUP BY dl.Region ORDER BY Revenue DESC
            """)
            region_df['Revenue_Label'] = region_df['Revenue'].apply(lambda x: f"${x:,.0f}")
            region_df['Pct'] = (region_df['Revenue'] / region_df['Revenue'].sum() * 100).round(1)
            region_df['Label'] = region_df['Revenue_Label'] + ' (' + region_df['Pct'].astype(str) + '%)'
            region_sorted = region_df.sort_values('Revenue', ascending=True)

            colors = ['#2d3748', '#4a5568', '#718096', '#a0aec0', '#ed8936', '#48bb78', '#4299e1']
            fig = go.Figure(go.Bar(
                x=region_sorted['Revenue'],
                y=region_sorted['Region'],
                orientation='h',
                text=region_sorted['Label'],
                textposition='auto',
                marker_color=colors[:len(region_sorted)],
                marker_line=dict(color='white', width=0.5)
            ))
            fig.update_layout(
                template='plotly_dark', height=350,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis_title='Revenue ($)',
                yaxis_title=''
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown('<div class="section-header">🏗️ Pipeline Architecture (Medallion)</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🥉 Bronze (Raw)")
        st.markdown("- `staging_retail` (1M+ rows)\n- `staging_reviews` (23K rows)\n- `staging_amazon` (550K rows)")
    with col2:
        st.markdown("### 🥈 Silver (Cleaned)")
        st.markdown("- `clean_retail` (779K rows)\n- `clean_reviews` (22K rows)\n- `clean_amazon` (551K rows)")
    with col3:
        st.markdown("### 🥇 Gold (Star Schema)")
        st.markdown("- `Dim_Customer` + `Dim_Product`\n- `Dim_Time` + `Dim_Location`\n- `Fact_Orders` + `Fact_Reviews`")

elif page == "👥 Customer Analysis":
    st.markdown("# 👥 Customer Analysis & RFM Segmentation")

    try:
        customers = load_data("SELECT * FROM Dim_Customer")

        col1, col2, col3, col4 = st.columns(4)
        churn_rate = customers['ChurnLabel'].mean() * 100
        with col1: metric_card("Total Customers", f"{len(customers):,}")
        with col2: metric_card("Avg Monetary", f"${customers['Monetary'].mean():,.0f}", "green")
        with col3: metric_card("Avg Frequency", f"{customers['Frequency'].mean():.1f}", "orange")
        with col4: metric_card("Churn Rate", f"{churn_rate:.1f}%", "red")

        st.markdown("")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">📊 RFM Distribution</div>', unsafe_allow_html=True)
            fig = make_subplots(rows=1, cols=3, subplot_titles=('Recency', 'Frequency', 'Monetary'))
            fig.add_trace(go.Histogram(x=customers['Recency'], marker_color='#4299e1', name='Recency'), row=1, col=1)
            fig.add_trace(go.Histogram(x=customers['Frequency'], marker_color='#48bb78', name='Frequency'), row=1, col=2)
            fig.add_trace(go.Histogram(x=customers['Monetary'], marker_color='#ed8936', name='Monetary'), row=1, col=3)
            fig.update_layout(template='plotly_dark', height=300, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">🔄 Churn Distribution</div>', unsafe_allow_html=True)
            churn_df = customers['ChurnLabel'].value_counts().reset_index()
            churn_df.columns = ['Status', 'Count']
            churn_df['Status'] = churn_df['Status'].map({0: 'Active', 1: 'Churned'})
            fig = px.pie(churn_df, values='Count', names='Status',
                         color_discrete_map={'Active': '#48bb78', 'Churned': '#fc8181'})
            fig.update_layout(template='plotly_dark', height=300, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">🏆 Top 10 Customers by Revenue</div>', unsafe_allow_html=True)
        top_cust = customers.nlargest(10, 'Monetary')[['CustomerID', 'Country', 'Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'ChurnLabel']]
        top_cust['ChurnLabel'] = top_cust['ChurnLabel'].map({0: '✅ Active', 1: '❌ Churned'})
        st.dataframe(top_cust, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error loading customer data: {e}")

elif page == "📦 Product Analysis":
    st.markdown("# 📦 Product & Category Analysis")

    try:
        products = load_data("SELECT * FROM Dim_Product")

        col1, col2, col3 = st.columns(3)
        with col1: metric_card("Total Products", f"{len(products):,}")
        with col2: metric_card("Avg Price", f"${products['AvgPrice'].mean():.2f}", "green")
        with col3: metric_card("Categories", f"{products['Category'].nunique()}", "purple")

        st.markdown("")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">📂 Products by Category</div>', unsafe_allow_html=True)
            cat_df = products['Category'].value_counts().reset_index()
            cat_df.columns = ['Category', 'Count']
            fig = px.bar(cat_df, x='Count', y='Category', orientation='h',
                         color='Count', color_continuous_scale='Blues')
            fig.update_layout(template='plotly_dark', height=400, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">💰 Products by Price Range</div>', unsafe_allow_html=True)
            price_df = products['PriceRange'].value_counts().reset_index()
            price_df.columns = ['PriceRange', 'Count']
            fig = px.pie(price_df, values='Count', names='PriceRange',
                         color_discrete_sequence=['#48bb78', '#4299e1', '#ed8936', '#9f7aea'])
            fig.update_layout(template='plotly_dark', height=400, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")

elif page == "💬 Sentiment Analysis":
    st.markdown("# 💬 Sentiment Analysis (NLP)")

    try:
        reviews = load_data("""
            SELECT Rating, ClassName, CNN_Matched_Class, SentimentScore,
                   Recommended, PositiveFeedback, ReviewLength
            FROM Fact_Reviews WHERE SentimentScore IS NOT NULL
        """)

        reviews['Sentiment'] = reviews['SentimentScore'].apply(
            lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
        )

        col1, col2, col3, col4 = st.columns(4)
        avg_sent = reviews['SentimentScore'].mean()
        pos_pct = (reviews['Sentiment'] == 'Positive').mean() * 100
        with col1: metric_card("Reviews Analyzed", f"{len(reviews):,}")
        with col2: metric_card("Avg Sentiment", f"{avg_sent:+.3f}", "green")
        with col3: metric_card("Positive %", f"{pos_pct:.1f}%", "green")
        with col4: metric_card("Avg Rating", f"{reviews['Rating'].mean():.1f}⭐", "orange")

        st.markdown("")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">🎭 Sentiment Distribution</div>', unsafe_allow_html=True)
            sent_df = reviews['Sentiment'].value_counts().reset_index()
            sent_df.columns = ['Sentiment', 'Count']
            fig = px.pie(sent_df, values='Count', names='Sentiment',
                         color_discrete_map={'Positive': '#48bb78', 'Neutral': '#ed8936', 'Negative': '#fc8181'})
            fig.update_layout(template='plotly_dark', height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">⭐ Sentiment by Rating</div>', unsafe_allow_html=True)
            rating_sent = reviews.groupby('Rating')['SentimentScore'].mean().reset_index()
            colors = ['#fc8181' if v < 0 else '#ed8936' if v < 0.3 else '#48bb78' for v in rating_sent['SentimentScore']]
            fig = go.Figure(go.Bar(x=rating_sent['Rating'], y=rating_sent['SentimentScore'],
                                   marker_color=colors))
            fig.update_layout(template='plotly_dark', height=350, xaxis_title='Star Rating',
                              yaxis_title='Avg Sentiment', margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">👗 Sentiment by Clothing Category (CNN Integration)</div>', unsafe_allow_html=True)
        cat_sent = reviews.groupby('CNN_Matched_Class').agg(
            Avg_Sentiment=('SentimentScore', 'mean'),
            Avg_Rating=('Rating', 'mean'),
            Reviews=('Rating', 'count')
        ).sort_values('Avg_Sentiment', ascending=False).reset_index()

        fig = px.bar(cat_sent, x='CNN_Matched_Class', y='Avg_Sentiment',
                     color='Avg_Rating', color_continuous_scale='RdYlGn',
                     text=cat_sent['Avg_Sentiment'].apply(lambda x: f"{x:+.3f}"))
        fig.update_layout(template='plotly_dark', height=400, xaxis_title='CNN Category',
                          yaxis_title='Average Sentiment Score', margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")

elif page == "🔗 Association Rules":
    st.markdown("# 🔗 Association Rules (Market Basket Analysis)")

    try:
        rules_path = DATA_DIR / 'association_rules.csv'
        if rules_path.exists():
            rules = pd.read_csv(rules_path)

            col1, col2, col3 = st.columns(3)
            with col1: metric_card("Rules Found", f"{len(rules)}")
            with col2: metric_card("Avg Confidence", f"{rules['Confidence'].mean():.1%}", "green")
            with col3: metric_card("Max Lift", f"{rules['Lift'].max():.1f}x", "orange")

            st.markdown("")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="section-header">📊 Support vs Confidence</div>', unsafe_allow_html=True)
                fig = px.scatter(rules, x='Support', y='Confidence', size='Lift',
                                 color='Lift', color_continuous_scale='RdYlGn',
                                 hover_data=['Antecedent', 'Consequent'])
                fig.update_layout(template='plotly_dark', height=400, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown('<div class="section-header">🏆 Top Rules by Lift</div>', unsafe_allow_html=True)
                top10 = rules.nlargest(10, 'Lift').copy()
                top10['Rule'] = top10['Antecedent'].str[:20] + ' → ' + top10['Consequent'].str[:20]
                fig = px.bar(top10, x='Lift', y='Rule', orientation='h',
                             color='Confidence', color_continuous_scale='Blues')
                fig.update_layout(template='plotly_dark', height=400, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="section-header">📋 All Association Rules</div>', unsafe_allow_html=True)
            display_rules = rules.copy()
            display_rules['Confidence'] = display_rules['Confidence'].apply(lambda x: f"{x:.1%}")
            display_rules['Support'] = display_rules['Support'].apply(lambda x: f"{x:.3f}")
            display_rules['Lift'] = display_rules['Lift'].apply(lambda x: f"{x:.2f}")
            st.dataframe(display_rules, use_container_width=True, hide_index=True)
        else:
            st.warning("Run `python scripts/05_association_rules.py` first")
    except Exception as e:
        st.error(f"Error: {e}")

elif page == "🤖 Churn Prediction":
    st.markdown("# 🤖 Churn Prediction (Supervised ML)")

    try:
        comparison_path = DATA_DIR / 'churn_model_comparison.csv'
        if comparison_path.exists():
            comparison = pd.read_csv(comparison_path)

            best = comparison.loc[comparison['F1_Score'].idxmax()]
            col1, col2, col3, col4 = st.columns(4)
            with col1: metric_card("Models Trained", f"{len(comparison)}")
            with col2: metric_card("Best Model", best['Model'].split()[0], "green")
            with col3: metric_card("Best F1", f"{best['F1_Score']:.4f}", "green")
            with col4: metric_card("Best AUC", f"{best['AUC']:.4f}", "purple")

            st.markdown("")

            st.markdown('<div class="section-header">📊 Model Performance Comparison</div>', unsafe_allow_html=True)
            metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC']
            fig = go.Figure()
            colors = ['#4299e1', '#48bb78', '#ed8936', '#fc8181', '#9f7aea']
            for i, metric in enumerate(metrics_to_plot):
                fig.add_trace(go.Bar(name=metric, x=comparison['Model'], y=comparison[metric],
                                     marker_color=colors[i]))
            fig.update_layout(template='plotly_dark', barmode='group', height=400,
                              yaxis_range=[0, 1.0], margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="section-header">📋 Detailed Comparison</div>', unsafe_allow_html=True)
            st.dataframe(comparison.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC'],
                                                         color='#2d6a4f'), use_container_width=True, hide_index=True)

            chart_img = DATA_DIR / 'churn_prediction_charts.png'
            if chart_img.exists():
                st.markdown('<div class="section-header">📈 Evaluation Charts</div>', unsafe_allow_html=True)
                st.image(str(chart_img), use_container_width=True)
        else:
            st.warning("Run `python scripts/07_churn_prediction.py` first")
    except Exception as e:
        st.error(f"Error: {e}")

elif page == "🎯 Recommendation Engine":
    st.markdown("# 🎯 Hybrid Recommendation Engine")
    st.markdown("*Combining Collaborative Filtering (60%) + Content-Based Filtering (40%)*")

    try:
        profile_path = DATA_DIR / 'category_content_profiles.csv'
        if profile_path.exists():
            profiles = pd.read_csv(profile_path)

            col1, col2, col3 = st.columns(3)
            with col1: metric_card("CNN Categories", f"{len(profiles)}")
            with col2: metric_card("Best Category", profiles.loc[profiles['Content_Score_Normalized'].idxmax(), 'CNN_Matched_Class'], "green")
            with col3: metric_card("Reviews Used", f"{profiles['Total_Reviews'].sum():,}", "purple")

            st.markdown("")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="section-header">👗 Content Scores by Category</div>', unsafe_allow_html=True)
                fig = px.bar(profiles.sort_values('Content_Score_Normalized'),
                             x='Content_Score_Normalized', y='CNN_Matched_Class',
                             orientation='h', color='Content_Score_Normalized',
                             color_continuous_scale='RdYlGn')
                fig.update_layout(template='plotly_dark', height=400, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown('<div class="section-header">⭐ Rating vs Sentiment</div>', unsafe_allow_html=True)
                fig = px.scatter(profiles, x='Avg_Rating', y='Avg_Sentiment',
                                 size='Total_Reviews', color='Content_Score_Normalized',
                                 text='CNN_Matched_Class', color_continuous_scale='RdYlGn',
                                 size_max=40)
                fig.update_traces(textposition='top center')
                fig.update_layout(template='plotly_dark', height=400, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="section-header">🏗️ How It Works</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                ### 📊 Collaborative (60%)
                - Builds User-Item Matrix from `Fact_Orders`
                - Computes Cosine Similarity between users
                - Recommends products that similar users bought
                """)
            with col2:
                st.markdown("""
                ### 🧠 Content-Based (40%)
                - CNN classifies clothing images
                - Maps to `CNN_Matched_Class` in reviews
                - Recommends best-reviewed items in that category
                """)
            with col3:
                st.markdown("""
                ### ⚡ Hybrid Fusion
                - Weighted combination: `0.6×Collab + 0.4×Content`
                - Solves Cold Start problem
                - Multi-modal: Images + Text + Purchases
                """)

            chart_img = DATA_DIR / 'recommendation_engine_charts.png'
            if chart_img.exists():
                st.markdown('<div class="section-header">📈 System Analysis</div>', unsafe_allow_html=True)
                st.image(str(chart_img), use_container_width=True)
        else:
            st.warning("Run `python scripts/08_recommendation.py` first")
    except Exception as e:
        st.error(f"Error: {e}")
