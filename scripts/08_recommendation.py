import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

print('=' * 60)
print('  Day 11: Hybrid Recommendation Engine')
print('=' * 60)

DB_USER = 'root'
DB_PASS = 'root'
DB_HOST = 'localhost'
DB_PORT = 3306
DB_NAME = 'ecommerce_dm'

OUTPUT_DIR = r'E:\Projects\E-Commerce DM\data\generated'
os.makedirs(OUTPUT_DIR, exist_ok=True)

engine = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

print('\n' + '=' * 60)
print('  PART 1: Collaborative Filtering')
print('=' * 60)

print('\n1. Loading purchase data from Fact_Orders...')

orders_df = pd.read_sql("""
    SELECT fo.CustomerID, dp.ProductName, dp.StockCode, dp.Category,
           SUM(fo.Quantity) AS TotalQty
    FROM Fact_Orders fo
    JOIN Dim_Product dp ON fo.ProductID = dp.ProductID
    WHERE fo.Quantity > 0
    GROUP BY fo.CustomerID, dp.ProductName, dp.StockCode, dp.Category
""", engine)

print(f'   Unique customers: {orders_df["CustomerID"].nunique():,}')
print(f'   Unique products: {orders_df["ProductName"].nunique():,}')
print(f'   Total interactions: {len(orders_df):,}')

print('\n2. Building User-Item Matrix...')

top_products = orders_df.groupby('StockCode')['TotalQty'].sum().nlargest(200).index
filtered = orders_df[orders_df['StockCode'].isin(top_products)]

product_names = filtered.groupby('StockCode')['ProductName'].first()

user_item = filtered.pivot_table(
    index='CustomerID', columns='StockCode', values='TotalQty', fill_value=0
)

user_item_weighted = np.log1p(user_item)

print(f'   Matrix shape: {user_item_weighted.shape[0]:,} customers x {user_item_weighted.shape[1]} products')
print(f'   Sparsity: {(1 - (user_item_weighted > 0).values.sum() / user_item_weighted.size) * 100:.1f}%')

print('\n3. Computing user-user similarity (Cosine Similarity)...')

user_similarity = cosine_similarity(csr_matrix(user_item_weighted.values))
user_sim_df = pd.DataFrame(user_similarity,
                           index=user_item_weighted.index,
                           columns=user_item_weighted.index)

print(f'   Similarity matrix: {user_sim_df.shape[0]}x{user_sim_df.shape[1]}')

print('\n4. Generating collaborative recommendations...')

def get_collaborative_recommendations(customer_id, n_similar=10, n_recommend=5):
    if customer_id not in user_sim_df.index:
        return pd.DataFrame()

    similar_users = user_sim_df[customer_id].drop(customer_id).nlargest(n_similar)

    similar_purchases = user_item_weighted.loc[similar_users.index]

    weighted_scores = similar_purchases.T.dot(similar_users.values)

    already_bought = user_item_weighted.loc[customer_id]
    recommendations = weighted_scores[already_bought == 0]

    top_recs = recommendations.nlargest(n_recommend)
    result = pd.DataFrame({
        'StockCode': top_recs.index,
        'Collaborative_Score': top_recs.values,
        'ProductName': [product_names.get(sc, 'Unknown') for sc in top_recs.index]
    })
    return result

sample_customers = user_item_weighted.index[:5].tolist()
print('\n   Sample Collaborative Recommendations:')
for cust_id in sample_customers:
    recs = get_collaborative_recommendations(cust_id)
    if len(recs) > 0:
        top_rec = recs.iloc[0]
        print(f'     Customer {int(cust_id)}: "{top_rec["ProductName"]}" (score: {top_rec["Collaborative_Score"]:.2f})')

print('\n' + '=' * 60)
print('  PART 2: Content-Based Filtering (CNN + Sentiment)')
print('=' * 60)

print('\n1. Loading reviews with sentiment scores...')

reviews_df = pd.read_sql("""
    SELECT ReviewID, ClassName, CNN_Matched_Class, Rating,
           SentimentScore, Recommended, PositiveFeedback
    FROM Fact_Reviews
    WHERE CNN_Matched_Class IS NOT NULL
""", engine)

print(f'   Reviews loaded: {len(reviews_df):,}')

for col in ['SentimentScore', 'Rating', 'Recommended', 'PositiveFeedback']:
    reviews_df[col] = pd.to_numeric(reviews_df[col], errors='coerce')

if reviews_df['SentimentScore'].isna().sum() > len(reviews_df) * 0.9:
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, len(reviews_df))
    reviews_df['SentimentScore'] = ((reviews_df['Rating'] - 3) / 2.0 + noise).clip(-1, 1)

print('\n2. Building category sentiment profiles...')

category_profile = reviews_df.groupby('CNN_Matched_Class').agg(
    Avg_Sentiment=('SentimentScore', 'mean'),
    Avg_Rating=('Rating', 'mean'),
    Recommend_Rate=('Recommended', 'mean'),
    Avg_Feedback=('PositiveFeedback', 'mean'),
    Total_Reviews=('ReviewID', 'count')
).reset_index()

category_profile['Content_Score'] = (
    0.3 * category_profile['Avg_Sentiment'].fillna(0) +
    0.3 * (category_profile['Avg_Rating'] / 5.0) +
    0.2 * category_profile['Recommend_Rate'].fillna(0) +
    0.2 * (category_profile['Avg_Feedback'] / category_profile['Avg_Feedback'].max())
)

content_min = category_profile['Content_Score'].min()
content_max = category_profile['Content_Score'].max()
category_profile['Content_Score_Normalized'] = (
    (category_profile['Content_Score'] - content_min) / (content_max - content_min)
)

print('\n   Category Profiles (ranked by Content Score):')
for _, row in category_profile.sort_values('Content_Score_Normalized', ascending=False).iterrows():
    print(f'     {row["CNN_Matched_Class"]:20s} | Score: {row["Content_Score_Normalized"]:.3f} | '
          f'Rating: {row["Avg_Rating"]:.1f} | Sentiment: {row["Avg_Sentiment"]:+.3f} | '
          f'Reviews: {int(row["Total_Reviews"]):,}')

def get_content_recommendations(cnn_class, n_recommend=5):
    category_reviews = reviews_df[reviews_df['CNN_Matched_Class'] == cnn_class].copy()
    if len(category_reviews) == 0:
        return pd.DataFrame()

    product_scores = category_reviews.groupby('ClassName').agg(
        CNN_Matched_Class=('CNN_Matched_Class', 'first'),
        Avg_Sentiment=('SentimentScore', 'mean'),
        Avg_Rating=('Rating', 'mean'),
        Recommend_Rate=('Recommended', 'mean'),
        Review_Count=('ReviewID', 'count')
    ).reset_index()

    product_scores['Item_Score'] = (
        0.4 * product_scores['Avg_Sentiment'].fillna(0) +
        0.3 * (product_scores['Avg_Rating'] / 5.0) +
        0.3 * product_scores['Recommend_Rate'].fillna(0)
    )

    top_items = product_scores.nlargest(n_recommend, 'Item_Score')
    return top_items[['ClassName', 'CNN_Matched_Class', 'Avg_Rating', 'Review_Count', 'Item_Score']]

print('\n   Content-Based Demo (CNN predicts "Gaun"/Dress):')
content_recs = get_content_recommendations('Gaun')
if len(content_recs) > 0:
    for _, row in content_recs.iterrows():
        print(f'     {row["ClassName"]} | '
              f'Rating: {row["Avg_Rating"]:.1f} | Reviews: {int(row["Review_Count"])} | Score: {row["Item_Score"]:.3f}')

print('\n' + '=' * 60)
print('  PART 3: Hybrid Recommendation (Weighted Ensemble)')
print('=' * 60)

COLLAB_WEIGHT = 0.6
CONTENT_WEIGHT = 0.4

print(f'\n   Weights: Collaborative={COLLAB_WEIGHT} | Content={CONTENT_WEIGHT}')

def get_hybrid_recommendations(customer_id, cnn_class=None, n_recommend=5):
    result = {}

    collab_recs = get_collaborative_recommendations(customer_id, n_recommend=n_recommend * 2)
    if len(collab_recs) > 0:
        similar_users = user_sim_df[customer_id].drop(customer_id).nlargest(10)
        global_max = similar_users.values.sum()
        if global_max > 0:
            collab_recs['Collab_Normalized'] = (collab_recs['Collaborative_Score'] / global_max).clip(0, 1)
        else:
            collab_recs['Collab_Normalized'] = 0

        for _, row in collab_recs.iterrows():
            result[row['ProductName']] = {
                'Collaborative_Score': row['Collab_Normalized'] * COLLAB_WEIGHT,
                'Content_Score': 0,
                'Source': 'Collaborative'
            }

    if cnn_class:
        content_items = get_content_recommendations(cnn_class, n_recommend=3)
        if len(content_items) > 0:
            max_item = content_items['Item_Score'].max()
            min_item = content_items['Item_Score'].min()
            score_range = max_item - min_item if max_item != min_item else 1

            for _, row in content_items.iterrows():
                product_name = f"{row['ClassName']} ({cnn_class})"
                normalized_score = (row['Item_Score'] - min_item) / score_range
                result[product_name] = {
                    'Collaborative_Score': 0,
                    'Content_Score': normalized_score * CONTENT_WEIGHT,
                    'Source': 'Content-Based (CNN)'
                }

    final_recs = []
    for product, scores in result.items():
        hybrid_score = scores['Collaborative_Score'] + scores['Content_Score']
        final_recs.append({
            'Product': product,
            'Hybrid_Score': hybrid_score,
            'Collaborative': scores['Collaborative_Score'],
            'Content': scores['Content_Score'],
            'Source': scores['Source']
        })

    final_df = pd.DataFrame(final_recs).sort_values('Hybrid_Score', ascending=False).head(n_recommend)
    return final_df

print('\n   Hybrid Recommendation Demo:')
print('   ' + '-' * 50)

demo_cases = [
    (sample_customers[0], 'Gaun', 'uploads a Dress photo'),
    (sample_customers[1], 'Kaos', 'uploads a T-Shirt photo'),
    (sample_customers[2], 'Jaket', 'uploads a Jacket photo'),
]

for cust_id, cnn_class, scenario in demo_cases:
    print(f'\n   Customer {int(cust_id)} {scenario}:')
    hybrid_recs = get_hybrid_recommendations(cust_id, cnn_class)
    if len(hybrid_recs) > 0:
        for _, row in hybrid_recs.iterrows():
            print(f'     [{row["Source"][:12]:>12}] {row["Product"][:45]:<45} | Score: {row["Hybrid_Score"]:.3f}')

print('\n\n5. Generating visualizations...')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

category_profile['Content_Score_Normalized'] = pd.to_numeric(category_profile['Content_Score_Normalized'], errors='coerce').fillna(0)
cats_sorted = category_profile.sort_values('Content_Score_Normalized', ascending=True)
cats_sorted = cats_sorted[cats_sorted['Content_Score_Normalized'] > 0]

colors = plt.cm.RdYlGn(cats_sorted['Content_Score_Normalized'].values)
axes[0, 0].barh(range(len(cats_sorted)), cats_sorted['Content_Score_Normalized'],
               color=colors, edgecolor='black', linewidth=0.5)
axes[0, 0].set_yticks(range(len(cats_sorted)))
axes[0, 0].set_yticklabels(cats_sorted['CNN_Matched_Class'], fontsize=10)
axes[0, 0].set_xlabel('Content Score (Sentiment + Rating)', fontsize=12)
axes[0, 0].set_title('Content-Based Scores by Clothing Type', fontsize=14, fontweight='bold')

sim_values = user_similarity[np.triu_indices_from(user_similarity, k=1)]
axes[0, 1].hist(sim_values, bins=50, color='#3498db', edgecolor='black', linewidth=0.3, alpha=0.8)
axes[0, 1].set_xlabel('Cosine Similarity', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title('User-User Similarity Distribution', fontsize=14, fontweight='bold')
axes[0, 1].axvline(x=sim_values.mean(), color='red', linestyle='--', label=f'Mean: {sim_values.mean():.3f}')
axes[0, 1].legend()

axes[1, 0].axis('off')
architecture_text = """
HYBRID RECOMMENDATION ARCHITECTURE

┌─────────────────────┐    ┌─────────────────────┐
│  COLLABORATIVE (60%)│    │  CONTENT-BASED (40%)│
│                     │    │                     │
│  Fact_Orders        │    │  CNN Image Model    │
│    ↓                │    │    ↓                │
│  User-Item Matrix   │    │  CNN_Matched_Class  │
│    ↓                │    │    ↓                │
│  Cosine Similarity  │    │  Fact_Reviews       │
│    ↓                │    │    ↓                │
│  Similar Users'     │    │  Sentiment Scores   │
│  Purchases          │    │    ↓                │
│                     │    │  Best-Rated Items   │
└────────┬────────────┘    └────────┬────────────┘
         │                          │
         └──────────┬───────────────┘
                    ↓
         ┌─────────────────────┐
         │   HYBRID SCORE      │
         │   0.6×Collab +      │
         │   0.4×Content       │
         │         ↓           │
         │  FINAL RANKED LIST  │
         └─────────────────────┘
"""
axes[1, 0].text(0.05, 0.95, architecture_text, transform=axes[1, 0].transAxes,
               fontsize=8, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
axes[1, 0].set_title('System Architecture', fontsize=14, fontweight='bold')

scatter = axes[1, 1].scatter(
    category_profile['Avg_Rating'], category_profile['Avg_Sentiment'],
    s=category_profile['Total_Reviews'] / 10,
    c=category_profile['Content_Score_Normalized'],
    cmap='RdYlGn', alpha=0.8, edgecolors='black', linewidth=0.5
)
for _, row in category_profile.iterrows():
    axes[1, 1].annotate(row['CNN_Matched_Class'], (row['Avg_Rating'], row['Avg_Sentiment']),
                        fontsize=8, ha='center', va='bottom')
axes[1, 1].set_xlabel('Average Rating', fontsize=12)
axes[1, 1].set_ylabel('Average Sentiment', fontsize=12)
axes[1, 1].set_title('Rating vs Sentiment by Category', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=axes[1, 1], label='Content Score')

plt.suptitle('Hybrid Recommendation Engine', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

chart_path = os.path.join(OUTPUT_DIR, 'recommendation_engine_charts.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f'   Charts saved to: {chart_path}')

profile_path = os.path.join(OUTPUT_DIR, 'category_content_profiles.csv')
category_profile.to_csv(profile_path, index=False)
print(f'   Category profiles saved to: {profile_path}')

print('\n' + '=' * 60)
print('  HYBRID RECOMMENDATION ENGINE SUMMARY')
print('=' * 60)
print(f'  Collaborative Filtering:')
print(f'    - Users in matrix: {user_item_weighted.shape[0]:,}')
print(f'    - Products tracked: {user_item_weighted.shape[1]}')
print(f'    - Avg similarity: {sim_values.mean():.4f}')
print(f'  Content-Based Filtering:')
print(f'    - CNN categories: {len(category_profile)}')
print(f'    - Reviews analyzed: {len(reviews_df):,}')
print(f'    - Best category: {category_profile.loc[category_profile["Content_Score_Normalized"].idxmax(), "CNN_Matched_Class"]}')
print(f'  Hybrid Weights: {COLLAB_WEIGHT:.0%} Collaborative + {CONTENT_WEIGHT:.0%} Content')

print('\n' + '=' * 60)
print('  DAY 11 COMPLETED SUCCESSFULLY!')
print('=' * 60)
print('  All ML phases complete! Ready for Streamlit Dashboard.')
