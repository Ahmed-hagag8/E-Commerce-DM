import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt
import os
import warnings
import re

warnings.filterwarnings('ignore')

print('=' * 60)
print('  Day 8 (Part 2): Sentiment Analysis (NLP)')
print('=' * 60)

DB_USER = 'root'
DB_PASS = 'root'
DB_HOST = 'localhost'
DB_PORT = 3306
DB_NAME = 'ecommerce_dm'

OUTPUT_DIR = r'E:\Projects\E-Commerce DM\data\generated'
os.makedirs(OUTPUT_DIR, exist_ok=True)

engine = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

print('\n1. Loading reviews from Fact_Reviews...')

df = pd.read_sql("""
    SELECT ReviewID, Rating, ReviewText, ClassName, CNN_Matched_Class, 
           Recommended, PositiveFeedback, Age, ReviewLength
    FROM Fact_Reviews
    WHERE ReviewText IS NOT NULL AND TRIM(ReviewText) != ''
""", engine)

print(f'   Loaded {len(df):,} reviews with text')
print(f'   Rating distribution:')
for rating in sorted(df['Rating'].dropna().unique()):
    count = len(df[df['Rating'] == rating])
    print(f'     {int(rating)} stars: {count:,} reviews')

print('\n   Preprocessing review text (cleaning HTML, special chars)...')

def clean_text(text):
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\w\s.,!?\'-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['CleanText'] = df['ReviewText'].apply(clean_text)
print(f'   Preprocessing complete.')

print('\n2. Running VADER sentiment analysis...')

sia = SentimentIntensityAnalyzer()

df['VADER_Score'] = df['CleanText'].apply(lambda x: sia.polarity_scores(x)['compound'])

df['VADER_Label'] = df['VADER_Score'].apply(
    lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral')
)

print('   VADER Results:')
vader_dist = df['VADER_Label'].value_counts()
for label, count in vader_dist.items():
    pct = count / len(df) * 100
    print(f'     {label}: {count:,} ({pct:.1f}%)')

print('\n3. Running TextBlob sentiment analysis...')

def textblob_scores(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

tb_results = df['CleanText'].apply(textblob_scores)
df['TextBlob_Score'] = tb_results.apply(lambda x: x[0])
df['TextBlob_Subjectivity'] = tb_results.apply(lambda x: x[1])

df['TextBlob_Label'] = df['TextBlob_Score'].apply(
    lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral')
)

print('   TextBlob Results:')
tb_dist = df['TextBlob_Label'].value_counts()
for label, count in tb_dist.items():
    pct = count / len(df) * 100
    print(f'     {label}: {count:,} ({pct:.1f}%)')

print('\n4. Computing ensemble sentiment score...')

vader_mean, vader_std = df['VADER_Score'].mean(), df['VADER_Score'].std()
tb_mean, tb_std = df['TextBlob_Score'].mean(), df['TextBlob_Score'].std()

df['VADER_Normalized'] = (df['VADER_Score'] - vader_mean) / vader_std
df['TextBlob_Normalized'] = (df['TextBlob_Score'] - tb_mean) / tb_std

df['Ensemble_ZScore'] = (df['VADER_Normalized'] + df['TextBlob_Normalized']) / 2
df['Ensemble_Score'] = np.tanh(df['Ensemble_ZScore'])

df['Final_Sentiment'] = df['Ensemble_Score'].apply(
    lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
)

print('   Ensemble Results:')
final_dist = df['Final_Sentiment'].value_counts()
for label, count in final_dist.items():
    pct = count / len(df) * 100
    print(f'     {label}: {count:,} ({pct:.1f}%)')

print('\n5. Updating Fact_Reviews in database with sentiment scores...')

with engine.connect() as conn:
    update_data = [
        {'score': float(row['Ensemble_Score']), 'rid': int(row['ReviewID'])}
        for _, row in df.iterrows()
    ]
    batch_size = 5000
    for i in range(0, len(update_data), batch_size):
        batch = update_data[i:i+batch_size]
        conn.execute(
            text("UPDATE Fact_Reviews SET SentimentScore = :score WHERE ReviewID = :rid"),
            batch
        )
        conn.commit()
        print(f'   Updated {min(i + batch_size, len(update_data)):,}/{len(update_data):,} rows...')

print('   Database updated successfully!')

print('\n' + '=' * 60)
print('  SENTIMENT BY CLOTHING CATEGORY (CNN_Matched_Class)')
print('=' * 60)

category_sentiment = df.groupby('CNN_Matched_Class').agg(
    Avg_Sentiment=('Ensemble_Score', 'mean'),
    Avg_Rating=('Rating', 'mean'),
    Total_Reviews=('ReviewID', 'count'),
    Positive_Pct=('Final_Sentiment', lambda x: (x == 'Positive').mean() * 100)
).sort_values('Avg_Sentiment', ascending=False)

for cat, row in category_sentiment.iterrows():
    print(f'  {cat:20s} | Sentiment: {row["Avg_Sentiment"]:+.3f} | Rating: {row["Avg_Rating"]:.1f} | Reviews: {int(row["Total_Reviews"]):,} | Positive: {row["Positive_Pct"]:.0f}%')

print('\n6. Generating visualizations...')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

colors = {'Positive': '#2ecc71', 'Neutral': '#f39c12', 'Negative': '#e74c3c'}
labels = ['Positive', 'Neutral', 'Negative']
sizes = [final_dist.get(l, 0) for l in labels]
plot_colors = [colors[l] for l in labels]
axes[0, 0].pie(sizes, labels=labels, colors=plot_colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 12})
axes[0, 0].set_title('Overall Sentiment Distribution', fontsize=14, fontweight='bold')

rating_sentiment = df.groupby('Rating')['Ensemble_Score'].mean()
bar_colors = ['#e74c3c' if v < 0 else '#f39c12' if v < 0.2 else '#2ecc71' for v in rating_sentiment.values]
axes[0, 1].bar(rating_sentiment.index, rating_sentiment.values, color=bar_colors, edgecolor='black', linewidth=0.5)
axes[0, 1].set_xlabel('Star Rating', fontsize=12)
axes[0, 1].set_ylabel('Average Sentiment Score', fontsize=12)
axes[0, 1].set_title('Sentiment Score by Star Rating', fontsize=14, fontweight='bold')
axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)

top_cats = category_sentiment.head(10)
cat_colors = ['#2ecc71' if v > 0.1 else '#f39c12' if v > 0 else '#e74c3c' for v in top_cats['Avg_Sentiment']]
axes[1, 0].barh(range(len(top_cats)), top_cats['Avg_Sentiment'], color=cat_colors, edgecolor='black', linewidth=0.5)
axes[1, 0].set_yticks(range(len(top_cats)))
axes[1, 0].set_yticklabels(top_cats.index, fontsize=10)
axes[1, 0].set_xlabel('Average Sentiment Score', fontsize=12)
axes[1, 0].set_title('Sentiment by Clothing Category', fontsize=14, fontweight='bold')
axes[1, 0].axvline(x=0, color='black', linestyle='--', alpha=0.3)

scatter = axes[1, 1].scatter(df['VADER_Score'], df['TextBlob_Score'],
                              alpha=0.1, s=5, c=df['Rating'], cmap='RdYlGn')
axes[1, 1].set_xlabel('VADER Score', fontsize=12)
axes[1, 1].set_ylabel('TextBlob Score', fontsize=12)
axes[1, 1].set_title('VADER vs TextBlob Agreement', fontsize=14, fontweight='bold')
axes[1, 1].plot([-1, 1], [-1, 1], 'r--', alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 1], label='Star Rating')

plt.suptitle('Sentiment Analysis - Women\'s Clothing Reviews', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

chart_path = os.path.join(OUTPUT_DIR, 'sentiment_analysis_charts.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f'   Charts saved to: {chart_path}')

sentiment_export = df[['ReviewID', 'Rating', 'ClassName', 'CNN_Matched_Class',
                        'VADER_Score', 'TextBlob_Score', 'Ensemble_Score',
                        'Final_Sentiment', 'TextBlob_Subjectivity']].copy()
sentiment_path = os.path.join(OUTPUT_DIR, 'sentiment_results.csv')
sentiment_export.to_csv(sentiment_path, index=False)
print(f'   Results saved to: {sentiment_path}')

print('\n' + '=' * 60)
print('  SENTIMENT ANALYSIS SUMMARY')
print('=' * 60)
print(f'  Total reviews analyzed: {len(df):,}')
print(f'  VADER avg score: {df["VADER_Score"].mean():.3f}')
print(f'  TextBlob avg score: {df["TextBlob_Score"].mean():.3f}')
print(f'  Ensemble avg score: {df["Ensemble_Score"].mean():.3f}')
print(f'  Correlation (VADER vs TextBlob): {df["VADER_Score"].corr(df["TextBlob_Score"]):.3f}')
print(f'  Correlation (Sentiment vs Rating): {df["Ensemble_Score"].corr(df["Rating"]):.3f}')

print('\n' + '=' * 60)
print('  DAY 8 (Part 2) COMPLETED SUCCESSFULLY!')
print('=' * 60)
print('  Next: python scripts/07_churn_prediction.py')
