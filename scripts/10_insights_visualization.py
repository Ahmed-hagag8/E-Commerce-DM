import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sqlalchemy import create_engine
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = r'E:\Projects\E-Commerce DM\data\generated'
engine = create_engine('mysql+pymysql://root:root@localhost:3306/ecommerce_dm')

print('=' * 60)
print('  E-Commerce Data Insights & Visualizations')
print('=' * 60)

# ============================================================
# FIGURE 1: Sales Overview & Category Distribution (2x2)
# ============================================================
print('\n1. Generating Sales Overview...')

orders = pd.read_sql('''
    SELECT fo.TotalPrice, fo.Quantity, fo.UnitPrice,
           dt.FullDate, dt.Year, dt.Month, dt.MonthName, dt.Quarter, dt.IsWeekend,
           dp.Category, dp.PriceRange,
           dl.Region
    FROM Fact_Orders fo
    JOIN Dim_Time dt ON fo.TimeID = dt.TimeID
    JOIN Dim_Product dp ON fo.ProductID = dp.ProductID
    JOIN Dim_Location dl ON fo.LocationID = dl.LocationID
''', engine)
orders['FullDate'] = pd.to_datetime(orders['FullDate'])

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('E-Commerce Sales Overview & Distribution', fontsize=18, fontweight='bold', y=1.02)

# 1a: Monthly Revenue Trend (Increase vs Decrease)
monthly = orders.groupby([orders['FullDate'].dt.to_period('M')])['TotalPrice'].sum().reset_index()
monthly.columns = ['Month', 'Revenue']
monthly['Month'] = monthly['Month'].astype(str)
monthly['Change'] = monthly['Revenue'].pct_change() * 100
colors = ['#2ecc71' if c >= 0 else '#e74c3c' for c in monthly['Change'].fillna(0)]

axes[0, 0].bar(range(len(monthly)), monthly['Revenue'], color=colors, edgecolor='black', linewidth=0.5)
axes[0, 0].set_xticks(range(0, len(monthly), 3))
axes[0, 0].set_xticklabels(monthly['Month'].iloc[::3], rotation=45, ha='right', fontsize=8)
axes[0, 0].set_title('Monthly Revenue Trend (Green=Increase, Red=Decrease)', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('Revenue (£)')
axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'£{x:,.0f}'))

# 1b: Category Revenue Distribution (Pie)
cat_rev = orders.groupby('Category')['TotalPrice'].sum().sort_values(ascending=False).head(8)
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(cat_rev)))
wedges, texts, autotexts = axes[0, 1].pie(cat_rev, labels=cat_rev.index, autopct='%1.1f%%',
                                            colors=colors_pie, textprops={'fontsize': 9})
axes[0, 1].set_title('Revenue Share by Product Category', fontsize=13, fontweight='bold')

# 1c: Price Range Distribution
pr_counts = orders['PriceRange'].value_counts()
pr_colors = {'Low': '#3498db', 'Medium': '#f39c12', 'High': '#e74c3c', 'Premium': '#9b59b6'}
bars = axes[1, 0].bar(pr_counts.index, pr_counts.values,
                       color=[pr_colors.get(x, '#95a5a6') for x in pr_counts.index],
                       edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, pr_counts.values):
    pct = val / pr_counts.sum() * 100
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                     f'{pct:.1f}%', ha='center', fontweight='bold', fontsize=11)
axes[1, 0].set_title('Orders by Price Range (%)', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('Number of Orders')

# 1d: Weekend vs Weekday Sales
wk = orders.groupby('IsWeekend')['TotalPrice'].agg(['sum', 'count']).reset_index()
wk['Label'] = wk['IsWeekend'].map({0: 'Weekday', 1: 'Weekend'})
wk_colors = ['#3498db', '#e67e22']
bars = axes[1, 1].bar(wk['Label'], wk['sum'], color=wk_colors, edgecolor='black', linewidth=0.5)
for bar, val, cnt in zip(bars, wk['sum'], wk['count']):
    pct = val / wk['sum'].sum() * 100
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                     f'{pct:.1f}%\n({cnt:,} orders)', ha='center', fontweight='bold', fontsize=10)
axes[1, 1].set_title('Revenue: Weekday vs Weekend', fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel('Total Revenue (£)')

plt.tight_layout()
path1 = os.path.join(OUTPUT_DIR, 'insights_sales_overview.png')
plt.savefig(path1, dpi=150, bbox_inches='tight')
plt.close()
print(f'   Saved: {path1}')

# ============================================================
# FIGURE 2: Customer Behavior & What People Buy (2x2)
# ============================================================
print('2. Generating Customer Behavior Insights...')

customers = pd.read_sql('SELECT * FROM Dim_Customer', engine)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Customer Behavior & Purchase Patterns', fontsize=18, fontweight='bold', y=1.02)

# 2a: Churn Rate
churn_counts = customers['ChurnLabel'].value_counts()
labels = ['Active', 'Churned']
colors_ch = ['#2ecc71', '#e74c3c']
axes[0, 0].pie(churn_counts, labels=labels, autopct='%1.1f%%', colors=colors_ch,
               textprops={'fontsize': 12, 'fontweight': 'bold'}, startangle=90,
               explode=(0, 0.05))
axes[0, 0].set_title('Customer Churn Rate', fontsize=13, fontweight='bold')

# 2b: Top 10 Most Purchased Categories
top_cats = orders.groupby('Category')['Quantity'].sum().sort_values(ascending=True).tail(10)
bar_colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_cats)))
axes[0, 1].barh(top_cats.index, top_cats.values, color=bar_colors, edgecolor='black', linewidth=0.5)
for i, (val, name) in enumerate(zip(top_cats.values, top_cats.index)):
    pct = val / orders['Quantity'].sum() * 100
    axes[0, 1].text(val + 500, i, f'{pct:.1f}%', va='center', fontweight='bold', fontsize=9)
axes[0, 1].set_title('Top 10 Categories by Quantity Sold', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Total Quantity')

# 2c: Purchase Frequency Distribution
freq_bins = [0, 1, 3, 5, 10, 50, 1000]
freq_labels = ['1 order', '2-3', '4-5', '6-10', '11-50', '50+']
customers['FreqBin'] = pd.cut(customers['Frequency'], bins=freq_bins, labels=freq_labels, right=True)
freq_dist = customers['FreqBin'].value_counts().reindex(freq_labels)
freq_colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(freq_dist)))
bars = axes[1, 0].bar(freq_dist.index, freq_dist.values, color=freq_colors, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, freq_dist.values):
    pct = val / customers.shape[0] * 100
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                     f'{pct:.1f}%', ha='center', fontweight='bold', fontsize=9)
axes[1, 0].set_title('Purchase Frequency Distribution', fontsize=13, fontweight='bold')
axes[1, 0].set_ylabel('Number of Customers')
axes[1, 0].set_xlabel('Number of Orders')

# 2d: Top 10 Regions by Revenue
region_rev = orders.groupby('Region')['TotalPrice'].sum().sort_values(ascending=True).tail(10)
reg_colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(region_rev)))
axes[1, 1].barh(region_rev.index, region_rev.values, color=reg_colors, edgecolor='black', linewidth=0.5)
for i, val in enumerate(region_rev.values):
    pct = val / orders['TotalPrice'].sum() * 100
    axes[1, 1].text(val + 1000, i, f'{pct:.1f}%', va='center', fontweight='bold', fontsize=9)
axes[1, 1].set_title('Top 10 Regions by Revenue', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Revenue (£)')

plt.tight_layout()
path2 = os.path.join(OUTPUT_DIR, 'insights_customer_behavior.png')
plt.savefig(path2, dpi=150, bbox_inches='tight')
plt.close()
print(f'   Saved: {path2}')

# ============================================================
# FIGURE 3: Female Reviews Focus (2x2)
# ============================================================
print('3. Generating Female Reviews Insights...')

reviews = pd.read_sql('SELECT * FROM Fact_Reviews', engine)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Female E-Commerce Reviews Analysis', fontsize=18, fontweight='bold', y=1.02)

# 3a: Rating Distribution
rating_dist = reviews['Rating'].value_counts().sort_index()
r_colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']
bars = axes[0, 0].bar(rating_dist.index, rating_dist.values, color=r_colors, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, rating_dist.values):
    pct = val / len(reviews) * 100
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                     f'{pct:.1f}%', ha='center', fontweight='bold', fontsize=11)
axes[0, 0].set_title('Rating Distribution (Female Reviews)', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Star Rating')
axes[0, 0].set_ylabel('Number of Reviews')

# 3b: Sentiment by Department
dept_sent = reviews.groupby('DepartmentName')['SentimentScore'].mean().sort_values()
sent_colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in dept_sent.values]
axes[0, 1].barh(dept_sent.index, dept_sent.values, color=sent_colors, edgecolor='black', linewidth=0.5)
axes[0, 1].axvline(x=0, color='black', linewidth=1, linestyle='--')
axes[0, 1].set_title('Average Sentiment by Department', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Sentiment Score (Negative ← 0 → Positive)')

# 3c: Recommendation Rate by Class
class_rec = reviews.groupby('ClassName').agg(
    total=('Recommended', 'count'),
    recommended=('Recommended', 'sum')
).reset_index()
class_rec['pct'] = class_rec['recommended'] / class_rec['total'] * 100
class_rec = class_rec.sort_values('pct')
rec_colors = plt.cm.RdYlGn(class_rec['pct'].values / 100)
bars = axes[1, 0].barh(class_rec['ClassName'], class_rec['pct'], color=rec_colors, edgecolor='black', linewidth=0.5)
for i, (pct, total) in enumerate(zip(class_rec['pct'], class_rec['total'])):
    axes[1, 0].text(pct + 0.5, i, f'{pct:.1f}% ({total:,})', va='center', fontweight='bold', fontsize=9)
axes[1, 0].set_title('Recommendation Rate by Clothing Class', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Recommendation %')
axes[1, 0].set_xlim(0, 110)

# 3d: Age Group vs Rating Heatmap
reviews['AgeGroup'] = pd.cut(reviews['Age'], bins=[17, 25, 35, 45, 55, 100],
                              labels=['18-25', '26-35', '36-45', '46-55', '56+'])
age_rating = reviews.groupby(['AgeGroup', 'Rating']).size().unstack(fill_value=0)
sns.heatmap(age_rating, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 1], linewidths=0.5)
axes[1, 1].set_title('Age Group vs Rating (Heatmap)', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Star Rating')
axes[1, 1].set_ylabel('Age Group')

plt.tight_layout()
path3 = os.path.join(OUTPUT_DIR, 'insights_female_reviews.png')
plt.savefig(path3, dpi=150, bbox_inches='tight')
plt.close()
print(f'   Saved: {path3}')

# ============================================================
# FIGURE 4: Correlations & Percentages Between Columns (2x2)
# ============================================================
print('4. Generating Correlation & Column Analysis...')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Data Correlations & Feature Relationships', fontsize=18, fontweight='bold', y=1.02)

# 4a: RFM Correlation Heatmap
rfm = customers[['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'ChurnLabel']].dropna()
corr = rfm.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[0, 0],
            mask=mask, vmin=-1, vmax=1, linewidths=0.5)
axes[0, 0].set_title('RFM Feature Correlations', fontsize=13, fontweight='bold')

# 4b: Recency vs Monetary (colored by churn)
sample = rfm.sample(min(2000, len(rfm)), random_state=42)
scatter = axes[0, 1].scatter(sample['Recency'], sample['Monetary'],
                              c=sample['ChurnLabel'], cmap='RdYlGn_r', alpha=0.5, s=15, edgecolors='none')
axes[0, 1].set_xlabel('Recency (days)')
axes[0, 1].set_ylabel('Monetary (£)')
axes[0, 1].set_title('Recency vs Monetary (Color=Churn)', fontsize=13, fontweight='bold')
legend = axes[0, 1].legend(*scatter.legend_elements(), title='Churn', labels=['Active', 'Churned'])

# 4c: Review Sentiment vs Rating Correlation
sent_by_rating = reviews.groupby('Rating')['SentimentScore'].agg(['mean', 'std']).reset_index()
axes[1, 0].bar(sent_by_rating['Rating'], sent_by_rating['mean'],
               yerr=sent_by_rating['std'], color='#3498db', edgecolor='black',
               linewidth=0.5, capsize=5, alpha=0.8)
axes[1, 0].set_title('Average Sentiment Score per Star Rating', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Star Rating')
axes[1, 0].set_ylabel('Mean Sentiment Score')
axes[1, 0].axhline(y=0, color='red', linewidth=1, linestyle='--')

# 4d: Quarterly Revenue Growth Rate
quarterly = orders.groupby([orders['FullDate'].dt.to_period('Q')])['TotalPrice'].sum().reset_index()
quarterly.columns = ['Quarter', 'Revenue']
quarterly['Quarter'] = quarterly['Quarter'].astype(str)
quarterly['GrowthRate'] = quarterly['Revenue'].pct_change() * 100
growth_colors = ['#2ecc71' if g >= 0 else '#e74c3c' for g in quarterly['GrowthRate'].fillna(0)]

axes[1, 1].bar(range(len(quarterly)), quarterly['GrowthRate'].fillna(0),
               color=growth_colors, edgecolor='black', linewidth=0.5)
axes[1, 1].set_xticks(range(len(quarterly)))
axes[1, 1].set_xticklabels(quarterly['Quarter'], rotation=45, ha='right', fontsize=9)
axes[1, 1].axhline(y=0, color='black', linewidth=1, linestyle='--')
axes[1, 1].set_title('Quarterly Revenue Growth Rate (%)', fontsize=13, fontweight='bold')
axes[1, 1].set_ylabel('Growth Rate (%)')

plt.tight_layout()
path4 = os.path.join(OUTPUT_DIR, 'insights_correlations.png')
plt.savefig(path4, dpi=150, bbox_inches='tight')
plt.close()
print(f'   Saved: {path4}')

print('\n' + '=' * 60)
print('  ALL INSIGHT VISUALIZATIONS GENERATED SUCCESSFULLY!')
print('=' * 60)
print(f'\nFiles saved to: {OUTPUT_DIR}')
print('  1. insights_sales_overview.png')
print('  2. insights_customer_behavior.png')
print('  3. insights_female_reviews.png')
print('  4. insights_correlations.png')
