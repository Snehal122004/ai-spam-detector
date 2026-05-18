"""
============================================================
  SPAM EMAIL / SMS DETECTOR — WEEK 1
  IBM SkillsBuild AI Internship Project
============================================================
  What this file covers:
    Step 1  →  Load the dataset
    Step 2  →  Basic exploration (shape, samples, info)
    Step 3  →  Spam vs Ham distribution
    Step 4  →  Message length analysis
    Step 5  →  Text cleaning (lowercase, punctuation, stopwords)
    Step 6  →  Top spam trigger words bar chart
    Step 7  →  Spam word cloud
    Step 8  →  Save cleaned dataset for Week 2
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import string
import warnings
warnings.filterwarnings('ignore')

# ── Try to import optional libraries ──────────────────────
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("Note: wordcloud not installed. Run: pip install wordcloud")

try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    # Fallback stopwords if NLTK not available
    STOP_WORDS = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
        'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she',
        'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
        'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
        'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
        'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
        'against', 'between', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
        'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'both', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
        'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
        'u', 'ur', 'r', 'im', 'ok', 'got', 'get', 'go', 'hi', 'hey',
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 1: LOAD THE DATASET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 55)
print("  WEEK 1 — SPAM DETECTOR  ")
print("=" * 55)
print("\n📂 STEP 1: Loading dataset...")

# ---------------------------------------------------------
# HOW TO GET THE REAL DATASET (for your actual internship):
#   1. Go to https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
#   2. Download 'spam.csv'
#   3. Place it in the same folder as this file
#   4. Uncomment the real loader below and comment out the demo one
# ---------------------------------------------------------

# ── Real dataset loader (uncomment when you have the file) ──
# df = pd.read_csv('spam.csv', encoding='latin-1')
# df = df[['v1', 'v2']]               # keep only label and message columns
# df.columns = ['label', 'message']   # rename for clarity

# ── Demo dataset (works right now, no download needed) ──────
df = pd.read_csv('spam.csv', encoding='latin-1')

# FIX: rename columns FIRST (VERY IMPORTANT)
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print(f"\n   Rows loaded    : {len(df)}")
print(f"   Columns        : {list(df.columns)}")
print(f"\n   First 5 rows:")
print(df.head().to_string(index=False))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 2: BASIC DATA EXPLORATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 55)
print("  STEP 2: Data Exploration")
print("=" * 55)

# Dataset shape
print(f"\n   Shape: {df.shape}  ({df.shape[0]} messages, {df.shape[1]} columns)")

# Label distribution
counts = df['label'].value_counts()
print(f"\n   Label distribution:")
print(f"   ├── Ham  (real)  : {counts.get('ham', 0)} messages")
print(f"   └── Spam         : {counts.get('spam', 0)} messages")

spam_pct = round(counts.get('spam', 0) / len(df) * 100, 1)
ham_pct  = round(counts.get('ham',  0) / len(df) * 100, 1)
print(f"\n   Spam % : {spam_pct}%  |  Ham % : {ham_pct}%")

# Missing values
missing = df.isnull().sum()
print(f"\n   Missing values:")
for col, cnt in missing.items():
    status = "None" if cnt == 0 else f"{cnt} MISSING ⚠"
    print(f"   ├── {col:10s}: {status}")

# Sample messages
print(f"\n   Sample SPAM message:")
print(f"   → {df[df['label']=='spam']['message'].iloc[0][:90]}...")
print(f"\n   Sample HAM message:")
print(f"   → {df[df['label']=='ham']['message'].iloc[0][:90]}...")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 3: SPAM vs HAM DISTRIBUTION CHART
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 55)
print("  STEP 3: Plotting Spam vs Ham Distribution")
print("=" * 55)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#F8F9FA')

# ── Pie chart ──────────────────────────────────────────
ax1 = axes[0]
ax1.set_facecolor('#F8F9FA')
colors = ['#2196F3', '#F44336']
explode = (0.04, 0.04)
wedges, texts, autotexts = ax1.pie(
    [counts.get('ham', 0), counts.get('spam', 0)],
    labels=['Ham (Real)', 'Spam'],
    autopct='%1.1f%%',
    colors=colors,
    explode=explode,
    startangle=90,
    textprops={'fontsize': 12},
    wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'}
)
for at in autotexts:
    at.set_fontsize(12)
    at.set_fontweight('bold')
    at.set_color('white')
ax1.set_title('Spam vs Ham Distribution', fontsize=14, fontweight='bold', pad=15)

# ── Bar chart ──────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor('#F8F9FA')
bar_colors = ['#2196F3', '#F44336']
bars = ax2.bar(['Ham (Real)', 'Spam'],
               [counts.get('ham', 0), counts.get('spam', 0)],
               color=bar_colors, width=0.5, edgecolor='white', linewidth=1.5)

for bar, val in zip(bars, [counts.get('ham', 0), counts.get('spam', 0)]):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.5,
             str(val), ha='center', va='bottom',
             fontsize=13, fontweight='bold')

ax2.set_title('Message Count by Category', fontsize=14, fontweight='bold')
ax2.set_ylabel('Number of Messages', fontsize=11)
ax2.set_ylim(0, max(counts) * 1.15)
ax2.spines[['top', 'right']].set_visible(False)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

plt.suptitle('SMS Spam Dataset — Overview', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('chart1_distribution.png', dpi=150, bbox_inches='tight',
            facecolor='#F8F9FA')
plt.close()
print("   Saved → chart1_distribution.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 4: MESSAGE LENGTH ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 55)
print("  STEP 4: Message Length Analysis")
print("=" * 55)

df['message_length'] = df['message'].apply(len)
df['word_count']     = df['message'].apply(lambda x: len(x.split()))

spam_df = df[df['label'] == 'spam']
ham_df  = df[df['label'] == 'ham']

print(f"\n   Average message length:")
print(f"   ├── Spam messages : {spam_df['message_length'].mean():.0f} characters")
print(f"   └── Ham messages  : {ham_df['message_length'].mean():.0f} characters")

print(f"\n   Average word count:")
print(f"   ├── Spam messages : {spam_df['word_count'].mean():.1f} words")
print(f"   └── Ham messages  : {ham_df['word_count'].mean():.1f} words")

# Plot length distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#F8F9FA')

for ax in axes:
    ax.set_facecolor('#F8F9FA')
    ax.spines[['top', 'right']].set_visible(False)

# Character length histogram
axes[0].hist(ham_df['message_length'],  bins=15, alpha=0.7,
             color='#2196F3', label='Ham',  edgecolor='white')
axes[0].hist(spam_df['message_length'], bins=15, alpha=0.7,
             color='#F44336', label='Spam', edgecolor='white')
axes[0].set_title('Message Length (characters)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Number of Characters')
axes[0].set_ylabel('Frequency')
axes[0].legend(fontsize=11)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')

# Word count histogram
axes[1].hist(ham_df['word_count'],  bins=12, alpha=0.7,
             color='#2196F3', label='Ham',  edgecolor='white')
axes[1].hist(spam_df['word_count'], bins=12, alpha=0.7,
             color='#F44336', label='Spam', edgecolor='white')
axes[1].set_title('Message Length (words)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Number of Words')
axes[1].set_ylabel('Frequency')
axes[1].legend(fontsize=11)
axes[1].grid(axis='y', alpha=0.3, linestyle='--')

plt.suptitle('Are spam messages longer than ham?', fontsize=15,
             fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('chart2_length_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='#F8F9FA')
plt.close()
print("   Saved → chart2_length_analysis.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 5: TEXT CLEANING FUNCTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 55)
print("  STEP 5: Text Cleaning")
print("=" * 55)

def clean_text(text):
    """
    Clean a raw SMS/email message for NLP processing.
    Steps:
      1. Lowercase everything
      2. Remove URLs (http links)
      3. Remove punctuation and numbers
      4. Remove extra whitespace
      5. Remove stopwords (common words like 'the', 'is', 'and')
    Returns: cleaned string
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)         # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)            # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()           # remove extra spaces
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(words)

# Apply cleaning
df['cleaned_message'] = df['message'].apply(clean_text)

# Show before/after comparison
print("\n   Cleaning example (SPAM):")
sample = df[df['label'] == 'spam'].iloc[0]
print(f"   BEFORE: {sample['message'][:80]}...")
print(f"   AFTER : {sample['cleaned_message'][:80]}...")

print("\n   Cleaning example (HAM):")
sample2 = df[df['label'] == 'ham'].iloc[0]
print(f"   BEFORE: {sample2['message'][:80]}")
print(f"   AFTER : {sample2['cleaned_message'][:80]}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 6: TOP SPAM TRIGGER WORDS BAR CHART
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 55)
print("  STEP 6: Top Spam Trigger Words")
print("=" * 55)

from collections import Counter

def get_top_words(df_subset, n=20):
    """Get the n most frequent words from a set of messages."""
    all_words = ' '.join(df_subset['cleaned_message'].tolist()).split()
    return Counter(all_words).most_common(n)

spam_words = get_top_words(df[df['label'] == 'spam'], n=20)
ham_words  = get_top_words(df[df['label'] == 'ham'],  n=20)

spam_labels, spam_vals = zip(*spam_words)
ham_labels,  ham_vals  = zip(*ham_words)

print(f"\n   Top 10 spam words:")
for word, count in spam_words[:10]:
    bar = '█' * min(count, 30)
    print(f"   {word:12s} {bar} ({count})")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#F8F9FA')

# Spam words
axes[0].set_facecolor('#F8F9FA')
bars_s = axes[0].barh(spam_labels[::-1], spam_vals[::-1],
                       color='#F44336', alpha=0.85, edgecolor='white')
axes[0].set_title('Top 20 Words in SPAM Messages',
                   fontsize=13, fontweight='bold', color='#C62828')
axes[0].set_xlabel('Frequency', fontsize=11)
axes[0].spines[['top', 'right']].set_visible(False)
axes[0].grid(axis='x', alpha=0.3, linestyle='--')
for bar, val in zip(bars_s, spam_vals[::-1]):
    axes[0].text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                 str(val), va='center', fontsize=9)

# Ham words
axes[1].set_facecolor('#F8F9FA')
bars_h = axes[1].barh(ham_labels[::-1], ham_vals[::-1],
                       color='#2196F3', alpha=0.85, edgecolor='white')
axes[1].set_title('Top 20 Words in HAM Messages',
                   fontsize=13, fontweight='bold', color='#0D47A1')
axes[1].set_xlabel('Frequency', fontsize=11)
axes[1].spines[['top', 'right']].set_visible(False)
axes[1].grid(axis='x', alpha=0.3, linestyle='--')
for bar, val in zip(bars_h, ham_vals[::-1]):
    axes[1].text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                 str(val), va='center', fontsize=9)

plt.suptitle('What words appear most in Spam vs Real messages?',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('chart3_top_words.png', dpi=150, bbox_inches='tight',
            facecolor='#F8F9FA')
plt.close()
print("   Saved → chart3_top_words.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 7: SPAM WORD CLOUD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 55)
print("  STEP 7: Spam Word Cloud")
print("=" * 55)

if WORDCLOUD_AVAILABLE:
    spam_text = ' '.join(df[df['label'] == 'spam']['cleaned_message'].tolist())
    ham_text  = ' '.join(df[df['label'] == 'ham']['cleaned_message'].tolist())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#1A1A2E')

    wc_spam = WordCloud(
        width=600, height=400,
        background_color='#1A1A2E',
        colormap='Reds',
        max_words=80,
        contour_color='#F44336',
        contour_width=1
    ).generate(spam_text)

    wc_ham = WordCloud(
        width=600, height=400,
        background_color='#1A1A2E',
        colormap='Blues',
        max_words=80,
        contour_color='#2196F3',
        contour_width=1
    ).generate(ham_text)

    axes[0].imshow(wc_spam, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title('SPAM Word Cloud', fontsize=15, fontweight='bold',
                       color='#F44336', pad=10)

    axes[1].imshow(wc_ham, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title('HAM Word Cloud', fontsize=15, fontweight='bold',
                       color='#42A5F5', pad=10)

    plt.suptitle('Visual pattern of spam vs real messages',
                 fontsize=15, fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig('chart4_wordcloud.png', dpi=150, bbox_inches='tight',
                facecolor='#1A1A2E')
    plt.close()
    print("   Saved → chart4_wordcloud.png")
else:
    print("   Skipped (wordcloud library not installed)")
    print("   Install with: pip install wordcloud")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 8: SAVE CLEANED DATASET FOR WEEK 2
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 55)
print("  STEP 8: Saving Cleaned Dataset")
print("=" * 55)

# Encode label as number (needed for ML in Week 2)
df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})

# Save cleaned version
df.to_csv('spam_cleaned.csv', index=False)

print(f"\n   Cleaned dataset saved → spam_cleaned.csv")
print(f"   Rows     : {len(df)}")
print(f"   Columns  : {list(df.columns)}")
print(f"\n   label_num encoding: spam=1, ham=0")
print(f"   This file will be loaded in Week 2 for model training.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  WEEK 1 SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 55)
print("  WEEK 1 COMPLETE — Summary")
print("=" * 55)
print(f"""
   What you did this week:
   ✓  Loaded the SMS Spam dataset ({len(df)} messages)
   ✓  Explored spam vs ham distribution
   ✓  Analyzed message length patterns
   ✓  Built a text cleaning pipeline (lowercase,
      remove URLs, punctuation, stopwords)
   ✓  Found top 20 spam trigger words
   ✓  Created a visual word cloud
   ✓  Saved cleaned data for Week 2

   Charts saved:
   ✓  chart1_distribution.png
   ✓  chart2_length_analysis.png
   ✓  chart3_top_words.png
   ✓  chart4_wordcloud.png  (if wordcloud installed)

   Coming up in Week 2:
   →  TF-IDF Vectorization
   →  Train Naive Bayes classifier
   →  Achieve 97%+ accuracy
   →  Confusion matrix evaluation
""")
print("=" * 55)
