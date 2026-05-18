import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ===========================================================
# STEP 1: LOAD CLEANED DATA
# ===========================================================
print("=" * 58)
print("  WEEK 2 - SPAM DETECTOR AI MODEL TRAINING")
print("=" * 58)
print("\nSTEP 1: Loading cleaned dataset...")

df = pd.read_csv('spam_cleaned.csv')

if 'cleaned_message' not in df.columns:
    df['cleaned_message'] = df['message']

if 'label_num' not in df.columns:
    df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})

df = df.dropna(subset=['cleaned_message', 'label_num'])

print(f"\n   Total messages loaded : {len(df)}")
print(f"   Spam messages         : {df[df['label']=='spam'].shape[0]}")
print(f"   Ham  messages         : {df[df['label']=='ham'].shape[0]}")
print(f"\n   Sample spam message:")
print(f"   -> {df[df['label']=='spam']['cleaned_message'].iloc[0][:70]}...")


# ===========================================================
# STEP 2: TF-IDF VECTORIZATION
# ===========================================================
print("\n" + "=" * 58)
print("  STEP 2: TF-IDF Vectorization (Text -> Numbers)")
print("=" * 58)
print("""
   What is TF-IDF?
   ----------------
   Machine cannot read text. It only understands numbers.
   TF-IDF gives every word a number (score).

   TF  = Term Frequency
         How many times a word appears in one message

   IDF = Inverse Document Frequency
         How rare the word is across all messages

   Example:
   "FREE"  appears a lot in spam  -> HIGH score
   "the"   appears everywhere     -> LOW  score

   Result: AI learns that FREE, WIN, PRIZE = spam signals
""")

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=1
)

X = tfidf.fit_transform(df['cleaned_message'])
y = df['label_num']

feature_names = tfidf.get_feature_names_out()

print(f"   TF-IDF Matrix shape : {X.shape}")
print(f"   Messages x Words    : {X.shape[0]} x {X.shape[1]}")
print(f"   Vocabulary size     : {len(tfidf.vocabulary_)} unique words")
print(f"\n   Top 15 words found  : {', '.join(feature_names[:15])}")


# ===========================================================
# STEP 3: TRAIN / TEST SPLIT
# ===========================================================
print("\n" + "=" * 58)
print("  STEP 3: Train / Test Split")
print("=" * 58)
print("""
   Why do we split data?
   ----------------------
   If we train and test on the same data,
   the model just memorizes it. That is cheating!

   So we split:
   80% -> Model learns from this  (Training)
   20% -> Model is tested on this (Testing)

   This is like:
   Reading a book  = Training
   Writing an exam = Testing
""")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"   Total data           : {X.shape[0]} messages")
print(f"   Training data (80%)  : {X_train.shape[0]} messages")
print(f"   Testing  data (20%)  : {X_test.shape[0]} messages")
print(f"\n   Training set breakdown:")
print(f"   Spam : {sum(y_train == 1)}")
print(f"   Ham  : {sum(y_train == 0)}")


# ===========================================================
# STEP 4: NAIVE BAYES MODEL TRAINING
# ===========================================================
print("\n" + "=" * 58)
print("  STEP 4: Naive Bayes Model Training")
print("=" * 58)
print("""
   What is Naive Bayes?
   ---------------------
   A probability-based algorithm.
   Same algorithm used inside Gmail spam filter!

   How it works:
   -> "FREE" appears in spam 90% of the time
   -> "meeting" appears in ham 95% of the time
   -> New message arrives -> checks which words it has
   -> Calculates probability -> decides Spam or Ham

   Advantages:
   * Trains in under 2 seconds on any laptop
   * Works very well even with small datasets
   * Best algorithm for text classification tasks
""")

print("   Training model now...")

model = MultinomialNB(alpha=0.1)
model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train)) * 100
print(f"   Training complete!")
print(f"   Training Accuracy : {train_acc:.2f}%")


# ===========================================================
# STEP 5: MODEL ACCURACY ON TEST DATA
# ===========================================================
print("\n" + "=" * 58)
print("  STEP 5: Model Accuracy")
print("=" * 58)

y_pred       = model.predict(X_test)
y_pred_prob  = model.predict_proba(X_test)[:, 1]
accuracy     = accuracy_score(y_test, y_pred) * 100
correct      = sum(y_pred == y_test)
incorrect    = sum(y_pred != y_test)

print(f"""
   +------------------------------------------+
   |                                          |
   |   TEST ACCURACY  :  {accuracy:.2f}%             |
   |                                          |
   +------------------------------------------+
""")
print(f"   Total test messages : {len(y_test)}")
print(f"   Correct predictions : {correct}")
print(f"   Wrong  predictions  : {incorrect}")

if accuracy >= 97:
    print("\n   EXCELLENT! Accuracy is above 97%")
elif accuracy >= 94:
    print("\n   VERY GOOD! Accuracy is above 94%")
else:
    print("\n   Good accuracy. More data will improve it further.")


# ===========================================================
# STEP 6: CONFUSION MATRIX CHART
# ===========================================================
print("\n" + "=" * 58)
print("  STEP 6: Confusion Matrix")
print("=" * 58)
print("""
   What is a Confusion Matrix?
   ----------------------------
   Shows exactly what the model got right and wrong.

   True Negative  (TN) : Ham  -> correctly predicted Ham
   False Positive (FP) : Ham  -> wrongly predicted Spam
   False Negative (FN) : Spam -> wrongly predicted Ham
   True Positive  (TP) : Spam -> correctly predicted Spam

   TN + TP = correct   (we want these HIGH)
   FN + FP = mistakes  (we want these LOW)
""")

cm             = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"   True Negative  (Ham  -> Ham ) : {tn}")
print(f"   False Positive (Ham  -> Spam) : {fp}")
print(f"   False Negative (Spam -> Ham ) : {fn}")
print(f"   True Positive  (Spam -> Spam) : {tp}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#F8F9FA')

ax1 = axes[0]
ax1.set_facecolor('#F8F9FA')
colors = np.array([[0.2, 0.8], [0.8, 0.2]])
ax1.imshow(colors, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

cm_labels = np.array([
    [f'True Negative\n(Ham -> Ham)\n{tn}',
     f'False Positive\n(Ham -> Spam)\n{fp}'],
    [f'False Negative\n(Spam -> Ham)\n{fn}',
     f'True Positive\n(Spam -> Spam)\n{tp}']
])

for i in range(2):
    for j in range(2):
        ax1.text(j, i, cm_labels[i][j],
                 ha='center', va='center',
                 fontsize=11, fontweight='bold', color='#1A1A1A')

ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Predicted: Ham', 'Predicted: Spam'], fontsize=10)
ax1.set_yticklabels(['Actual: Ham',    'Actual: Spam'],    fontsize=10)
ax1.set_title('Confusion Matrix', fontsize=13, fontweight='bold', pad=12)

green_patch = mpatches.Patch(color='#90EE90', label='Correct')
red_patch   = mpatches.Patch(color='#FF9999', label='Wrong')
ax1.legend(handles=[green_patch, red_patch],
           loc='lower center', bbox_to_anchor=(0.5, -0.18),
           ncol=2, fontsize=10)

ax2 = axes[1]
ax2.set_facecolor('#F8F9FA')

total_spam  = fn + tp
total_ham   = tn + fp
spam_recall = (tp / total_spam * 100) if total_spam > 0 else 0
ham_recall  = (tn / total_ham  * 100) if total_ham  > 0 else 0
precision   = (tp / (tp + fp)  * 100) if (tp + fp)  > 0 else 0

metric_names = ['Accuracy', 'Spam\nDetected', 'Ham\nProtected', 'Precision']
metric_vals  = [accuracy, spam_recall, ham_recall, precision]
bar_colors   = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']

bars = ax2.bar(metric_names, metric_vals,
               color=bar_colors, width=0.5,
               edgecolor='white', linewidth=1.5)

for bar, val in zip(bars, metric_vals):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.8,
             f'{val:.1f}%', ha='center', va='bottom',
             fontsize=11, fontweight='bold')

ax2.set_ylim(0, 118)
ax2.set_ylabel('Percentage (%)', fontsize=11)
ax2.set_title('Model Performance Metrics', fontsize=13, fontweight='bold')
ax2.spines[['top', 'right']].set_visible(False)
ax2.axhline(y=97, color='red', linestyle='--',
            linewidth=1, alpha=0.4, label='97% target')
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

plt.suptitle(f'Naive Bayes Model  —  Accuracy: {accuracy:.2f}%',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('chart5_confusion_matrix.png', dpi=150,
            bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print("\n   Saved -> chart5_confusion_matrix.png")


# ===========================================================
# STEP 7: CLASSIFICATION REPORT
# ===========================================================
print("\n" + "=" * 58)
print("  STEP 7: Classification Report")
print("=" * 58)
print("""
   3 Important Metrics:
   ---------------------
   Precision : Of all messages called Spam, how many were really Spam?
   Recall    : Of all real Spam messages, how many did we catch?
   F1-Score  : Balance between Precision and Recall
""")

report = classification_report(
    y_test, y_pred,
    target_names=['Ham', 'Spam'],
    output_dict=True
)

print(f"   {'Category':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Count':>8}")
print(f"   {'-'*52}")
for label in ['Ham', 'Spam']:
    r = report[label]
    print(f"   {label:<12} "
          f"{r['precision']*100:>9.1f}% "
          f"{r['recall']*100:>9.1f}% "
          f"{r['f1-score']*100:>9.1f}% "
          f"{int(r['support']):>8}")
print(f"   {'-'*52}")
print(f"   {'Overall':<12} {'':>10} {'':>10} {accuracy:>9.1f}% {len(y_test):>8}")


# ===========================================================
# STEP 8: LIVE SPAM PREDICTION TEST
# ===========================================================
print("\n" + "=" * 58)
print("  STEP 8: Live Spam Prediction Test")
print("=" * 58)

STOP_WORDS = {
    'i','me','my','we','our','you','your','he','him','his',
    'she','her','it','its','they','them','their','what','which',
    'this','that','these','those','am','is','are','was','were',
    'be','been','have','has','had','do','does','did','a','an',
    'the','and','but','if','or','as','of','at','by','for',
    'with','to','from','in','out','on','so','than','too','very',
    'u','ur','r','ok','hi','hey','just','get','got','go','im'
}

def clean_text(text):
    text  = str(text).lower()
    text  = re.sub(r'http\S+|www\S+', '', text)
    text  = re.sub(r'[^a-zA-Z\s]', '', text)
    text  = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(words)

def predict_spam(message):
    cleaned    = clean_text(message)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    spam_prob  = probability[1] * 100
    ham_prob   = probability[0] * 100
    label      = "SPAM" if prediction == 1 else "HAM (REAL)"
    if prediction == 1:
        risk = "HIGH RISK"   if spam_prob >= 90 else \
               "MEDIUM RISK" if spam_prob >= 70 else "LOW RISK"
    else:
        risk = "SAFE"
    return {
        'label'     : label,
        'risk'      : risk,
        'spam_prob' : spam_prob,
        'ham_prob'  : ham_prob
    }

test_messages = [
    "Congratulations! You have WON a FREE prize worth 1000 pounds. Call now to CLAIM!",
    "URGENT: Your SBI account will be blocked. Update KYC at www.sbi-verify.net",
    "FREE entry! Win FA Cup tickets. Text WIN to 87121. Std rate applies.",
    "Dear customer, you have won Rs.25000 in lucky draw. Call 9876543210 to claim.",
    "Hey, can we meet tomorrow at 5pm for the project discussion?",
    "Mom I will be home late tonight. Please do not wait for dinner.",
    "Can you send me the notes from today's class? I missed the lecture.",
    "Great job on the presentation! The client was really impressed."
]

print("\n   Testing 8 messages — 4 spam + 4 real:\n")
line = "-" * 58

for i, msg in enumerate(test_messages):
    result   = predict_spam(msg)
    expected = "SPAM" if i < 4 else "HAM"
    print(f"   {line}")
    print(f"   Message  : {msg[:54]}...")
    print(f"   Expected : {expected}")
    print(f"   Result   : {result['label']}  |  {result['risk']}")
    print(f"   Spam %   : {result['spam_prob']:.1f}%   "
          f"Ham %  : {result['ham_prob']:.1f}%")

print(f"   {line}")


# ===========================================================
# STEP 9: TOP SPAM vs HAM WORDS CHART  (FIXED — both sides)
# ===========================================================
print("\n" + "=" * 58)
print("  STEP 9: What did the AI learn?")
print("=" * 58)

feature_names = tfidf.get_feature_names_out()
spam_log_prob = model.feature_log_prob_[1]
ham_log_prob  = model.feature_log_prob_[0]
spam_scores   = spam_log_prob - ham_log_prob

# Top 15 spam words
top_spam_idx   = np.argsort(spam_scores)[-15:][::-1]
top_spam_words = [feature_names[i] for i in top_spam_idx]
top_spam_vals  = [float(spam_scores[i]) for i in top_spam_idx]

# Top 15 ham words
top_ham_idx    = np.argsort(spam_scores)[:15]
top_ham_words  = [feature_names[i] for i in top_ham_idx]
top_ham_vals   = [float(-spam_scores[i]) for i in top_ham_idx]

print("\n   Top 10 SPAM words the AI learned:")
for w, s in zip(top_spam_words[:10], top_spam_vals[:10]):
    bar = '#' * min(int(s * 4), 28)
    print(f"   {w:20s} {bar} ({s:.2f})")

print("\n   Top 10 HAM words the AI learned:")
for w, s in zip(top_ham_words[:10], top_ham_vals[:10]):
    bar = '#' * min(int(s * 4), 28)
    print(f"   {w:20s} {bar} ({s:.2f})")

# Chart — FIXED: both sides now visible correctly
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor('#F8F9FA')

# ── LEFT chart — Spam ──────────────────────────────────
sw    = top_spam_words[::-1]
sv    = top_spam_vals[::-1]
y_pos = np.arange(len(sw))
ax1   = axes[0]
ax1.set_facecolor('#F8F9FA')
ax1.barh(y_pos, sv, color='#F44336', alpha=0.85,
         edgecolor='white', linewidth=1.2, height=0.6)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(sw, fontsize=11)
ax1.set_xlabel('Spam Score (higher = stronger spam signal)', fontsize=11)
ax1.set_title('Top 15 SPAM Words\n(AI learned these as SPAM signals)',
              fontsize=12, fontweight='bold', color='#C62828')
ax1.spines[['top', 'right']].set_visible(False)
ax1.grid(axis='x', alpha=0.3, linestyle='--')
for i, val in enumerate(sv):
    ax1.text(val + 0.05, i, f'{val:.1f}',
             va='center', fontsize=9,
             color='#C62828', fontweight='bold')

# ── RIGHT chart — Ham ──────────────────────────────────
hw     = top_ham_words[::-1]
hv     = top_ham_vals[::-1]
y_pos2 = np.arange(len(hw))
ax2    = axes[1]
ax2.set_facecolor('#F8F9FA')
ax2.barh(y_pos2, hv, color='#2196F3', alpha=0.85,
         edgecolor='white', linewidth=1.2, height=0.6)
ax2.set_yticks(y_pos2)
ax2.set_yticklabels(hw, fontsize=11)
ax2.set_xlabel('Ham Score (higher = stronger real message signal)', fontsize=11)
ax2.set_title('Top 15 HAM Words\n(AI learned these as REAL message signals)',
              fontsize=12, fontweight='bold', color='#0D47A1')
ax2.spines[['top', 'right']].set_visible(False)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
for i, val in enumerate(hv):
    ax2.text(val + 0.05, i, f'{val:.1f}',
             va='center', fontsize=9,
             color='#0D47A1', fontweight='bold')

plt.suptitle('What did the AI learn?  —  Spam vs Ham Word Patterns',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('chart6_model_learned_words.png', dpi=150,
            bbox_inches='tight', facecolor='#F8F9FA')
plt.close()
print("\n   Saved -> chart6_model_learned_words.png")


# ===========================================================
# STEP 10: SAVE MODEL FOR WEEK 3
# ===========================================================
print("\n" + "=" * 58)
print("  STEP 10: Save Model for Week 3")
print("=" * 58)
print("""
   Why save the model?
   --------------------
   Training takes time. We do not want to retrain every time.
   We save the trained model to a .pkl file.

   In Week 3, Streamlit UI will load this saved model
   and use it directly to make predictions instantly.
""")

with open('spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("   spam_model.pkl       -> Saved!")
print("   tfidf_vectorizer.pkl -> Saved!")

with open('spam_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    loaded_tfidf = pickle.load(f)

test_msg   = "FREE prize CLAIM now call 09061234567"
test_clean = clean_text(test_msg)
test_vec   = loaded_tfidf.transform([test_clean])
test_pred  = loaded_model.predict(test_vec)[0]
result_lbl = "SPAM" if test_pred == 1 else "HAM"

print(f"\n   Verification test:")
print(f"   Message    : {test_msg}")
print(f"   Prediction : {result_lbl}  <- Model loaded correctly!")


# ===========================================================
# WEEK 2 COMPLETE SUMMARY
# ===========================================================
print("\n" + "=" * 58)
print("  WEEK 2 COMPLETE!")
print("=" * 58)
print(f"""
   What was done this week:
   -------------------------
   Step 1  : Loaded spam_cleaned.csv
   Step 2  : TF-IDF Vectorization (text -> numbers)
   Step 3  : 80/20 Train/Test Split
   Step 4  : Naive Bayes Model Trained
   Step 5  : Test Accuracy = {accuracy:.2f}%
   Step 6  : Confusion Matrix chart saved
   Step 7  : Classification Report printed
   Step 8  : 8 live messages tested
   Step 9  : Spam vs Ham word chart saved (RED + BLUE)
   Step 10 : Model saved to .pkl files

   Files created in SpamDetector folder:
   ---------------------------------------
   chart5_confusion_matrix.png
   chart6_model_learned_words.png  <- RED + BLUE both visible
   spam_model.pkl
   tfidf_vectorizer.pkl

   Coming in Week 3:
   ------------------
   -> Build Streamlit web app UI
   -> Paste any message -> Get instant prediction
   -> Word highlight feature
   -> Deploy app on the internet
""")
print("=" * 58)