# Spam Detector AI
### IBM SkillsBuild AICTE 6-Week AI/ML Internship Project

## What is this project?

A machine learning web application that detects whether an
email or SMS message is **SPAM or REAL** using Natural Language
Processing (NLP) and Naive Bayes classification.

The app supports both **English** and **Indian SMS** (Hinglish)
spam patterns — fake KYC alerts, OTP frauds, lottery scams,
and more.

---

## Features

- **Instant Spam Detection** — paste any message, get result in seconds
- **Confidence Score** — shows how sure the AI is (0–100%)
- **Word Highlight** — spam trigger words highlighted in red
- **Spam Word Cloud** — visual of most common spam words
- **Indian SMS Support** — detects Hinglish spam patterns
- **Bulk CSV Upload** — check hundreds of messages at once
- **Downloadable Results** — export spam check results as CSV

---

## App Screenshots

### Check Message Tab
- Paste any email or SMS
- Click Check → instant SPAM or REAL result
- Red highlighted words show spam signals

### Word Cloud Tab
- Visual word cloud of spam vs real messages
- Top 10 spam trigger words bar chart

### Indian SMS Test Tab
- Tests 40 Indian messages (spam + real)
- Shows accuracy on Hinglish messages

### Bulk CSV Upload Tab
- Upload CSV with message column
- Check all messages at once
- Download results CSV

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Programming language |
| Scikit-learn | ML model (Naive Bayes, TF-IDF) |
| NLTK | Text preprocessing |
| Streamlit | Web app UI |
| WordCloud | Word cloud visualization |
| Matplotlib | Charts and graphs |
| Pandas | Data handling |
| NumPy | Numerical operations |

---

## Dataset

- **UCI SMS Spam Collection** — 5,572 labeled messages
  (Kaggle: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Custom Indian SMS Dataset** — 50 Hinglish spam/ham messages
- **Combined Total** — 5,199 messages after cleaning

---

## Model Performance

| Metric | Score |
|---|---|
| Overall Accuracy | 97.98% |
| Spam Detection (Recall) | 97%+ |
| Ham Protection | 99%+ |
| Indian SMS Accuracy | 91.7% |

---

## Project Structure

```
spam-detector/
│
├── app_week5.py                  # Final Streamlit web app
├── week1_data_exploration.py     # Data loading and EDA
├── week2_model_training.py       # Model training and evaluation
├── week4_features_wordcloud.py   # Word cloud and highlight features
├── week5_model_improvement.py    # Indian data + model v2
│
├── spam_model_v2.pkl             # Trained ML model (v2)
├── tfidf_vectorizer_v2.pkl       # TF-IDF vectorizer (v2)
│
├── spam.csv                      # Original UCI dataset
├── spam_cleaned.csv              # Cleaned dataset (Week 1 output)
├── spam_combined.csv             # English + Indian combined (Week 5)
├── sample_messages.csv           # Sample CSV for bulk testing
│
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## How to Run Locally

### Step 1 — Clone the repository
```bash
git clone https://github.com/Snehal122004/ai-spam-detector.git
cd ai-spam-detector
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the app
```bash
streamlit run app_week5.py
```

### Step 4 — Open in browser
```
http://localhost:8501
```

---

## Week-by-Week Development

| Week | Task | Output |
|---|---|---|
| Week 1 | Data exploration and cleaning | spam_cleaned.csv |
| Week 2 | Naive Bayes model training | 97% accuracy |
| Week 3 | Streamlit web app | Working UI |
| Week 4 | Word highlight, word cloud, Indian SMS test | 3 new features |
| Week 5 | Indian data added, model v2, bulk CSV | 97.98% accuracy |
| Week 6 | GitHub + Streamlit Cloud deployment | Live app |

---

## How It Works

```
User Input Message
       ↓
Text Cleaning
(lowercase, remove links, remove stopwords)
       ↓
TF-IDF Vectorization
(convert text to numbers)
       ↓
Naive Bayes Model
(calculate spam probability)
       ↓
Result: SPAM or HAM
+ Confidence Score
+ Word Highlight
+ Advice
```

---

## Requirements

```
streamlit
scikit-learn
pandas
numpy
matplotlib
wordcloud
```

---

## About

Built by **Snehal** as part of the
**IBM SkillsBuild AICTE 6-Week Technical Internship Program**
in Artificial Intelligence — May 2026.

*IBM SkillsBuild AICTE AI Internship — 2026*
