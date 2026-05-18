import streamlit as st
import pickle
import re
import numpy as np

# ===========================================================
#  PAGE CONFIG  (must be first streamlit command)
# ===========================================================
st.set_page_config(
    page_title="Spam Detector AI",
    page_icon="shield",
    layout="centered"
)

# ===========================================================
#  CUSTOM CSS  — makes the app look professional
# ===========================================================
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #F0F4F8; }

    /* Title area */
    .main-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 700;
        color: #1A237E;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        font-size: 1rem;
        color: #546E7A;
        margin-bottom: 2rem;
    }

    /* Result boxes */
    .spam-box {
        background: linear-gradient(135deg, #FF5252, #D32F2F);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(211,47,47,0.4);
    }
    .ham-box {
        background: linear-gradient(135deg, #43A047, #1B5E20);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(27,94,32,0.4);
    }

    /* Info cards */
    .info-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Highlight words */
    .spam-word {
        background-color: #FFCDD2;
        color: #B71C1C;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        margin: 2px;
        display: inline-block;
    }
    .safe-word {
        background-color: #C8E6C9;
        color: #1B5E20;
        padding: 2px 6px;
        border-radius: 4px;
        margin: 2px;
        display: inline-block;
    }

    /* Progress bar label */
    .metric-label {
        font-size: 0.85rem;
        color: #546E7A;
        margin-bottom: 2px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #90A4AE;
        font-size: 0.8rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)


# ===========================================================
#  LOAD MODEL  (cached so it loads only once)
# ===========================================================
@st.cache_resource
def load_model():
    with open('spam_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    return model, tfidf

try:
    model, tfidf = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ===========================================================
#  TEXT CLEANING  (same as Week 1 and Week 2)
# ===========================================================
STOP_WORDS = {
    'i','me','my','we','our','you','your','he','him','his',
    'she','her','it','its','they','them','their','what','which',
    'this','that','these','those','am','is','are','was','were',
    'be','been','have','has','had','do','does','did','a','an',
    'the','and','but','if','or','as','of','at','by','for',
    'with','to','from','in','out','on','so','than','too','very',
    'u','ur','r','ok','hi','hey','just','get','got','go','im'
}

# Words that are strong spam signals — used for highlighting
SPAM_TRIGGER_WORDS = {
    'free','win','winner','won','prize','claim','cash','reward',
    'urgent','call','now','click','offer','limited','exclusive',
    'guarantee','guaranteed','congratulations','selected','bonus',
    'discount','credit','loan','money','earn','income','profit',
    'ringtone','text','mobile','sms','message','reply','stop',
    'account','verify','kyc','blocked','suspended','otp','bank',
    'lucky','draw','lotto','jackpot','million','thousand','pounds',
    'dollars','rupees','gift','voucher','coupon','deals','buy',
    'cheap','apply','approved','final','notice','warning','alert'
}

def clean_text(text):
    text  = str(text).lower()
    text  = re.sub(r'http\S+|www\S+', '', text)
    text  = re.sub(r'[^a-zA-Z\s]', '', text)
    text  = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(words)

def predict_spam(message, model, tfidf):
    cleaned    = clean_text(message)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    proba      = model.predict_proba(vectorized)[0]
    spam_prob  = proba[1] * 100
    ham_prob   = proba[0] * 100
    is_spam    = prediction == 1
    if is_spam:
        risk = "HIGH RISK"   if spam_prob >= 90 else \
               "MEDIUM RISK" if spam_prob >= 70 else "LOW RISK"
    else:
        risk = "SAFE"
    return is_spam, spam_prob, ham_prob, risk, cleaned

def highlight_words(message):
    """Highlight spam trigger words in red, others normal."""
    words  = message.split()
    result = []
    found_spam_words = []
    for word in words:
        clean_w = re.sub(r'[^a-zA-Z]', '', word.lower())
        if clean_w in SPAM_TRIGGER_WORDS:
            result.append(f'<span class="spam-word">{word}</span>')
            found_spam_words.append(clean_w.upper())
        else:
            result.append(f'<span class="safe-word">{word}</span>')
    return ' '.join(result), found_spam_words


# ===========================================================
#  HEADER
# ===========================================================
st.markdown('<div class="main-title">Spam Detector AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Paste any Email or SMS — AI will check if it is Spam or Real</div>', unsafe_allow_html=True)

st.divider()

# ===========================================================
#  MODEL STATUS CHECK
# ===========================================================
if not model_loaded:
    st.error("""
    **Model files not found!**

    Make sure these two files are in the same folder as app.py:
    - spam_model.pkl
    - tfidf_vectorizer.pkl

    Run week2_spam_detector.py first to generate these files.
    """)
    st.stop()


# ===========================================================
#  SAMPLE MESSAGES  (quick test buttons)
# ===========================================================
st.markdown("**Quick Test — Click a sample message:**")

col1, col2, col3, col4 = st.columns(4)

sample_spam1 = "Congratulations! You WON a FREE prize worth 1000 pounds. Call now to CLAIM your reward!"
sample_spam2 = "URGENT: Your SBI account will be blocked. Update KYC now at www.sbi-verify.net"
sample_ham1  = "Hey, can we meet tomorrow at 5pm for the project discussion?"
sample_ham2  = "Please send me the notes from today's class. I missed the lecture."

if col1.button("Spam Example 1"):
    st.session_state['input_text'] = sample_spam1
if col2.button("Spam Example 2"):
    st.session_state['input_text'] = sample_spam2
if col3.button("Real Message 1"):
    st.session_state['input_text'] = sample_ham1
if col4.button("Real Message 2"):
    st.session_state['input_text'] = sample_ham2

# Pre-fill text area if sample was clicked
default_text = st.session_state.get('input_text', '')


# ===========================================================
#  TEXT INPUT
# ===========================================================
st.markdown("**Or type / paste your own message below:**")

user_input = st.text_area(
    label="Message",
    value=default_text,
    height=140,
    placeholder="Paste your email or SMS message here and click Check...",
    label_visibility="collapsed"
)

col_btn1, col_btn2 = st.columns([1, 5])
check_btn = col_btn1.button("Check", type="primary", use_container_width=True)
clear_btn = col_btn2.button("Clear", use_container_width=False)

if clear_btn:
    st.session_state['input_text'] = ''
    st.rerun()


# ===========================================================
#  PREDICTION + RESULTS
# ===========================================================
if check_btn and user_input.strip():

    is_spam, spam_prob, ham_prob, risk, cleaned = predict_spam(
        user_input, model, tfidf
    )

    st.divider()
    st.markdown("### Result")

    # Main result box
    if is_spam:
        st.markdown(
            f'<div class="spam-box">SPAM DETECTED — {risk}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="ham-box">REAL MESSAGE — SAFE</div>',
            unsafe_allow_html=True
        )

    # Confidence scores
    st.markdown("**Confidence Scores:**")
    col_s, col_h = st.columns(2)

    with col_s:
        st.markdown('<div class="metric-label">Spam Probability</div>',
                    unsafe_allow_html=True)
        st.progress(int(spam_prob) / 100)
        st.markdown(f"**{spam_prob:.1f}%**")

    with col_h:
        st.markdown('<div class="metric-label">Real (Ham) Probability</div>',
                    unsafe_allow_html=True)
        st.progress(int(ham_prob) / 100)
        st.markdown(f"**{ham_prob:.1f}%**")

    st.divider()

    # Word highlight section
    st.markdown("### Word Analysis")
    st.markdown("Red words = spam signals detected in your message:")

    highlighted, found_words = highlight_words(user_input)
    st.markdown(
        f'<div class="info-card" style="line-height:2.2;">{highlighted}</div>',
        unsafe_allow_html=True
    )

    if found_words:
        st.markdown(
            f"**Spam trigger words found:** `{'`  `'.join(set(found_words))}`"
        )
    else:
        st.success("No spam trigger words found in this message.")

    st.divider()

    # Advice section
    st.markdown("### What should you do?")

    if is_spam:
        if spam_prob >= 90:
            st.error("""
            **HIGH RISK — Do NOT interact with this message!**
            - Do not click any links
            - Do not call any numbers mentioned
            - Do not share any personal details
            - Delete this message immediately
            - Report it to your email provider or TRAI (1909)
            """)
        elif spam_prob >= 70:
            st.warning("""
            **MEDIUM RISK — Be careful!**
            - This message shows several spam patterns
            - Verify the sender independently before responding
            - Do not click suspicious links
            - When in doubt, ignore and delete
            """)
        else:
            st.warning("""
            **LOW RISK — Possibly spam.**
            - Some spam signals detected but not conclusive
            - Verify sender identity before taking any action
            - If unsure, do not respond
            """)
    else:
        st.success("""
        **This message appears to be REAL and SAFE.**
        - No major spam signals detected
        - Safe to read and respond
        - Always stay cautious with unknown senders
        """)

elif check_btn and not user_input.strip():
    st.warning("Please enter or paste a message first.")


# ===========================================================
#  HOW IT WORKS  (expandable section)
# ===========================================================
st.divider()

with st.expander("How does this AI work?"):
    st.markdown("""
    **Step 1 — Text Cleaning**
    The message is converted to lowercase, links are removed,
    and common words like "the", "is", "and" are removed.

    **Step 2 — TF-IDF Vectorization**
    Each word gets a score based on how often it appears
    in spam vs real messages. Words like FREE, WIN, PRIZE
    get high spam scores.

    **Step 3 — Naive Bayes Model**
    The same algorithm used in Gmail's spam filter.
    It calculates the probability that this message is spam
    based on the word scores.

    **Step 4 — Result**
    If spam probability > 50%, the message is flagged as SPAM.
    The confidence score tells you how sure the AI is.

    **Training Data**
    This model was trained on the UCI SMS Spam Collection
    dataset — 5,572 real and spam messages.

    **Accuracy: ~97%**
    """)

with st.expander("About this project"):
    st.markdown("""
    **Spam Detector AI**
    Built as part of the IBM SkillsBuild AICTE 6-Week AI Internship Program.

    **Tech Stack:**
    - Python 3
    - Scikit-learn (Naive Bayes, TF-IDF)
    - NLTK (text preprocessing)
    - Streamlit (web UI)
    - Pandas, NumPy (data handling)

    **Dataset:**
    UCI SMS Spam Collection Dataset — 5,572 labeled messages

    **Week-by-week build:**
    - Week 1 : Data exploration and cleaning
    - Week 2 : Model training and evaluation
    - Week 3 : Streamlit web app (this!)
    """)


# ===========================================================
#  FOOTER
# ===========================================================
st.markdown(
    '<div class="footer">IBM SkillsBuild AI Internship — Spam Detector Project</div>',
    unsafe_allow_html=True
)
