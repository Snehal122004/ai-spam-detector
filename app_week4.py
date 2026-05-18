import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# ===========================================================
#  PAGE CONFIG
# ===========================================================
st.set_page_config(
    page_title="Spam Detector AI",
    page_icon="🛡️",
    layout="centered"
)

# ===========================================================
#  CUSTOM CSS
# ===========================================================
st.markdown("""
<style>
    .stApp { background-color: #F0F4F8; }
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
    .spam-box {
        background: linear-gradient(135deg, #FF5252, #D32F2F);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 1rem 0;
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
    }
    .info-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .spam-word {
        background-color: #FFCDD2;
        color: #B71C1C;
        padding: 3px 8px;
        border-radius: 5px;
        font-weight: 700;
        margin: 2px;
        display: inline-block;
        font-size: 0.95rem;
    }
    .safe-word {
        background-color: #F5F5F5;
        color: #37474F;
        padding: 3px 8px;
        border-radius: 5px;
        margin: 2px;
        display: inline-block;
        font-size: 0.95rem;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #546E7A;
        margin-bottom: 2px;
    }
    .indian-badge {
        background-color: #FF9800;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .footer {
        text-align: center;
        color: #90A4AE;
        font-size: 0.8rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)


# ===========================================================
#  INDIAN SMS SPAM DATASET
#  (Week 4 - नवीन feature)
#  Real Indian spam patterns — Hinglish + English
# ===========================================================
INDIAN_SPAM_MESSAGES = [
    # Fake bank / KYC alerts
    "Dear customer your SBI account will be blocked within 24 hours. Update your KYC immediately at www.sbi-kyc-update.com or call 9876543210",
    "Your HDFC bank account has been suspended. Please verify your Aadhaar linked mobile number at www.hdfc-verify.net immediately",
    "URGENT: Your PNB account shows suspicious activity. Login at www.pnb-secure.net to verify or your account will be frozen",
    "Aapka Bank of Baroda account band ho jayega. Abhi KYC update karo www.bob-kyc.net par. Turant karo!",
    "Dear UPI user your BHIM account has been deactivated. Reactivate now at www.bhim-support.in",

    # Fake lottery / prize
    "Congratulations! Aapka number lucky draw mein select hua hai. Rs 25000 jeeto. Call karo 9988776655 par abhi",
    "Jio aur Airtel ki taraf se aapko 50GB free data mila hai. Claim karo is link se: www.jio-free.net",
    "Aap BSNL ke 10000th customer hain! Aapko Rs 10000 ka cashback milega. Abhi claim karo 8800123456 par",
    "Your mobile number has won Rs 50000 in TRAI lucky draw. Collect prize by calling 7700123456 today only",
    "Flipkart Big Sale Winner! You have won iPhone 15. Delivery charges Rs 99 pay karo www.flipkart-prize.com",

    # Fake job offers
    "Work from home job. Roz Rs 500 kamao sirf videos like karke. WhatsApp karo 9876012345 abhi join karo",
    "Part time job offer. Earn Rs 800 daily by completing simple tasks online. No experience needed. Call 9988001122",
    "Data entry job ghar se karo. Rs 15000 per month guaranteed. Registration fee Rs 500 only. Call 8877665544",
    "Amazon warehouse mein job hai. Rs 20000 salary. Apply karo www.amazon-jobs-india.net par aaj hi",
    "Urgent requirement. 50 candidates needed for online work. Daily payment Rs 600. WhatsApp 9900112233",

    # Fake loan offers
    "Instant personal loan Rs 50000 to Rs 500000. No documents required. No CIBIL check. Apply now 9876543210",
    "Aapka loan approved ho gaya hai. Rs 100000 aapke account mein abhi transfer hoga. Call 8800991122",
    "Gold loan at 0% interest for 3 months. Apply online at www.instant-loan-india.com. Instant approval",
    "Business loan without collateral. Rs 5 lakh to Rs 50 lakh. Same day disbursement. Contact 7788990011",
    "Credit card ka bill zyada hai? Hamari company settle karti hai 30% mein. Call karo 9900887766",

    # Fake OTP / account alerts
    "Your OTP for SBI net banking is 452819. NEVER share this OTP with anyone. Valid for 10 minutes only",
    "Paytm account verification. Your OTP is 983421. Share this OTP with our executive to complete KYC",
    "Google Pay account mein problem hai. Verify karo www.gpay-verify.net par ya OTP batao 9876543210 ko",
    "IRCTC account blocked. Login at www.irctc-reactivate.com and enter your ticket booking OTP to verify",
    "Electricity bill payment failed. Pay now at www.bescom-payment.net or supply will be disconnected today",

    # Fake insurance / investment
    "LIC policy bonus Rs 85000 ready for release. Claim within 7 days. Call LIC helpline 9988776655 now",
    "Invest Rs 5000 per month and get Rs 50 lakh guaranteed after 10 years. Zero risk plan. Call 8877001122",
    "Mutual fund mein invest karo. 40% annual return guaranteed. Minimum investment Rs 1000. WhatsApp 9900112233",
    "Free health insurance for 1 year from Ayushman Bharat. Register at www.ayushman-free.gov-in.net now",
    "Your EPF withdrawal of Rs 45000 is pending. Submit Aadhaar at www.epfo-claim.net to receive payment",

    # Ham messages (real Indian messages)
    "Bhai kal college aayega na? Practical hai subah 9 baje",
    "Aaj dinner mein kya banana hai? Ghar par vegetable leke aa",
    "Meeting 3 baje hai conference room mein. Please on time aao",
    "Happy birthday yaar! God bless you. Party kab de raha hai?",
    "Train 20 minute late hai. Platform 3 par wait karo",
    "Mummy ne bulaya hai. Ghar jaldi aa. Important baat karni hai",
    "Assignment submit kar diya kya? Last date aaj hai",
    "Doctor ne kaha 2 din rest karo. Medicine time pe lo",
    "Salary credit ho gayi account mein. Check karo",
    "Bijli ka bill bhar diya kya? Due date 15 hai",
]

INDIAN_SPAM_LABELS = (
    ['spam'] * 30 +
    ['ham']  * 10
)


# ===========================================================
#  LOAD MODEL
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
#  LOAD DATASET FOR WORD CLOUD
# ===========================================================
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('spam_cleaned.csv')
        if 'cleaned_message' not in df.columns:
            df['cleaned_message'] = df['message']
        return df
    except Exception:
        return None

df_data = load_dataset()


# ===========================================================
#  STOP WORDS
# ===========================================================
STOP_WORDS = {
    'i','me','my','we','our','you','your','he','him','his',
    'she','her','it','its','they','them','their','what','which',
    'this','that','these','those','am','is','are','was','were',
    'be','been','have','has','had','do','does','did','a','an',
    'the','and','but','if','or','as','of','at','by','for',
    'with','to','from','in','out','on','so','than','too','very',
    'u','ur','r','ok','hi','hey','just','get','got','go','im',
    'will','can','now','call','your','please','dear','sir','madam'
}

# ===========================================================
#  WEEK 4 FEATURE 1 — ENHANCED SPAM TRIGGER WORDS
#  (Added Indian spam words)
# ===========================================================
SPAM_TRIGGER_WORDS = {
    # English spam words
    'free','win','winner','won','prize','claim','cash','reward',
    'urgent','click','offer','limited','exclusive','congratulations',
    'selected','bonus','discount','credit','loan','money','earn',
    'income','profit','ringtone','reply','stop','blocked','suspended',
    'verify','kyc','otp','bank','lucky','draw','lotto','jackpot',
    'million','thousand','pounds','dollars','gift','voucher','cheap',
    'approved','final','notice','warning','alert','guaranteed',
    'account','immediate','immediately','register','activation',
    # Indian spam words (Hinglish)
    'jeeto','kamao','paisa','rupees','rupaye','lakh','crore',
    'lottery','jackpot','naukri','job','ghar','online','daily',
    'payment','transfer','deposit','withdrawal','pending','release',
    'aadhaar','aadhar','pan','kyc','upi','paytm','phonepe','gpay',
    'sbi','hdfc','icici','pnb','axis','lic','irctc','bsnl','jio',
    'airtel','flipkart','amazon','meesho','whatsapp','instagram',
    'invest','return','profit','guaranteed','zero','risk','scheme',
    'insurance','policy','bonus','maturity','claim','nominee',
    'blocked','suspend','deactivate','frozen','limited','expire',
    'abhi','turant','jaldi','sirf','aaj','kal','free','muft',
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
    proba      = model.predict_proba(vectorized)[0]
    spam_prob  = proba[1] * 100
    ham_prob   = proba[0] * 100
    is_spam    = prediction == 1
    if is_spam:
        risk = "HIGH RISK"   if spam_prob >= 90 else \
               "MEDIUM RISK" if spam_prob >= 70 else "LOW RISK"
    else:
        risk = "SAFE"
    return is_spam, spam_prob, ham_prob, risk


# ===========================================================
#  WEEK 4 FEATURE 1 — ENHANCED WORD HIGHLIGHT
# ===========================================================
def highlight_words(message):
    """
    Highlight spam trigger words in RED.
    Non-spam words shown in grey.
    Returns HTML string + list of found spam words.
    """
    words            = message.split()
    result           = []
    found_spam_words = []

    for word in words:
        clean_w = re.sub(r'[^a-zA-Z]', '', word.lower())
        if clean_w in SPAM_TRIGGER_WORDS and len(clean_w) > 2:
            result.append(
                f'<span class="spam-word" title="Spam signal word">'
                f'{word}</span>'
            )
            if clean_w.upper() not in found_spam_words:
                found_spam_words.append(clean_w.upper())
        else:
            result.append(
                f'<span class="safe-word">{word}</span>'
            )

    return ' '.join(result), found_spam_words


# ===========================================================
#  WEEK 4 FEATURE 2 — SPAM WORD CLOUD GENERATOR
# ===========================================================
def generate_wordcloud(df, label='spam'):
    """
    Generate a word cloud from spam or ham messages.
    Returns matplotlib figure.
    """
    if df is None:
        return None

    subset = df[df['label'] == label]['cleaned_message'].dropna()

    if len(subset) == 0:
        return None

    all_text = ' '.join(subset.tolist())

    if label == 'spam':
        colormap   = 'Reds'
        background = '#1A0000'
        title      = 'SPAM Word Cloud — Most Common Spam Words'
        title_color = '#FF5252'
    else:
        colormap   = 'Greens'
        background = '#001A00'
        title      = 'HAM Word Cloud — Most Common Real Message Words'
        title_color = '#43A047'

    wc = WordCloud(
        width=700,
        height=350,
        background_color=background,
        colormap=colormap,
        max_words=100,
        min_font_size=10,
        max_font_size=80,
        random_state=42,
        collocations=False,
        stopwords=STOP_WORDS
    ).generate(all_text)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(background)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold',
                 color=title_color, pad=10)
    plt.tight_layout(pad=0)
    return fig


# ===========================================================
#  WEEK 4 FEATURE 3 — INDIAN SMS SPAM TESTER
# ===========================================================
def test_indian_spam():
    """
    Test the model on Indian SMS spam messages.
    Returns results dataframe.
    """
    results = []
    for msg, true_label in zip(INDIAN_SPAM_MESSAGES, INDIAN_SPAM_LABELS):
        is_spam, spam_prob, ham_prob, risk = predict_spam(msg)
        predicted = 'spam' if is_spam else 'ham'
        correct   = '✅' if predicted == true_label else '❌'
        results.append({
            'Message'   : msg[:60] + '...' if len(msg) > 60 else msg,
            'Expected'  : true_label.upper(),
            'Predicted' : predicted.upper(),
            'Spam %'    : f"{spam_prob:.1f}%",
            'Result'    : correct
        })
    return pd.DataFrame(results)


# ===========================================================
#  HEADER
# ===========================================================
st.markdown(
    '<div class="main-title">Spam Detector AI</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-title">'
    'Paste any Email or SMS — AI checks if it is Spam or Real'
    ' &nbsp;|&nbsp; '
    '<span class="indian-badge">Indian SMS Support Added</span>'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ===========================================================
#  MODEL STATUS
# ===========================================================
if not model_loaded:
    st.error("""
    **Model files not found!**
    Run `week2_spam_detector.py` first to generate:
    - spam_model.pkl
    - tfidf_vectorizer.pkl
    """)
    st.stop()


# ===========================================================
#  TABS — Week 4 mein tabs add kele
# ===========================================================
tab1, tab2, tab3 = st.tabs([
    "Check Message",
    "Word Cloud",
    "Indian SMS Test"
])


# ───────────────────────────────────────────────────────────
#  TAB 1 — CHECK MESSAGE
# ───────────────────────────────────────────────────────────
with tab1:

    st.markdown("**Quick Test — Click a sample:**")

    sample_msgs = {
        "Spam 1 (English)" : "Congratulations! You WON a FREE prize worth 1000 pounds. Call now to CLAIM!",
        "Spam 2 (Indian)"  : "Aapka SBI account band ho jayega. KYC update karo www.sbi-kyc-update.com par abhi",
        "Spam 3 (Lottery)" : "Your mobile number won Rs 50000 in TRAI lucky draw. Call 7700123456 today only",
        "Real Message 1"   : "Hey, can we meet tomorrow at 5pm for the project discussion?",
        "Real Message 2"   : "Bhai kal college aayega na? Practical hai subah 9 baje",
    }

    cols = st.columns(3)
    for idx, (label, msg) in enumerate(sample_msgs.items()):
        if cols[idx % 3].button(label, use_container_width=True):
            st.session_state['input_text'] = msg

    default_text = st.session_state.get('input_text', '')

    st.markdown("**Or paste your own message:**")
    user_input = st.text_area(
        label="Message",
        value=default_text,
        height=140,
        placeholder="Paste any email, SMS, or WhatsApp message here...",
        label_visibility="collapsed"
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    check_btn = col_btn1.button("Check", type="primary",
                                use_container_width=True)
    clear_btn = col_btn2.button("Clear")

    if clear_btn:
        st.session_state['input_text'] = ''
        st.rerun()

    if check_btn and user_input.strip():

        is_spam, spam_prob, ham_prob, risk = predict_spam(user_input)

        st.divider()
        st.markdown("### Result")

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
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                '<div class="metric-label">Spam Probability</div>',
                unsafe_allow_html=True
            )
            st.progress(int(spam_prob) / 100)
            st.markdown(f"**{spam_prob:.1f}%**")
        with c2:
            st.markdown(
                '<div class="metric-label">Real (Ham) Probability</div>',
                unsafe_allow_html=True
            )
            st.progress(int(ham_prob) / 100)
            st.markdown(f"**{ham_prob:.1f}%**")

        st.divider()

        # ── WEEK 4 FEATURE 1 — ENHANCED WORD HIGHLIGHT ──
        st.markdown("### Word Highlight Analysis")
        st.markdown(
            "**Red words** = Spam signals detected in your message:"
        )

        highlighted, found_words = highlight_words(user_input)

        st.markdown(
            f'<div class="info-card" style="line-height:2.4;">'
            f'{highlighted}</div>',
            unsafe_allow_html=True
        )

        if found_words:
            st.markdown(
                f"**Spam trigger words found ({len(found_words)}):** "
                f"`{'`  `'.join(found_words)}`"
            )

            # Show count bar
            st.markdown(
                f"Danger level: {len(found_words)} spam "
                f"signal word{'s' if len(found_words) > 1 else ''} found"
            )
            danger = min(len(found_words) / 10.0, 1.0)
            st.progress(danger)

        else:
            st.success(
                "No spam trigger words found — message looks clean!"
            )

        st.divider()

        # Advice
        st.markdown("### What should you do?")
        if is_spam:
            if spam_prob >= 90:
                st.error("""
                **HIGH RISK — Do NOT interact!**
                - Do not click any links
                - Do not call any number mentioned
                - Do not share OTP, Aadhaar, or bank details
                - Delete this message immediately
                - Report spam SMS to TRAI: 1909
                """)
            elif spam_prob >= 70:
                st.warning("""
                **MEDIUM RISK — Be careful!**
                - Verify sender before responding
                - Do not click suspicious links
                - When in doubt, ignore and delete
                """)
            else:
                st.warning("""
                **LOW RISK — Possibly spam.**
                - Some spam signals detected
                - Verify sender identity before taking action
                """)
        else:
            st.success("""
            **This message appears REAL and SAFE.**
            - No major spam signals detected
            - Safe to read and respond
            """)

    elif check_btn and not user_input.strip():
        st.warning("Please enter a message first!")


# ───────────────────────────────────────────────────────────
#  TAB 2 — WORD CLOUD  (Week 4 Feature 2)
# ───────────────────────────────────────────────────────────
with tab2:

    st.markdown("### Spam Word Cloud")
    st.markdown(
        "Visual representation of most common words "
        "in spam and real messages from the training dataset."
    )

    if df_data is not None:

        wc_option = st.radio(
            "Select which word cloud to view:",
            ["Spam Messages", "Real (Ham) Messages", "Both Side by Side"],
            horizontal=True
        )

        if wc_option == "Spam Messages":
            fig = generate_wordcloud(df_data, 'spam')
            if fig:
                st.pyplot(fig)
                st.caption(
                    "Bigger word = appears more often in SPAM messages. "
                    "These are the words AI uses to detect spam."
                )

        elif wc_option == "Real (Ham) Messages":
            fig = generate_wordcloud(df_data, 'ham')
            if fig:
                st.pyplot(fig)
                st.caption(
                    "Bigger word = appears more often in REAL messages. "
                    "These words indicate a message is safe."
                )

        else:
            col_wc1, col_wc2 = st.columns(2)
            with col_wc1:
                st.markdown("**SPAM words (Red)**")
                fig_spam = generate_wordcloud(df_data, 'spam')
                if fig_spam:
                    st.pyplot(fig_spam)

            with col_wc2:
                st.markdown("**HAM words (Green)**")
                fig_ham = generate_wordcloud(df_data, 'ham')
                if fig_ham:
                    st.pyplot(fig_ham)

        st.divider()

        # Top 10 spam words bar chart
        st.markdown("### Top 10 Spam Trigger Words")
        spam_msgs  = df_data[df_data['label']=='spam']['cleaned_message'].dropna()
        all_words  = ' '.join(spam_msgs.tolist()).split()
        word_count = Counter(
            [w for w in all_words
             if w not in STOP_WORDS and len(w) > 2]
        ).most_common(10)

        if word_count:
            words_list = [w[0] for w in word_count]
            count_list = [w[1] for w in word_count]

            fig2, ax = plt.subplots(figsize=(8, 4))
            fig2.patch.set_facecolor('#F8F9FA')
            ax.set_facecolor('#F8F9FA')
            bars = ax.barh(
                words_list[::-1], count_list[::-1],
                color='#F44336', alpha=0.85,
                edgecolor='white', linewidth=1.2
            )
            for bar, val in zip(bars, count_list[::-1]):
                ax.text(
                    val + 0.1,
                    bar.get_y() + bar.get_height() / 2,
                    str(val),
                    va='center', fontsize=10,
                    color='#C62828', fontweight='bold'
                )
            ax.set_xlabel('Frequency in Spam Messages', fontsize=11)
            ax.set_title(
                'Most Frequent Words in SPAM Messages',
                fontsize=13, fontweight='bold'
            )
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            plt.tight_layout()
            st.pyplot(fig2)

    else:
        st.warning(
            "spam_cleaned.csv not found. "
            "Run week1_spam_detector.py first."
        )


# ───────────────────────────────────────────────────────────
#  TAB 3 — INDIAN SMS SPAM TEST  (Week 4 Feature 3)
# ───────────────────────────────────────────────────────────
with tab3:

    st.markdown("### Indian SMS Spam Dataset Test")
    st.markdown("""
    This tab tests the AI model on **real Indian spam patterns** —
    fake KYC alerts, OTP frauds, lottery scams, and Hinglish messages
    that are common in India but not in the original UCI dataset.
    """)

    st.info(
        "30 Indian spam messages + 10 real Indian messages "
        "= 40 total test cases"
    )

    if st.button("Run Indian SMS Test", type="primary"):

        with st.spinner("Testing model on Indian messages..."):
            results_df = test_indian_spam()

        correct_count = results_df['Result'].value_counts().get('✅', 0)
        total_count   = len(results_df)
        accuracy      = correct_count / total_count * 100

        # Summary metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Messages", total_count)
        c2.metric("Correctly Detected", correct_count)
        c3.metric("Accuracy", f"{accuracy:.1f}%")

        if accuracy >= 80:
            st.success(
                f"Great! Model correctly detected "
                f"{correct_count}/{total_count} Indian messages!"
            )
        elif accuracy >= 60:
            st.warning(
                f"Model detected {correct_count}/{total_count}. "
                f"Model needs more Indian training data for better accuracy."
            )
        else:
            st.error(
                f"Model struggled with Indian messages ({accuracy:.1f}%). "
                f"It was trained on English SMS — needs Indian data."
            )

        st.divider()
        st.markdown("**Detailed Results:**")
        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()
        st.markdown("### Why does accuracy differ for Indian messages?")
        st.markdown("""
        **Original model** was trained on **English SMS** (UCI dataset).

        Indian spam often uses:
        - Hinglish words: *abhi, turant, jeeto, kamao*
        - Brand names: *SBI, HDFC, Paytm, Jio, Aadhaar*
        - Mixed script references

        **How to improve:** Add Indian spam messages to the training
        dataset and retrain the model. This is Week 5 improvement!
        """)

        # Show some examples
        st.markdown("**Sample Indian Spam Messages tested:**")
        for msg in INDIAN_SPAM_MESSAGES[:5]:
            is_spam, spam_prob, _, _ = predict_spam(msg)
            icon = "🔴" if is_spam else "🟢"
            st.markdown(
                f"{icon} `{msg[:80]}...`  "
                f"→ **{spam_prob:.0f}% spam**"
            )


# ===========================================================
#  HOW IT WORKS
# ===========================================================
st.divider()

with st.expander("How does this AI work?"):
    st.markdown("""
    **Step 1 — Text Cleaning**
    Message converted to lowercase, links removed,
    stopwords removed.

    **Step 2 — TF-IDF Vectorization**
    Each word gets a score. Spam words like FREE, WIN, CLAIM
    get high scores. Common words get low scores.

    **Step 3 — Naive Bayes Model**
    Same algorithm used in Gmail spam filter.
    Calculates probability that the message is spam.

    **Step 4 — Word Highlight (Week 4)**
    Checks each word against 80+ spam trigger words
    including Indian spam words like KYC, OTP, Aadhaar, Paytm.

    **Step 5 — Word Cloud (Week 4)**
    Visual representation of most common words in
    spam vs real messages from training data.

    **Accuracy: ~97% on English SMS**
    **Indian SMS accuracy improves in Week 5**
    """)

with st.expander("About this project"):
    st.markdown("""
    **Spam Detector AI — Week 4**
    IBM SkillsBuild AICTE 6-Week AI Internship Project.

    **Week 4 New Features:**
    - Enhanced word highlight with Indian spam words
    - Spam word cloud visualization
    - Indian SMS spam dataset testing (40 messages)
    - 3-tab interface

    **Tech Stack:**
    Python, Scikit-learn, NLTK, Streamlit,
    WordCloud, Matplotlib, Pandas, NumPy

    **Dataset:**
    UCI SMS Spam Collection + Custom Indian SMS Dataset
    """)

st.markdown(
    '<div class="footer">'
    'IBM SkillsBuild AI Internship — Spam Detector Week 4'
    '</div>',
    unsafe_allow_html=True
)