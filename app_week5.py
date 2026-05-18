import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from wordcloud import WordCloud
from collections import Counter

# ===========================================================
#  PAGE CONFIG
# ===========================================================
st.set_page_config(
    page_title="Spam Detector AI",
    page_icon="shield",
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
        font-size: 2.2rem;
        font-weight: 700;
        color: #1A237E;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        font-size: 0.95rem;
        color: #546E7A;
        margin-bottom: 1.5rem;
    }
    .spam-box {
        background: #D32F2F;
        color: white;
        padding: 1.4rem 2rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.7rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    .ham-box {
        background: #2E7D32;
        color: white;
        padding: 1.4rem 2rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.7rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    .info-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin: 0.5rem 0;
        border: 1px solid #E0E0E0;
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
    .v2-badge {
        background-color: #1565C0;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .indian-badge {
        background-color: #E65100;
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
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


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
    'will','can','now','please','dear','sir','madam','hello'
}

# ===========================================================
#  SPAM TRIGGER WORDS (English + Indian)
# ===========================================================
SPAM_TRIGGER_WORDS = {
    'free','win','winner','won','prize','claim','cash','reward',
    'urgent','click','offer','limited','exclusive','congratulations',
    'selected','bonus','discount','credit','loan','money','earn',
    'income','profit','ringtone','reply','stop','blocked','suspended',
    'verify','kyc','otp','bank','lucky','draw','lotto','jackpot',
    'million','thousand','pounds','dollars','gift','voucher','cheap',
    'approved','final','notice','warning','alert','guaranteed',
    'account','immediate','immediately','register','activation',
    'jeeto','kamao','paisa','rupees','rupaye','lakh','crore',
    'naukri','job','online','daily','payment','transfer','deposit',
    'withdrawal','pending','release','aadhaar','aadhar','pan',
    'upi','paytm','phonepe','gpay','sbi','hdfc','icici','pnb',
    'lic','irctc','bsnl','jio','airtel','flipkart','amazon',
    'invest','return','insurance','policy','expire','scheme',
    'abhi','turant','jaldi','sirf','aaj','muft','lottery',
}


# ===========================================================
#  LOAD MODEL — v2 (Week 5 improved)
# ===========================================================
@st.cache_resource
def load_model():
    try:
        with open('spam_model_v2.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer_v2.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        return model, tfidf, True
    except FileNotFoundError:
        try:
            with open('spam_model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('tfidf_vectorizer.pkl', 'rb') as f:
                tfidf = pickle.load(f)
            return model, tfidf, False
        except FileNotFoundError:
            return None, None, False

model, tfidf, is_v2 = load_model()


# ===========================================================
#  LOAD DATASET FOR WORD CLOUD
# ===========================================================
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('spam_combined.csv')
        return df
    except Exception:
        try:
            df = pd.read_csv('spam_cleaned.csv')
            return df
        except Exception:
            return None

df_data = load_dataset()


# ===========================================================
#  HELPER FUNCTIONS
# ===========================================================
def clean_text(text):
    text  = str(text).lower()
    text  = re.sub(r'http\S+|www\S+', '', text)
    text  = re.sub(r'[^a-zA-Z\s]', '', text)
    text  = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split()
             if w not in STOP_WORDS and len(w) > 2]
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
        risk = ("HIGH RISK"   if spam_prob >= 90 else
                "MEDIUM RISK" if spam_prob >= 70 else
                "LOW RISK")
    else:
        risk = "SAFE"
    return is_spam, spam_prob, ham_prob, risk


def highlight_words(message):
    words            = message.split()
    result           = []
    found_spam_words = []
    for word in words:
        clean_w = re.sub(r'[^a-zA-Z]', '', word.lower())
        if clean_w in SPAM_TRIGGER_WORDS and len(clean_w) > 2:
            result.append(
                f'<span class="spam-word">{word}</span>'
            )
            if clean_w.upper() not in found_spam_words:
                found_spam_words.append(clean_w.upper())
        else:
            result.append(
                f'<span class="safe-word">{word}</span>'
            )
    return ' '.join(result), found_spam_words


def generate_wordcloud(df, label='spam'):
    if df is None:
        return None
    col = 'cleaned_message' if 'cleaned_message' in df.columns else 'message'
    subset = df[df['label'] == label][col].dropna()
    if len(subset) == 0:
        return None
    all_text = ' '.join(subset.tolist())
    colormap   = 'Reds'   if label == 'spam' else 'Greens'
    background = '#1A0000' if label == 'spam' else '#001A00'
    title      = ('SPAM Word Cloud' if label == 'spam'
                  else 'HAM (Real) Word Cloud')
    title_color = '#FF5252' if label == 'spam' else '#43A047'
    wc = WordCloud(
        width=700, height=350,
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


def bulk_check(df_input):
    results = []
    for _, row in df_input.iterrows():
        msg     = str(row['message'])
        is_spam, spam_prob, ham_prob, risk = predict_spam(msg)
        results.append({
            'Message'    : msg,
            'Prediction' : 'SPAM' if is_spam else 'HAM',
            'Spam %'     : round(spam_prob, 1),
            'Risk Level' : risk
        })
    return pd.DataFrame(results)


# ===========================================================
#  HEADER
# ===========================================================
st.markdown(
    '<div class="main-title">Spam Detector AI</div>',
    unsafe_allow_html=True
)

badge_html = (
    '<span class="v2-badge">Model v2</span>&nbsp;&nbsp;'
    '<span class="indian-badge">Indian SMS Support</span>'
    if is_v2 else
    '<span class="v2-badge">Model v1</span>'
)
st.markdown(
    f'<div class="sub-title">Paste any Email or SMS — '
    f'AI checks instantly &nbsp;|&nbsp; {badge_html}</div>',
    unsafe_allow_html=True
)

st.divider()

# ===========================================================
#  MODEL STATUS CHECK
# ===========================================================
if model is None:
    st.error("""
    **Model files not found!**

    Please run this first in terminal:
    ```
    python week5_spam_detector.py
    ```
    This will create spam_model_v2.pkl and tfidf_vectorizer_v2.pkl
    """)
    st.stop()

if is_v2:
    st.success(
        "Model v2 loaded — Trained on English + Indian SMS "
        "(97.98% accuracy)"
    )
else:
    st.info(
        "Model v1 loaded — Run week5_spam_detector.py "
        "to get the improved v2 model"
    )


# ===========================================================
#  4 TABS
# ===========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Check Message",
    "Word Cloud",
    "Indian SMS Test",
    "Bulk CSV Upload"
])


# ──────────────────────────────────────────────────────────
#  TAB 1 — CHECK MESSAGE
# ──────────────────────────────────────────────────────────
with tab1:

    st.markdown("**Quick Test — Click a sample:**")

    sample_msgs = {
        "Spam (English)" : (
            "Congratulations! You WON a FREE prize worth "
            "1000 pounds. Call now to CLAIM your reward!"
        ),
        "Spam (Indian)"  : (
            "Aapka SBI account band ho jayega. "
            "KYC update karo abhi turant"
        ),
        "Spam (Lottery)" : (
            "Your mobile number won Rs 50000 in TRAI lucky "
            "draw. Call 7700123456 today only"
        ),
        "Real (English)" : (
            "Hey can we meet tomorrow at 5pm "
            "for the project discussion?"
        ),
        "Real (Hinglish)": (
            "Bhai kal college aayega na? "
            "Practical hai subah 9 baje"
        ),
    }

    col1, col2 = st.columns(2)
    for idx, (label, msg) in enumerate(sample_msgs.items()):
        btn_col = col1 if idx % 2 == 0 else col2
        if btn_col.button(label, use_container_width=True):
            st.session_state['input_text'] = msg

    default_text = st.session_state.get('input_text', '')

    st.markdown("**Or paste your own message:**")
    user_input = st.text_area(
        label="Message input",
        value=default_text,
        height=130,
        placeholder=(
            "Paste any email, SMS, or WhatsApp message here..."
        ),
        label_visibility="collapsed"
    )

    c_btn1, c_btn2 = st.columns([1, 5])
    check_btn = c_btn1.button(
        "Check", type="primary", use_container_width=True
    )
    clear_btn = c_btn2.button("Clear")

    if clear_btn:
        st.session_state['input_text'] = ''
        st.rerun()

    if check_btn and user_input.strip():

        is_spam, spam_prob, ham_prob, risk = predict_spam(
            user_input
        )

        st.divider()
        st.markdown("### Result")

        if is_spam:
            st.markdown(
                f'<div class="spam-box">'
                f'SPAM DETECTED — {risk}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="ham-box">'
                'REAL MESSAGE — SAFE</div>',
                unsafe_allow_html=True
            )

        st.markdown("**Confidence Scores:**")
        cs1, cs2 = st.columns(2)
        with cs1:
            st.caption("Spam Probability")
            st.progress(int(spam_prob) / 100)
            st.markdown(f"**{spam_prob:.1f}%**")
        with cs2:
            st.caption("Real (Ham) Probability")
            st.progress(int(ham_prob) / 100)
            st.markdown(f"**{ham_prob:.1f}%**")

        st.divider()

        st.markdown("### Word Highlight")
        st.caption(
            "Red words = spam signals found in your message"
        )
        highlighted, found_words = highlight_words(user_input)
        st.markdown(
            f'<div class="info-card" style="line-height:2.4;">'
            f'{highlighted}</div>',
            unsafe_allow_html=True
        )
        if found_words:
            st.markdown(
                f"**Spam words found ({len(found_words)}):** "
                f"`{'`  `'.join(found_words)}`"
            )
            st.progress(min(len(found_words) / 10.0, 1.0))
        else:
            st.success("No spam trigger words found — looks clean!")

        st.divider()

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
                - Verify sender identity before action
                """)
        else:
            st.success("""
            **This message appears REAL and SAFE.**
            - No major spam signals detected
            - Safe to read and respond
            """)

    elif check_btn and not user_input.strip():
        st.warning("Please enter a message first!")


# ──────────────────────────────────────────────────────────
#  TAB 2 — WORD CLOUD
# ──────────────────────────────────────────────────────────
with tab2:

    st.markdown("### Spam Word Cloud")
    st.caption(
        "Visual of most common words in spam vs real messages"
    )

    if df_data is not None:

        wc_opt = st.radio(
            "Select word cloud:",
            ["Spam Messages", "Real (Ham) Messages",
             "Both Side by Side"],
            horizontal=True
        )

        if wc_opt == "Spam Messages":
            fig = generate_wordcloud(df_data, 'spam')
            if fig:
                st.pyplot(fig)
                st.caption(
                    "Bigger word = appears more in SPAM. "
                    "AI uses these to detect spam."
                )

        elif wc_opt == "Real (Ham) Messages":
            fig = generate_wordcloud(df_data, 'ham')
            if fig:
                st.pyplot(fig)
                st.caption(
                    "Bigger word = appears more in REAL messages."
                )

        else:
            wc1, wc2 = st.columns(2)
            with wc1:
                st.markdown("**SPAM (Red)**")
                fig_s = generate_wordcloud(df_data, 'spam')
                if fig_s:
                    st.pyplot(fig_s)
            with wc2:
                st.markdown("**HAM (Green)**")
                fig_h = generate_wordcloud(df_data, 'ham')
                if fig_h:
                    st.pyplot(fig_h)

        st.divider()
        st.markdown("### Top 10 Spam Words")

        col_name = ('cleaned_message'
                    if 'cleaned_message' in df_data.columns
                    else 'message')
        spam_msgs  = df_data[df_data['label'] == 'spam'][col_name].dropna()
        all_words  = ' '.join(spam_msgs.tolist()).split()
        word_count = Counter(
            [w for w in all_words
             if w not in STOP_WORDS and len(w) > 2]
        ).most_common(10)

        if word_count:
            wds  = [w[0] for w in word_count]
            cnts = [w[1] for w in word_count]
            fig2, ax = plt.subplots(figsize=(8, 4))
            fig2.patch.set_facecolor('#F8F9FA')
            ax.set_facecolor('#F8F9FA')
            bars = ax.barh(
                wds[::-1], cnts[::-1],
                color='#F44336', alpha=0.85,
                edgecolor='white', linewidth=1.2
            )
            for bar, val in zip(bars, cnts[::-1]):
                ax.text(
                    val + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    str(val), va='center',
                    fontsize=10, color='#C62828',
                    fontweight='bold'
                )
            ax.set_xlabel('Frequency in Spam', fontsize=11)
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
            "Dataset not found. "
            "Run week1_spam_detector.py first."
        )


# ──────────────────────────────────────────────────────────
#  TAB 3 — INDIAN SMS TEST
# ──────────────────────────────────────────────────────────
with tab3:

    st.markdown("### Indian SMS Spam Test")
    st.markdown(
        "Tests the AI on real Indian spam patterns — "
        "fake KYC, OTP fraud, lottery scams, Hinglish messages."
    )
    st.info("30 Indian spam + 10 real = 40 total test cases")

    if st.button("Run Indian SMS Test", type="primary"):

        INDIAN_TEST = [
            ("spam", "Dear customer your SBI account will be blocked update KYC immediately"),
            ("spam", "Congratulations aapka number lucky draw mein select hua Rs jeeto call karo abhi"),
            ("spam", "Work from home job roz Rs kamao sirf videos like karke WhatsApp karo join karo"),
            ("spam", "Instant personal loan no documents required no CIBIL check apply now"),
            ("spam", "Your HDFC bank account suspended verify aadhaar mobile number at website immediately"),
            ("spam", "Free health insurance Ayushman Bharat register at website now"),
            ("spam", "LIC policy bonus ready for release claim within days call helpline now"),
            ("spam", "Flipkart winner you have won iPhone delivery charges pay karo"),
            ("spam", "Aapka Bank of Baroda account band ho jayega abhi KYC update karo turant"),
            ("spam", "TRAI lucky draw winner Rs collect prize by calling today only"),
            ("spam", "Part time job earn Rs daily completing simple tasks WhatsApp karo"),
            ("spam", "Paytm KYC expired update immediately account will be blocked"),
            ("spam", "Amazon warehouse job Rs salary apply website aaj hi"),
            ("spam", "EPF withdrawal pending submit Aadhaar at website to receive payment"),
            ("spam", "Mutual fund invest percent annual return guaranteed WhatsApp karo"),
            ("spam", "BSNL customer cashback milega abhi claim karo number"),
            ("spam", "Loan approved ho gaya account mein transfer hoga call karo"),
            ("spam", "IRCTC account blocked login website enter OTP to verify"),
            ("spam", "Electricity bill payment failed pay now website disconnected"),
            ("spam", "Google Pay account problem hai verify karo website OTP batao"),
            ("spam", "SBI YONO account suspend ho gaya verify karo permanent block"),
            ("spam", "Jio Airtel free data mila hai claim karo abhi turant link"),
            ("spam", "Data entry ghar se karo Rs month guaranteed registration fee call"),
            ("spam", "Credit card bill settle karti hai percent mein call karo"),
            ("spam", "Gold loan zero percent interest apply online instant approval"),
            ("spam", "LIC maturity bonus ready claim call helpline days"),
            ("spam", "BHIM UPI account deactivated reactivate now website"),
            ("spam", "Invest Rs get guaranteed returns zero risk scheme call"),
            ("spam", "Free ringtone download text number just per week"),
            ("spam", "Winner selected exclusive reward call claim now urgent"),
            ("ham",  "Bhai kal college aayega na practical hai subah"),
            ("ham",  "Salary credit ho gayi account mein check karo"),
            ("ham",  "Kal exam hai kya padhai kar li tune"),
            ("ham",  "Ghar kab aa raha hai khana ready hai"),
            ("ham",  "Meeting conference room mein hai please on time aao"),
            ("ham",  "Happy birthday yaar God bless you party kab de raha hai"),
            ("ham",  "Assignment submit kar diya kya last date aaj hai"),
            ("ham",  "Doctor ne kaha rest karo medicine time pe lo"),
            ("ham",  "Train late hai platform par wait karo"),
            ("ham",  "Weekend mein movie dekhne chalte hain kya"),
        ]

        with st.spinner("Testing 40 Indian messages..."):
            results = []
            for true_lbl, msg in INDIAN_TEST:
                is_spam, spam_prob, _, risk = predict_spam(msg)
                pred_lbl = 'spam' if is_spam else 'ham'
                correct  = pred_lbl == true_lbl
                results.append({
                    'Message'   : msg[:65] + '...' if len(msg) > 65 else msg,
                    'Expected'  : true_lbl.upper(),
                    'Predicted' : pred_lbl.upper(),
                    'Spam %'    : f"{spam_prob:.1f}%",
                    'Result'    : 'OK' if correct else 'MISS'
                })

            results_df     = pd.DataFrame(results)
            correct_count  = (results_df['Result'] == 'OK').sum()
            total          = len(results_df)
            accuracy       = correct_count / total * 100

        m1, m2, m3 = st.columns(3)
        m1.metric("Total", total)
        m2.metric("Correct", correct_count)
        m3.metric("Accuracy", f"{accuracy:.1f}%")

        if accuracy >= 80:
            st.success(
                f"Great! {correct_count}/{total} detected. "
                f"Big improvement from Week 4 (52.5%)!"
            )
        elif accuracy >= 60:
            st.warning(
                f"{correct_count}/{total} detected. "
                f"Good improvement. More Indian data will help."
            )
        else:
            st.error(
                f"Model needs more Indian training data."
            )

        st.markdown("**Detailed Results:**")
        st.dataframe(results_df, use_container_width=True,
                     hide_index=True)


# ──────────────────────────────────────────────────────────
#  TAB 4 — BULK CSV UPLOAD  (Week 5 New Feature)
# ──────────────────────────────────────────────────────────
with tab4:

    st.markdown("### Bulk CSV Spam Checker")
    st.markdown(
        "Upload a CSV file with hundreds of messages. "
        "AI checks ALL of them at once and gives a "
        "downloadable results file."
    )

    st.info("""
    **How to use:**
    1. Your CSV must have a column named **message**
    2. Upload the file below
    3. Click **Check All Messages**
    4. Download the results CSV
    """)

    # Download sample CSV button
    sample_data = pd.DataFrame({
        'message': [
            "Congratulations! You WON a FREE prize. Call now to CLAIM!",
            "URGENT: Your SBI account will be blocked. Update KYC now.",
            "Hey can we meet tomorrow at 5pm for project discussion?",
            "Aapka loan approved ho gaya hai abhi call karo",
            "Please send me the notes from today class",
            "FREE entry win FA Cup tickets text WIN to 87121",
            "Mom I will be home late tonight please do not wait for dinner",
            "Dear customer your Paytm KYC expired update immediately",
            "Bhai kal college aayega na practical hai subah 9 baje",
            "Your mobile number won Rs 50000 in TRAI lucky draw today",
        ]
    })

    sample_csv = sample_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Sample CSV (to test)",
        data=sample_csv,
        file_name="sample_messages.csv",
        mime="text/csv"
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload your CSV file here:",
        type=['csv']
    )

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)

            if 'message' not in df_upload.columns:
                st.error(
                    "CSV must have a column named 'message'. "
                    "Please fix and re-upload."
                )
            else:
                st.success(
                    f"File uploaded! "
                    f"{len(df_upload)} messages found."
                )
                st.dataframe(
                    df_upload.head(5),
                    use_container_width=True
                )

                if st.button(
                    "Check All Messages", type="primary"
                ):
                    progress_bar = st.progress(0)
                    status_text  = st.empty()

                    results_list = []
                    total_msgs   = len(df_upload)

                    for idx, row in df_upload.iterrows():
                        msg = str(row['message'])
                        is_spam, spam_prob, ham_prob, risk = (
                            predict_spam(msg)
                        )
                        results_list.append({
                            'message'    : msg,
                            'prediction' : 'SPAM' if is_spam else 'HAM',
                            'spam_%'     : round(spam_prob, 1),
                            'ham_%'      : round(ham_prob, 1),
                            'risk_level' : risk
                        })
                        progress = int((idx + 1) / total_msgs * 100)
                        progress_bar.progress(progress)
                        status_text.text(
                            f"Checking {idx + 1} / {total_msgs}..."
                        )

                    status_text.text("Done!")
                    results_df = pd.DataFrame(results_list)

                    spam_found = (
                        results_df['prediction'] == 'SPAM'
                    ).sum()
                    ham_found  = (
                        results_df['prediction'] == 'HAM'
                    ).sum()

                    r1, r2, r3 = st.columns(3)
                    r1.metric("Total Checked", total_msgs)
                    r2.metric("SPAM Found",    spam_found)
                    r3.metric("REAL (Safe)",   ham_found)

                    st.markdown("**Results:**")
                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    result_csv = results_df.to_csv(
                        index=False
                    ).encode('utf-8')

                    st.download_button(
                        label="Download Results CSV",
                        data=result_csv,
                        file_name="spam_check_results.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Error reading file: {e}")


# ===========================================================
#  HOW IT WORKS
# ===========================================================
st.divider()

with st.expander("How does this AI work?"):
    st.markdown("""
    **Step 1 — Text Cleaning**
    Message lowercased, links removed, stopwords removed.

    **Step 2 — TF-IDF Vectorization**
    Each word gets a score. Spam words like FREE, WIN, CLAIM,
    KYC, OTP get high scores.

    **Step 3 — Model Prediction**
    Week 5 uses Naive Bayes trained on English + Indian SMS data.
    Achieves 97.98% overall accuracy, 91.7% on Indian messages.

    **Step 4 — Word Highlight**
    80+ spam trigger words checked including Indian words.

    **Dataset:** UCI SMS Spam + Custom Indian SMS (5199 total)
    """)

with st.expander("About this project"):
    st.markdown("""
    **Spam Detector AI — Week 5 (Final)**

    IBM SkillsBuild AICTE 6-Week AI/ML Internship Project

    **Week-by-week:**
    - Week 1: Data exploration and cleaning
    - Week 2: Naive Bayes model training (97% accuracy)
    - Week 3: Streamlit web app
    - Week 4: Word highlight, word cloud, Indian SMS test
    - Week 5: Indian data added, model v2, bulk CSV upload

    **Tech Stack:**
    Python, Scikit-learn, NLTK, Streamlit,
    WordCloud, Matplotlib, Pandas, NumPy

    **Accuracy:** 97.98% overall, 91.7% Indian messages
    """)

# ===========================================================
#  FOOTER
# ===========================================================
st.markdown(
    '<div class="footer">'
    'IBM SkillsBuild AI Internship — '
    'Spam Detector AI | Week 5 Final App'
    '</div>',
    unsafe_allow_html=True
)