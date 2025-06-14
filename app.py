import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Setup
nltk.download('stopwords')
ps = PorterStemmer()

# Preprocess text
def transform_text(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    filtered = [word for word in tokens if word not in stopwords.words('english')]
    stemmed = [ps.stem(word) for word in filtered]
    return " ".join(stemmed)

# Loading models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# App configuration
st.set_page_config(page_title="SMS Spam Classifier 🚀", page_icon="📨", layout="centered")

# Custom CSS
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            text-align: center;
            font-size: 42px;
            font-weight: bold;
            color: #2c3e50;
        }
        .tagline {
            text-align: center;
            font-size: 18px;
            color: #7f8c8d;
            margin-bottom: 30px;
        }
        .result-card {
            font-size: 26px;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin-top: 20px;
            color: white;
        }
        .spam {
            background-color: #e74c3c;
        }
        .not-spam {
            background-color: #2ecc71;
        }
        .confidence {
            text-align: center;
            font-size: 16px;
            color: #34495e;
            margin-top: 10px;
        }
        .history {
            background-color: black;
            padding: 10px;
            border-radius: 10px;
            margin-top: 15px;
        }
        .footer {
            text-align: center;
            font-size: 12px;
            color: #95a5a6;
            margin-top: 50px;
        }
        .stButton>button {
            background-color: #2c3e50;
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">📨 SMS Spam Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">Let AI decide if your SMS is spam or not in real-time!</div>', unsafe_allow_html=True)

# History session init
if "history" not in st.session_state:
    st.session_state.history = []

# Input
input_sms = st.text_area("✍️ Your Message", height=150, placeholder="Type or paste your SMS here...")

# Prediction logic
if input_sms.strip() != "":
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    prediction = model.predict(vector_input)[0]
    prediction_proba = model.predict_proba(vector_input)[0]

    # Show result
    if prediction == 1:
        st.markdown('<div class="result-card spam">🚫 This message is <strong>SPAM</strong></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-card not-spam">✅ This message is <strong>NOT SPAM</strong></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="confidence">📊 Confidence: <b>{round(prediction_proba[prediction]*100, 2)}%</b></div>', unsafe_allow_html=True)

    # Append to history
    st.session_state.history.append((input_sms.strip(), "SPAM" if prediction else "NOT SPAM", round(prediction_proba[prediction]*100, 2)))

elif input_sms.strip() == "":
    st.markdown("<br>", unsafe_allow_html=True)

# Prediction history section
if st.session_state.history:
    st.markdown("### 📜 Prediction History")

    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.success("History cleared.")

    with st.expander("View Last 5 Predictions", expanded=False):
        for msg, result, prob in reversed(st.session_state.history[-5:]):
            st.markdown(f"""
                <div class="history">
                    <b>Message:</b> {msg}<br>
                    <b>Prediction:</b> {result}<br>
                    <b>Confidence:</b> {prob}%
                </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">🚀 Made by Aaryan Lunis</div>', unsafe_allow_html=True)
