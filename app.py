import streamlit as st
import pickle
import re
import gdown

# Load model and vectorizer from local path
# model = pickle.load(open(r'D:\Ai_projects\sentiment\src\sentiment_model.pkl', 'rb')) # Load the model from the specified path
# vectorizer = pickle.load(open(r'D:\Ai_projects\sentiment\src\vectorizer.pkl', 'rb')) # Load the vectorizer from the specified path

# ————— Download & load your model and vectorizer from google drive—————
@st.cache_resource
def download_model():
    url = "https://drive.google.com/uc?id=1c-OdneN0IAqydCy4kyJUqWet-cg_sGHG"
    gdown.download(url, "sentiment_model.pkl", quiet=True)
    return pickle.load(open("sentiment_model.pkl", "rb"))

@st.cache_resource
def download_vectorizer():
    url = "https://drive.google.com/uc?id=1ZWXxBnaKAb3DtYy2D5GkBkEV-NT0-8L1"
    gdown.download(url, "vectorizer.pkl", quiet=True)
    return pickle.load(open("vectorizer.pkl", "rb"))

model      = download_model()
vectorizer = download_vectorizer()
# vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ————— Simple cleaner that uses only regex & split —————
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # lowercase, remove numbers/punct/URLs/mentions/hashtags
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|#[A-Za-z0-9_]+", "", text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # collapse whitespace and return
    return " ".join(text.split())

# ————— Streamlit UI —————
st.title("Sentiment Analyzer")

user_input = st.text_area("Enter text here")
if st.button("Analyze"):
    cleaned    = clean_text(user_input)
    vect       = vectorizer.transform([cleaned])
    pred_label = model.predict(vect)[0]

    # map numeric → human
    sentiment_map = {0: "Irrelevant", 1: "Negative", 2: "Neutral", 3: "Positive"}
    sentiment     = sentiment_map.get(pred_label, "Unknown")

    color_map = {"Irrelevant":"orange","Negative":"red","Neutral":"gray","Positive":"green"}
    color     = color_map.get(sentiment,"black")

    st.markdown(
        f"<h4>Sentiment: <span style='color:{color}'>{sentiment}</span></h4>",
        unsafe_allow_html=True
    )
