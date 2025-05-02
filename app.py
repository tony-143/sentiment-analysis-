import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

# nltk.download('punkt')
# nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Init tools
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^A-Za-z\s]', '', text)  # remove punctuation & numbers
    text = re.sub(r"http\S+|@\S+|#[A-Za-z0-9_]+", "", text)  # remove URLs, mentions, hashtags
    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

st.title("Sentiment Analyzer")
user_input = st.text_area("Enter text")

if st.button("Analyze"):
    cleaned = clean_text(user_input)
    vect_text = vectorizer.transform([cleaned])
    prediction = model.predict(vect_text)[0]
    sentiment_map = {
        0: "Irrelevant",
        1: "Negative",
        2: "Neutral",
        3: "Positive"
    }
    sentiment = sentiment_map.get(prediction, "Unknown")
    st.write(f"Sentiment: {sentiment}")