import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import gdown

# Ensure NLTK data is downloaded properly
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

nltk.download('omw-1.4') 
nltk.download('wordnet')

# If you want to load model from a URL, uncomment the following lines
@st.cache_resource
def download_model():
    url = "https://drive.google.com/uc?id=1c-OdneN0IAqydCy4kyJUqWet-cg_sGHG"
    gdown.download(url, "sentiment_model.pkl", quiet=False)
    return pickle.load(open("sentiment_model.pkl", "rb"))

@st.cache_resource
def download_vectorizer():
    url = "https://drive.google.com/uc?id=1PqUSx5VHP16BJbSHYQw0KJEWjOJayLu_"
    gdown.download(url, "vectorizer.pkl", quiet=False)
    return pickle.load(open("vectorizer.pkl", "rb"))

model = download_model()
vectorizer = download_vectorizer()

# Init tools
stemmer = SnowballStemmer("english")
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

# Streamlit App
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

    color_map = {
        "Irrelevant": "orange",
        "Negative": "red",
        "Neutral": "gray",
        "Positive": "green"
    }
    
    color = color_map.get(sentiment, "black")

    st.markdown(
        f"<h4>Sentiment: <span style='color:{color}'>{sentiment}</span></h4>",
        unsafe_allow_html=True
    )
