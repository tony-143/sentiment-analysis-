# Sentiment Analysis Project

A simple sentiment analysis model trained on Twitter reviews using logistic regression & RandomForestClassifier and deployed via Streamlit.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run notebook and save the model: `using jupiter Notebook`
3. Train the model: `python src/train_model.py`
4. Run the app: `streamlit run app.py`

## Features
- Text cleaning and preprocessing
- TF-IDF vectorization
- Logistic Regression classifier and RandomForestClassifier
- Streamlit UI for predictions

## Sample Output
Input: "This movie was amazing!"
Output: Positive

## Dataset
IMDb dataset: [https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
"""
