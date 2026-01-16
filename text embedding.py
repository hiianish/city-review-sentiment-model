import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

df = pd.read_csv("preprocessed_city_reviews.csv")

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_tfidf = tfidf.fit_transform(df["clean_text"])

joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(X_tfidf, "tfidf_features.pkl")

print("TF-IDF vectorizer and feature matrix saved")
