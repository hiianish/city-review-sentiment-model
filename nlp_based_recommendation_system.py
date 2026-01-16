import pandas as pd
import numpy as np
import re
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk


nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


df = pd.read_csv("final_labeled_reviews.csv")
tfidf = joblib.load("tfidf_vectorizer.pkl")
X_tfidf = joblib.load("tfidf_features.pkl")
model_cat = joblib.load("category_classifier.pkl")


stop_words = set(stopwords.words("english"))
def clean_query(text):
    text = re.sub(r"[^a-z\s]", " ", str(text).lower())
    tokens = [w for w in word_tokenize(text) if w not in stop_words and len(w) > 2]
    return " ".join(tokens)


def recommend_city(query, top_k=10):
    query_clean = clean_query(query)
    query_vec = tfidf.transform([query_clean])

    
    sim_scores = cosine_similarity(query_vec, X_tfidf)[0]  # cosine similarity
    query_cat = model_cat.predict(query_vec)[0]
    cat_match = (df["category_cluster"] == query_cat).astype(int)
    sent_scores = df["sentiment_label"].astype(int)

    
    rating_norm = df["Rating"] / df["Rating"].max()
    pop_score = df.groupby("Place")["Review"].transform("count")
    pop_norm = pop_score / pop_score.max()

    
    final_score = (
        0.5 * sim_scores +
        0.15 * cat_match +
        0.15 * sent_scores +
        0.1 * rating_norm +
        0.1 * pop_norm
    )

    
    df_temp = df.copy()
    df_temp["similarity"] = sim_scores
    df_temp["final_score"] = final_score

    
    df_temp = df_temp[df_temp["sentiment_label"].isin([0, 1])]
    df_temp = df_temp.sort_values("final_score", ascending=False)

    
    df_temp = df_temp.drop_duplicates(subset=["Place"], keep="first").head(top_k)

    
    df_temp["sentiment_label"] = df_temp["sentiment_label"].map({1: "Positive", 0: "Negative"})
    df_temp["Review"] = df_temp["Review"].astype(str).apply(lambda x: (x[:120] + "...") if len(x) > 120 else x)

    
    print("\n" + "="*90)
    print(f"Top {len(df_temp)} Recommendations for: '{query}'")
    print("="*90)
    for _, row in df_temp.iterrows():
        print(f"\n{row['City']} — {row['Place']}")
        print(f"  Sentiment: {row['sentiment_label']} | Rating: {row['Rating']} | Similarity: {row['similarity']*100:.2f}%")
        print(f"  Review: {row['Review']}")
    print("\n" + "="*90)


recommend_city("hill station", top_k=10)
