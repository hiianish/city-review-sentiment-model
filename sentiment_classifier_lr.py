import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


df = pd.read_csv("category_labeled_reviews.csv")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_tfidf = tfidf.fit_transform(df["clean_text"])

def label_sentiment(rating):
    if rating >= 4:
        return 1   
    elif rating <= 2:
        return 0   
    else:
        return 2   
df["sentiment_label"] = df["Rating"].apply(label_sentiment)


df_binary = df[df["sentiment_label"] != 2].copy()


X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf[df_binary.index],
    df_binary["sentiment_label"],
    test_size=0.2,
    random_state=42
)


model_sent = LogisticRegression(max_iter=400, n_jobs=-1)
model_sent.fit(X_train, y_train)

y_pred = model_sent.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Sentiment model accuracy: {acc:.3f}")
print("\nClassification report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False)
plt.title("Logistic Regression Sentiment Model — Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


joblib.dump(model_sent, "sentiment_model.pkl")
df.to_csv("final_labeled_reviews.csv", index=False)

print("Logistic Regression Sentiment Model trained")
