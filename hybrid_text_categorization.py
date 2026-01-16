import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("preprocessed_city_reviews.csv")


tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_tfidf = tfidf.fit_transform(df["clean_text"])
print("TF-IDF feature matrix shape:", X_tfidf.shape)



n_clusters = 8   
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df["category_cluster"] = kmeans.fit_predict(X_tfidf)

print("Cluster distribution:")
print(df["category_cluster"].value_counts())


plt.figure(figsize=(8,5))
sns.countplot(x="category_cluster", data=df, palette="coolwarm")
plt.title("Distribution of Review Clusters (Pseudo-Categories)")
plt.xlabel("Cluster ID")
plt.ylabel("Number of Reviews")
plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, df["category_cluster"], test_size=0.2, random_state=42
)

model_cat = LogisticRegression(max_iter=500)
model_cat.fit(X_train, y_train)

y_pred = model_cat.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Category classifier accuracy: {acc:.3f}")


joblib.dump(model_cat, "category_classifier.pkl")
joblib.dump(kmeans, "category_kmeans.pkl")
df.to_csv("category_labeled_reviews.csv", index=False)

