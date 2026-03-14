import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



df = pd.read_csv("preprocessed_city_reviews.csv")

df["sentiment"] = df["Rating"].apply(lambda x: 1 if x >= 4 else 0)

X = df["clean_text"]
y = df["sentiment"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


lr = LogisticRegression(max_iter=500)
nb = MultinomialNB()
svm = LinearSVC()

ensemble_model = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('nb', nb),
        ('svm', svm)
    ],
    voting='hard'
)


ensemble_model.fit(X_train_vec, y_train)


y_pred = ensemble_model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()






while True:

    review = input("\nEnter a review (type exit to stop): ")

    if review.lower() == "exit":
        break

    review_vec = vectorizer.transform([review])
    prediction = ensemble_model.predict(review_vec)[0]

    if prediction == 1:
        print("Sentiment: Positive Review")
    else:
        print("Sentiment: Negative Review")