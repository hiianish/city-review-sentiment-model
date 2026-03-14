import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("preprocessed_city_reviews.csv")


df["sentiment"] = df["Rating"].apply(lambda x: 1 if x >= 4 else 0)

X = df["clean_text"]
y = df["sentiment"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}


for name, model in models.items():

    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

while True:

    review = input("\nEnter a review (type exit to stop): ")

    if review.lower() == "exit":
        break

    review_vec = vectorizer.transform([review])
    prediction = models.predict(review_vec)[0]

    if prediction == 1:
        print("Positive")
    else:
        print("Negative")