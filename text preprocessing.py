import pandas as pd
import re
import nltk
from bs4 import BeautifulSoup
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
tqdm.pandas()


df = pd.read_csv("indian_places_reviews.csv")
cols = ["City", "Place", "Review", "Rating", "Raw_Review"]
df = df[cols].dropna(subset=["Review"]).reset_index(drop=True)
print("Before cleaning:", df.shape)


df.drop_duplicates(subset=["Review"], inplace=True)
df = df[df["Review"].str.len() > 10]
df = df[df["Rating"].between(1, 5)]

print("After cleaning:", df.shape)


nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])

stop_words = set(stopwords.words('english'))

contractions = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
    "'t": " not", "'ve": " have", "'m": " am"
}

def clean_text(text):
    text = str(text)
    text = BeautifulSoup(text, "html.parser").get_text()  
    text = text.lower()
    text = re.sub(r"http\S+", "", text)                   
    text = re.sub(r"[^a-z\s]", " ", text)            
    for k,v in contractions.items():
        text = text.replace(k, v)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([t.lemma_ for t in doc if t.lemma_ not in stop_words and len(t) > 2])

def preprocess_text(text):
    return lemmatize_text(clean_text(text))


df["clean_text"] = df["Review"].progress_apply(preprocess_text)

df = df[df["clean_text"].str.len() > 10]
df.drop_duplicates(subset=["clean_text"], inplace=True)
df.to_csv("preprocessed_city_reviews.csv", index=False)

print("Preprocessing complete.")
print("Total cleaned reviews:", len(df))
print(df.head(3))
