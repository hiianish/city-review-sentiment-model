# STEP 3: Exploratory Data Analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import wordcloud
from collections import Counter


df = pd.read_csv("preprocessed_city_reviews.csv")

#1 Rating distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Rating", data=df, palette="viridis")
plt.title("Distribution of Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

# Review length distribution
df["review_length"] = df["clean_text"].apply(lambda x: len(x.split()))
plt.figure(figsize=(6,4))
sns.histplot(df["review_length"], bins=50, color="teal")
plt.title("Distribution of Review Lengths")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()

#Top 15 cities with most reviews
top_cities = df["City"].value_counts().head(15)
plt.figure(figsize=(8,5))
sns.barplot(x=top_cities.values, y=top_cities.index, palette="plasma")
plt.title("Top 15 Cities by Review Count")
plt.xlabel("Number of Reviews")
plt.ylabel("City")
plt.show()

#average rating per city
city_rating = df.groupby("City")["Rating"].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(8,5))
sns.barplot(x=city_rating.values, y=city_rating.index, palette="mako")
plt.title("Top 10 Cities by Average Rating")
plt.xlabel("Average Rating")
plt.ylabel("City")
plt.show()
