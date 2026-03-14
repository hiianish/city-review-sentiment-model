# Semantic City Recommendation System

## Overview
This project builds an NLP-based system that analyzes city reviews and performs two main tasks:
1. Sentiment classification of reviews (positive or negative).
2. Recommendation of similar cities based on semantic similarity of reviews.

The system uses machine learning and natural language processing techniques to understand user opinions and recommend relevant cities.

---

## Dataset
Dataset: Indian Places to Visit Reviews Data  
Source: Kaggle  
Contains reviews, city names, places, and ratings for tourist destinations across India.

---

## Project Pipeline

### 1. Text Preprocessing
- Removal of HTML tags, URLs, and special characters
- Lowercasing text
- Stopword removal
- Lemmatization using spaCy

### 2. Feature Engineering
- TF-IDF vectorization
- N-grams (unigram + bigram)
- Maximum features: 5000

### 3. Review Categorization
- K-Means clustering used to generate pseudo categories
- Logistic Regression classifier trained on cluster labels

### 4. Sentiment Classification
Reviews are classified into:
- Positive
- Negative

Models used:
- Random Forest (ensemble method)
- Gradient Boosting
- AdaBoost
- XGBoost

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### 5. City Recommendation
Cities are recommended using semantic similarity between reviews and user query.

---

## Model Performance
Example results from sentiment classification:

Accuracy: ~96%

Classification Metrics:

| Class | Precision | Recall | F1-score |
|------|------|------|------|
| Negative | 0.91 | 0.87 | 0.89 |
| Positive | 0.97 | 0.98 | 0.98 |

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- spaCy
- NLTK
- Matplotlib
- Seaborn
- Joblib


