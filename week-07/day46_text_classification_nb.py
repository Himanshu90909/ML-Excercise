"""
===================================================================================
GeeksforGeeks Machine Learning Course
Week 7: Machine Learning Algorithms
Day 46: Text Classification using Naive Bayes

Topics Covered:
- Text classification pipeline (Spam vs Ham Email Detection)
- Text Vectorization: CountVectorizer vs TfidfVectorizer
- Building Scikit-Learn Pipeline with MultinomialNB
- Model evaluation with Confusion Matrix, Precision, Recall, and F1-Score
===================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("=" * 80)
print("DAY 46: TEXT CLASSIFICATION USING NAIVE BAYES (SPAM DETECTION)")
print("=" * 80)

# ===================================================================================
# 1. CREATE SYNTHETIC TEXT DATASET
# ===================================================================================

emails = [
    # Spam emails (1)
    ("Win a $1000 gift card now! Click here to claim your cash reward", 1),
    ("Free trial membership! Limited time offer click link immediately", 1),
    ("Urgent action required: Your account password has expired verify now", 1),
    ("Exclusive loan offer 0% interest guaranteed instant approval", 1),
    ("Congratulations you have won a free vacation to Bahamas click now", 1),
    ("Earn $500 per day working from home simple task click here", 1),
    ("Double your income fast with low risk investment opportunity", 1),
    ("Claim your free bonus money now click this special link", 1),
    ("Selected winner! You won top prize reply with account details", 1),
    ("Discount Viagra fast delivery cheap price order today", 1),
    
    # Ham emails (0)
    ("Hi John, can we schedule our weekly project sync meeting tomorrow?", 0),
    ("Please find attached the quarterly financial report for review", 0),
    ("Are you coming to the team lunch at 1pm today?", 0),
    ("Reminder: Project deadline is next Friday please update your status", 0),
    ("Thank you for your application we would like to schedule an interview", 0),
    ("The code repository was updated with the latest bug fixes", 0),
    ("Here are the lecture slides from yesterday's machine learning class", 0),
    ("Could you please review the attached document and send feedback?", 0),
    ("Flight confirmation for your upcoming trip to Chicago", 0),
    ("Meeting notes from today's discussion with client are saved in drive", 0)
]

df = pd.DataFrame(emails, columns=['text', 'label'])
print(f"Dataset Size: {len(df)} samples")
print(f"Spam count: {df['label'].sum()}, Ham count: {len(df) - df['label'].sum()}\n")

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.3, random_state=42, stratify=df['label']
)


# ===================================================================================
# 2. COMPARISON: CountVectorizer vs TfidfVectorizer
# ===================================================================================

print("--- Vectorization Methods Comparison ---")

# Method A: CountVectorizer + MultinomialNB
pipeline_count = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english')),
    ('classifier', MultinomialNB())
])

pipeline_count.fit(X_train, y_train)
y_pred_count = pipeline_count.predict(X_test)
acc_count = accuracy_score(y_test, y_pred_count)
print(f"1. CountVectorizer + MultinomialNB Accuracy: {acc_count * 100:.2f}%")

# Method B: TfidfVectorizer + MultinomialNB
pipeline_tfidf = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english')),
    ('classifier', MultinomialNB())
])

pipeline_tfidf.fit(X_train, y_train)
y_pred_tfidf = pipeline_tfidf.predict(X_test)
acc_tfidf = accuracy_score(y_test, y_pred_tfidf)
print(f"2. TfidfVectorizer + MultinomialNB Accuracy:  {acc_tfidf * 100:.2f}%\n")


# ===================================================================================
# 3. DETAILED EVALUATION & CONFUSION MATRIX
# ===================================================================================

print("--- Model Performance Metrics (TF-IDF Pipeline) ---")
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_tfidf)
print(f"TN: {cm[0,0]}  FP: {cm[0,1]}")
print(f"FN: {cm[1,0]}  TP: {cm[1,1]}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred_tfidf, target_names=['Ham (0)', 'Spam (1)']))


# ===================================================================================
# 4. PREDICT ON UNSEEN SAMPLE MESSAGES
# ===================================================================================

print("--- Testing Pipeline on Custom Messages ---")
new_messages = [
    "Urgent! You have won $1000 cash prize click link now",
    "Hi team, please find the meeting agenda attached for review",
    "Free loan approval guaranteed reply with bank account"
]

predictions = pipeline_tfidf.predict(new_messages)
probabilities = pipeline_tfidf.predict_proba(new_messages)

for msg, pred, proba in zip(new_messages, predictions, probabilities):
    label = "SPAM" if pred == 1 else "HAM"
    confidence = proba[pred] * 100
    print(f"Message: '{msg}'")
    print(f"Prediction: [{label}] (Confidence: {confidence:.2f}%)\n")

print("=" * 80)
