"""
===================================================================================
GeeksforGeeks Machine Learning Course
Week 7: Machine Learning Algorithms
Day 45: Types of Naive Bayes Classifiers

Topics Covered:
- Gaussian Naive Bayes: For continuous / real-valued features
- Multinomial Naive Bayes: For discrete count features (e.g., text frequency)
- Bernoulli Naive Bayes: For binary / boolean features (0/1 presence)
- When to use each variant and comparison on representative datasets
===================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

# Set seed
np.random.seed(42)

print("=" * 80)
print("DAY 45: TYPES OF NAIVE BAYES CLASSIFIERS")
print("=" * 80)

# ===================================================================================
# 1. GAUSSIAN NAIVE BAYES (Continuous Features)
# ===================================================================================
print("\n1. GAUSSIAN NAIVE BAYES (Continuous Measurement Features)")
print("Use Case: Features following normal distribution (e.g., height, weight, BP)")

# Synthetic dataset: Medical parameters
n_samples = 200
X_gaussian = np.column_stack([
    np.random.normal(loc=120, scale=15, size=n_samples),  # Blood Pressure
    np.random.normal(loc=200, scale=30, size=n_samples),  # Cholesterol
    np.random.normal(loc=45, scale=10, size=n_samples)    # Age
])
y_gaussian = (X_gaussian[:, 0] * 0.02 + X_gaussian[:, 1] * 0.01 > 4.5).astype(int)

X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_gaussian, y_gaussian, test_size=0.3, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train_g, y_train_g)
y_pred_g = gnb.predict(X_test_g)
print(f"GaussianNB Accuracy: {accuracy_score(y_test_g, y_pred_g) * 100:.2f}%")


# ===================================================================================
# 2. MULTINOMIAL NAIVE BAYES (Count Features)
# ===================================================================================
print("\n2. MULTINOMIAL NAIVE BAYES (Discrete Count Features)")
print("Use Case: Word counts, term frequencies, item counts")

# Synthetic dataset: Document word frequency matrix (e.g., counts of 5 vocabulary words)
X_multinomial = np.random.poisson(lam=3, size=(n_samples, 5))
y_multinomial = (X_multinomial[:, 0] * 2 + X_multinomial[:, 1] * 3 > 12).astype(int)

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multinomial, y_multinomial, test_size=0.3, random_state=42)

mnb = MultinomialNB()
mnb.fit(X_train_m, y_train_m)
y_pred_m = mnb.predict(X_test_m)
print(f"MultinomialNB Accuracy: {accuracy_score(y_test_m, y_pred_m) * 100:.2f}%")


# ===================================================================================
# 3. BERNOULLI NAIVE BAYES (Binary/Boolean Features)
# ===================================================================================
print("\n3. BERNOULLI NAIVE BAYES (Binary Features 0/1)")
print("Use Case: Word presence/absence, survey yes/no questions")

# Synthetic dataset: Binary feature matrix
X_bernoulli = np.random.binomial(n=1, p=0.4, size=(n_samples, 6))
y_bernoulli = (X_bernoulli[:, 0] + X_bernoulli[:, 2] + X_bernoulli[:, 4] >= 2).astype(int)

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_bernoulli, y_bernoulli, test_size=0.3, random_state=42)

bnb = BernoulliNB()
bnb.fit(X_train_b, y_train_b)
y_pred_b = bnb.predict(X_test_b)
print(f"BernoulliNB Accuracy: {accuracy_score(y_test_b, y_pred_b) * 100:.2f}%")


# ===================================================================================
# 4. COMPARISON SUMMARY TABLE
# ===================================================================================
print("\n" + "=" * 80)
print("NAIVE BAYES VARIANTS COMPARISON SUMMARY")
print("=" * 80)

summary_df = pd.DataFrame({
    'Classifier': ['GaussianNB', 'MultinomialNB', 'BernoulliNB'],
    'Feature Type': ['Continuous (Real numbers)', 'Discrete Counts (Non-negative integers)', 'Binary / Boolean (0 or 1)'],
    'Distribution Assumption': ['Gaussian / Normal', 'Multinomial', 'Bernoulli'],
    'Typical Applications': [
        'Iris dataset, Sensor data, Medical metrics',
        'Text classification (Bag of Words), Document tagging',
        'Spam detection (word presence), Survey responses'
    ]
})

print(summary_df.to_string(index=False))
print("=" * 80)
