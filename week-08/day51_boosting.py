"""
===================================================================================
GeeksforGeeks Machine Learning Course
Week 7/8: Machine Learning Algorithms
Day 51: Boosting in Machine Learning

Topics Covered:
- Introduction to Boosting algorithms (Sequential Error Correction)
- AdaBoost (Adaptive Boosting) classifier mechanics
- Base estimators (Decision Stumps: trees with max_depth=1)
- Sample re-weighting logic: increasing weights on misclassified samples
- Bagging vs Boosting performance & bias-variance trade-off
===================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set seed
np.random.seed(42)

print("=" * 80)
print("DAY 51: BOOSTING IN MACHINE LEARNING (ADABOOST)")
print("=" * 80)

# ===================================================================================
# 1. ADABOOST SEQUENTIAL WEIGHT UPDATE CONCEPT DEMONSTRATION
# ===================================================================================

print("\n--- Step 1: AdaBoost Weight Update Mathematical Intuition ---")

n_samples = 5
sample_weights = np.ones(n_samples) / n_samples  # Initially uniform weights 1/N = 0.20
y_true = np.array([1, 1, -1, 1, -1])
y_pred_stump = np.array([1, -1, -1, 1, -1])     # Sample index 1 misclassified

incorrect = (y_pred_stump != y_true)
weighted_error = np.sum(sample_weights[incorrect])

# Estimator weight alpha = 0.5 * log((1 - error) / error)
alpha = 0.5 * np.log((1.0 - weighted_error) / (weighted_error + 1e-10))

# Update weights: w_i -> w_i * exp(-alpha * y_i * h(x_i))
new_weights = sample_weights * np.exp(-alpha * y_true * y_pred_stump)
new_weights /= np.sum(new_weights)              # Normalize to sum to 1.0

print(f"Initial Weights:           {sample_weights}")
print(f"Misclassified Indices:     {np.where(incorrect)[0].tolist()}")
print(f"Stump Weighted Error:      {weighted_error:.4f}")
print(f"Stump Estimator Weight α:  {alpha:.4f}")
print(f"Updated Sample Weights:    {np.round(new_weights, 4).tolist()}\n")


# ===================================================================================
# 2. ADABOOSTCLASSIFIER MODEL TRAINING
# ===================================================================================

print("--- Step 2: Training AdaBoostClassifier on Synthetic Data ---")

X, y = make_classification(
    n_samples=600, n_features=10, n_informative=8, n_redundant=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Stump (Weak Learner)
stump = DecisionTreeClassifier(max_depth=1, random_state=42)
stump.fit(X_train, y_train)
acc_stump = stump.score(X_test, y_test)
print(f"1. Single Weak Estimator (Decision Stump, max_depth=1) Accuracy: {acc_stump * 100:.2f}%")

# AdaBoost Ensemble of 50 Decision Stumps
adaboost_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

adaboost_clf.fit(X_train, y_train)
y_pred_ada = adaboost_clf.predict(X_test)
acc_ada = accuracy_score(y_test, y_pred_ada)
print(f"2. AdaBoost Ensemble (50 Stumps, learning_rate=1.0) Accuracy:  {acc_ada * 100:.2f}%\n")


# ===================================================================================
# 3. BAGGING VS BOOSTING COMPARISON
# ===================================================================================

print("--- Step 3: Bagging vs Boosting Performance ---")

bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
    n_estimators=50,
    random_state=42
)

bagging_clf.fit(X_train, y_train)
acc_bag = bagging_clf.score(X_test, y_test)

print(f"Bagging with 50 Stumps Accuracy:  {acc_bag * 100:.2f}%")
print(f"AdaBoost with 50 Stumps Accuracy: {acc_ada * 100:.2f}%")
print("Notice: Boosting builds strong learners from weak learners by focusing on difficult samples sequentially.")
print("=" * 80)
