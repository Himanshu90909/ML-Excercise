"""
===================================================================================
GeeksforGeeks Machine Learning Course
Week 7: Machine Learning Algorithms
Day 47: Ensemble Learning

Topics Covered:
- Introduction to Ensemble Learning principles
- Hard Voting vs Soft Voting Classifiers
- Individual base models vs Ensemble model performance
- Conceptual comparison: Bagging vs Boosting vs Stacking
===================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set seed
np.random.seed(42)

print("=" * 80)
print("DAY 47: INTRODUCTION TO ENSEMBLE LEARNING")
print("=" * 80)

# ===================================================================================
# 1. GENERATE SYNTHETIC DATASET
# ===================================================================================

X, y = make_classification(
    n_samples=500, n_features=10, n_informative=8, n_redundant=2,
    n_classes=2, flip_y=0.05, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Dataset shape: {X.shape}")
print(f"Training set: {X_train.shape[0]} samples, Testing set: {X_test.shape[0]} samples\n")


# ===================================================================================
# 2. INDIVIDUAL BASE LEARNERS
# ===================================================================================

print("--- Step 1: Base Classifier Evaluation ---")

# Define base estimators
clf1 = LogisticRegression(random_state=42)
clf2 = DecisionTreeClassifier(max_depth=4, random_state=42)
clf3 = SVC(probability=True, random_state=42)

models = {
    'Logistic Regression': clf1,
    'Decision Tree': clf2,
    'Support Vector Machine': clf3
}

base_acc = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    base_acc[name] = acc
    print(f"Base Model: {name:<25} | Accuracy: {acc * 100:.2f}%")


# ===================================================================================
# 3. ENSEMBLE: HARD VOTING VS SOFT VOTING
# ===================================================================================

print("\n--- Step 2: Voting Classifier Ensembles ---")

# Hard Voting (Majority Rule)
voting_hard = VotingClassifier(
    estimators=[('lr', clf1), ('dt', clf2), ('svm', clf3)],
    voting='hard'
)
voting_hard.fit(X_train, y_train)
y_pred_hard = voting_hard.predict(X_test)
acc_hard = accuracy_score(y_test, y_pred_hard)
print(f"Ensemble Model: Hard Voting Classifier   | Accuracy: {acc_hard * 100:.2f}%")

# Soft Voting (Weighted Average Probabilities)
voting_soft = VotingClassifier(
    estimators=[('lr', clf1), ('dt', clf2), ('svm', clf3)],
    voting='soft'
)
voting_soft.fit(X_train, y_train)
y_pred_soft = voting_soft.predict(X_test)
acc_soft = accuracy_score(y_test, y_pred_soft)
print(f"Ensemble Model: Soft Voting Classifier   | Accuracy: {acc_soft * 100:.2f}%")


# ===================================================================================
# 4. ENSEMBLE PARADIGMS SUMMARY
# ===================================================================================

print("\n" + "=" * 80)
print("ENSEMBLE METHODOLOGY COMPARISON SUMMARY")
print("=" * 80)

summary = pd.DataFrame({
    'Paradigm': ['Voting', 'Bagging (Bootstrap Aggregation)', 'Boosting', 'Stacking'],
    'Strategy': [
        'Combines diverse individual models via majority vote or average',
        'Trains homogenous base models independently on bootstrap random subsets',
        'Trains base models sequentially, focusing on misclassified samples',
        'Trains meta-learner to combine predictions of multiple base models'
    ],
    'Goal': [
        'Reduce variance and errors of individual models',
        'Reduce Variance (overfitting control)',
        'Reduce Bias and Variance',
        'Maximize predictive accuracy via meta-learning'
    ],
    'Example Algorithm': [
        'VotingClassifier',
        'Random Forest, BaggingClassifier',
        'AdaBoost, Gradient Boosting, XGBoost, CatBoost',
        'StackingClassifier'
    ]
})

print(summary.to_string(index=False))
print("=" * 80)
