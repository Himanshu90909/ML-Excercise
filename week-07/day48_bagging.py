"""
===================================================================================
GeeksforGeeks Machine Learning Course
Week 7: Machine Learning Algorithms
Day 48: Bagging Algorithm (Bootstrap Aggregation)

Topics Covered:
- Bootstrap sampling from scratch (sampling with replacement)
- Scikit-Learn BaggingClassifier
- Out-of-Bag (OOB) score estimation (~36.8% unselected samples)
- Variance reduction: Single Decision Tree vs Bagging Ensemble
===================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set seed
np.random.seed(42)

print("=" * 80)
print("DAY 48: BAGGING ALGORITHM (BOOTSTRAP AGGREGATION)")
print("=" * 80)

# ===================================================================================
# 1. BOOTSTRAP SAMPLING CONCEPT DEMONSTRATION
# ===================================================================================

print("\n--- Step 1: Bootstrap Sampling Concept ---")

dataset = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
n_samples = len(dataset)

# Perform bootstrap sample (sample with replacement of same size)
bootstrap_sample = np.random.choice(dataset, size=n_samples, replace=True)
oob_sample = np.setdiff1d(dataset, bootstrap_sample)

print(f"Original Data ({n_samples} items): {dataset.tolist()}")
print(f"Bootstrap Sample (Size {n_samples}): {bootstrap_sample.tolist()}")
print(f"Out-Of-Bag (OOB) Items:     {oob_sample.tolist()}")
print(f"Percentage of items left in OOB: {len(oob_sample)/n_samples * 100:.1f}% (Theoretical expected ~36.8%)\n")


# ===================================================================================
# 2. DATASET PREPARATION & BASELINE SINGLE TREE
# ===================================================================================

X, y = make_classification(
    n_samples=600, n_features=12, n_informative=8, n_redundant=4,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Single Decision Tree (prone to high variance)
single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)
y_pred_tree = single_tree.predict(X_test)
acc_tree = accuracy_score(y_test, y_pred_tree)

print(f"1. Single Decision Tree Accuracy: {acc_tree * 100:.2f}%")


# ===================================================================================
# 3. BAGGING CLASSIFIER WITH OOB SCORE
# ===================================================================================

# Bagging Classifier with Decision Trees as base estimators
bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    max_samples=1.0,
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

bagging_clf.fit(X_train, y_train)
y_pred_bag = bagging_clf.predict(X_test)
acc_bag = accuracy_score(y_test, y_pred_bag)

print(f"2. Bagging Ensemble (100 Trees) Accuracy: {acc_bag * 100:.2f}%")
print(f"3. Out-Of-Bag (OOB) Estimated Score:       {bagging_clf.oob_score_ * 100:.2f}%\n")


# ===================================================================================
# 4. EFFECT OF NUMBER OF ESTIMATORS IN BAGGING
# ===================================================================================

print("--- Step 3: Performance vs Number of Estimators ---")
for n_est in [5, 10, 25, 50, 100, 200]:
    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=n_est,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    bag.fit(X_train, y_train)
    test_acc = bag.score(X_test, y_test)
    print(f"n_estimators = {n_est:3d} | Test Accuracy: {test_acc * 100:.2f}% | OOB Score: {bag.oob_score_ * 100:.2f}%")

print("\nConclusion: Bagging significantly reduces variance and overfitting compared to a single decision tree.")
print("=" * 80)
