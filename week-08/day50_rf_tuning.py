"""
===================================================================================
GeeksforGeeks Machine Learning Course
Week 8: Machine Learning Algorithms
Day 50: Hyperparameter Tuning in Random Forest

Topics Covered:
- Hyperparameter tuning using GridSearchCV vs RandomizedSearchCV
- Key Random Forest parameters: n_estimators, max_depth, min_samples_split, min_samples_leaf
- Out-Of-Bag (OOB) error monitoring during tuning
- Performance benchmark: Default vs GridSearch vs RandomizedSearch
===================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Set seed
np.random.seed(42)

print("=" * 80)
print("DAY 50: HYPERPARAMETER TUNING IN RANDOM FOREST")
print("=" * 80)

# ===================================================================================
# 1. SYNTHETIC DATASET GENERATION
# ===================================================================================

X, y = make_classification(
    n_samples=800, n_features=15, n_informative=10, n_redundant=5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(f"Dataset Shape: {X.shape}")
print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}\n")


# ===================================================================================
# 2. BASELINE RANDOM FOREST MODEL
# ===================================================================================

rf_base = RandomForestClassifier(random_state=42, oob_score=True)
rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)
acc_base = accuracy_score(y_test, y_pred_base)
oob_base = rf_base.oob_score_

print("1. BASELINE RANDOM FOREST (Default Parameters)")
print(f"   Test Accuracy: {acc_base * 100:.2f}%")
print(f"   OOB Score:     {oob_base * 100:.2f}%\n")


# ===================================================================================
# 3. GRID SEARCH CV
# ===================================================================================

print("2. GRID SEARCH CV (Exhaustive Parameter Grid)")

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_rf_grid = grid_search.best_estimator_
y_pred_grid = best_rf_grid.predict(X_test)
acc_grid = accuracy_score(y_test, y_pred_grid)

print(f"   Best Parameters: {grid_search.best_params_}")
print(f"   Best CV Score:   {grid_search.best_score_ * 100:.2f}%")
print(f"   Test Accuracy:   {acc_grid * 100:.2f}%\n")


# ===================================================================================
# 4. RANDOMIZED SEARCH CV
# ===================================================================================

print("3. RANDOMIZED SEARCH CV (Randomized Sampling)")

param_dist = {
    'n_estimators': [int(x) for x in np.linspace(start=30, stop=150, num=5)],
    'max_depth': [3, 5, 8, 12, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=15,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
best_rf_random = random_search.best_estimator_
y_pred_random = best_rf_random.predict(X_test)
acc_random = accuracy_score(y_test, y_pred_random)

print(f"   Best Parameters: {random_search.best_params_}")
print(f"   Best CV Score:   {random_search.best_score_ * 100:.2f}%")
print(f"   Test Accuracy:   {acc_random * 100:.2f}%\n")


# ===================================================================================
# 5. COMPARISON SUMMARY
# ===================================================================================

print("=" * 80)
print("HYPERPARAMETER TUNING BENCHMARK SUMMARY")
print("=" * 80)

summary_df = pd.DataFrame({
    'Method': ['Default Baseline', 'GridSearchCV', 'RandomizedSearchCV'],
    'Test Accuracy (%)': [acc_base * 100, acc_grid * 100, acc_random * 100],
    'F1 Score': [f1_score(y_test, y_pred_base), f1_score(y_test, y_pred_grid), f1_score(y_test, y_pred_random)]
})

print(summary_df.to_string(index=False))
print("=" * 80)
