"""
===================================================================================
GeeksforGeeks Machine Learning Course
Week 8: Machine Learning Algorithms
Day 52: Gradient Boosting

Topics Covered:
- Stage-wise additive modeling & gradient descent in function space
- Fitting base trees on pseudo-residuals
- GradientBoostingClassifier & GradientBoostingRegressor in scikit-learn
- Key hyperparameters: learning_rate, n_estimators, subsample, max_depth
- Staged predictions monitoring (staged_predict)
===================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Set seed
np.random.seed(42)

print("=" * 80)
print("DAY 52: GRADIENT BOOSTING ALGORITHMS")
print("=" * 80)

# ===================================================================================
# 1. GRADIENT BOOSTING REGRESSION STEP-BY-STEP CONCEPT
# ===================================================================================

print("\n--- Step 1: Gradient Boosting Regression Intuition ---")

# Simple 1D toy regression problem
X_toy = np.array([[1], [2], [3], [4], [5]])
y_true = np.array([10.0, 13.0, 18.0, 25.0, 30.0])

# Step 0: Initial prediction F_0(x) = mean(y)
F_0 = np.mean(y_true)
residuals_0 = y_true - F_0

print(f"Target values y:                {y_true.tolist()}")
print(f"F0 (Initial Mean Prediction):   {F_0:.2f}")
print(f"Step 0 Residuals (y - F0):     {residuals_0.tolist()}")

learning_rate = 0.1
# In Stage 1: a decision tree h_1(x) is trained to fit residuals_0
# Simulating tree output prediction for residuals
h_1 = residuals_0 * 0.5   # Toy approximation of tree fit on residual
F_1 = F_0 + learning_rate * h_1
residuals_1 = y_true - F_1

print(f"F1 Prediction (F0 + lr * h1):  {F_1.round(2).tolist()}")
print(f"Step 1 Residuals (y - F1):     {residuals_1.round(2).tolist()}\n")


# ===================================================================================
# 2. GRADIENT BOOSTING CLASSIFIER
# ===================================================================================

print("--- Step 2: GradientBoostingClassifier on Synthetic Classification Dataset ---")

X_c, y_c = make_classification(
    n_samples=500, n_features=10, n_informative=7, n_redundant=3,
    random_state=42
)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.3, random_state=42)

gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,    # Stochastic Gradient Boosting
    random_state=42
)

gb_clf.fit(X_train_c, y_train_c)
y_pred_c = gb_clf.predict(X_test_c)
acc_c = accuracy_score(y_test_c, y_pred_c)

print(f"GradientBoostingClassifier Accuracy: {acc_c * 100:.2f}%")


# ===================================================================================
# 3. MONITORING STAGED PREDICTIONS (LEARNING CURVE)
# ===================================================================================

print("\n--- Step 3: Staged Predictions Evaluation (Loss Reduction per Stage) ---")

staged_accs = [accuracy_score(y_test_c, stage_pred) for stage_pred in gb_clf.staged_predict(X_test_c)]

print(f"Accuracy at Stage 1 (1st tree):   {staged_accs[0] * 100:.2f}%")
print(f"Accuracy at Stage 25 (25 trees):  {staged_accs[24] * 100:.2f}%")
print(f"Accuracy at Stage 50 (50 trees):  {staged_accs[49] * 100:.2f}%")
print(f"Accuracy at Stage 100 (100 trees): {staged_accs[99] * 100:.2f}%\n")


# ===================================================================================
# 4. GRADIENT BOOSTING REGRESSOR
# ===================================================================================

print("--- Step 4: GradientBoostingRegressor on Synthetic Regression Dataset ---")

X_r, y_r = make_regression(n_samples=500, n_features=8, noise=12.0, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r, y_r, test_size=0.3, random_state=42)

gb_reg = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

gb_reg.fit(X_train_r, y_train_r)
y_pred_r = gb_reg.predict(X_test_r)

rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
r2 = r2_score(y_test_r, y_pred_r)

print(f"GradientBoostingRegressor RMSE:     {rmse:.4f}")
print(f"GradientBoostingRegressor R² Score: {r2:.4f}")
print("=" * 80)
