"""
===================================================================================
GeeksforGeeks Machine Learning Course
Week 7: Machine Learning Algorithms
Day 49: Random Forest Classifier and Regression

Topics Covered:
- RandomForestClassifier and RandomForestRegressor
- Core concepts: Bagging + Random Feature Subspacing (max_features = sqrt(p))
- Key hyperparameters: n_estimators, max_depth, min_samples_split, max_features
- Feature importance calculation and ranking
===================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Set seed
np.random.seed(42)

print("=" * 80)
print("DAY 49: RANDOM FOREST CLASSIFICATION AND REGRESSION")
print("=" * 80)

# ===================================================================================
# 1. RANDOM FOREST CLASSIFICATION
# ===================================================================================

print("\n--- Part 1: Random Forest Classification ---")

feature_names_clf = ['Age', 'Income', 'Credit_Score', 'Years_Employed', 'Debt_Ratio', 'Num_Accounts']
X_clf, y_clf = make_classification(
    n_samples=500, n_features=6, n_informative=4, n_redundant=2,
    random_state=42
)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)

rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    max_features='sqrt',
    min_samples_split=4,
    random_state=42,
    n_jobs=-1
)

rf_clf.fit(X_train_c, y_train_c)
y_pred_c = rf_clf.predict(X_test_c)
acc_c = accuracy_score(y_test_c, y_pred_c)

print(f"RandomForestClassifier Accuracy: {acc_c * 100:.2f}%\n")

# Feature Importance Ranking
print("Feature Importances (Classification):")
importances_clf = pd.DataFrame({
    'Feature': feature_names_clf,
    'Importance': rf_clf.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in importances_clf.iterrows():
    print(f"  {row['Feature']:<15}: {row['Importance']:.4f} ({'█' * int(row['Importance'] * 30)})")


# ===================================================================================
# 2. RANDOM FOREST REGRESSION
# ===================================================================================

print("\n--- Part 2: Random Forest Regression ---")

feature_names_reg = ['Square_Feet', 'Bedrooms', 'Bathrooms', 'Zip_Rating', 'House_Age']
X_reg, y_reg = make_regression(
    n_samples=500, n_features=5, noise=10.0, random_state=42
)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    max_features=1.0,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

rf_reg.fit(X_train_r, y_train_r)
y_pred_r = rf_reg.predict(X_test_r)

rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
r2 = r2_score(y_test_r, y_pred_r)

print(f"RandomForestRegressor RMSE:     {rmse:.4f}")
print(f"RandomForestRegressor R² Score: {r2:.4f}\n")

# Feature Importance Ranking
print("Feature Importances (Regression):")
importances_reg = pd.DataFrame({
    'Feature': feature_names_reg,
    'Importance': rf_reg.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in importances_reg.iterrows():
    print(f"  {row['Feature']:<15}: {row['Importance']:.4f} ({'█' * int(row['Importance'] * 30)})")

print("\n" + "=" * 80)
