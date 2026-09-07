"""
Day 53: CatBoost in Machine Learning
Topic: CatBoost for categorical feature handling
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import make_classification

# --- CatBoost with synthetic categorical data ---
np.random.seed(42)
n = 500
data = pd.DataFrame({
    'age': np.random.randint(18, 70, n),
    'income': np.random.randint(20000, 120000, n),
    'city': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Pune', 'Chennai'], n),
    'occupation': np.random.choice(['Engineer', 'Doctor', 'Teacher', 'Business', 'Artist'], n),
    'education': np.random.choice(['High School', 'Bachelors', 'Masters', 'PhD'], n),
})
data['target'] = (data['income'] > 60000).astype(int)

cat_features = ['city', 'occupation', 'education']
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('target', axis=1), data['target'], test_size=0.3, random_state=42)

model = CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6,
                            cat_features=cat_features, verbose=0, random_seed=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"CatBoost Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# --- Feature importance ---
importances = model.get_feature_importance()
for name, imp in sorted(zip(data.columns[:-1], importances), key=lambda x: -x[1]):
    print(f"  {name:15s}: {imp:.2f}")

# --- CatBoost Regressor ---
X_reg, y_reg = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)
y_reg = y_reg.astype(float) + np.random.normal(0, 0.1, n)
X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
reg = CatBoostRegressor(iterations=200, learning_rate=0.1, depth=6, verbose=0, random_seed=42)
reg.fit(X_tr, y_tr)
print(f"\nCatBoost RMSE: {np.sqrt(mean_squared_error(y_te, reg.predict(X_te))):.4f}")

print("\nKey Concepts:")
print("- CatBoost handles categorical features natively without manual encoding")
print("- Uses ordered boosting to prevent target leakage")
print("- Symmetric tree structure for faster training")
