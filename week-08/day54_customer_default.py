"""
Day 54: Customer Default Prediction - Project
Topic: End-to-end credit risk classification project
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)
from sklearn.datasets import make_classification

# --- Generate synthetic credit data ---
np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'credit_score': np.random.normal(650, 100, n).clip(300, 850),
    'income': np.random.lognormal(10, 0.5, n),
    'debt_ratio': np.random.uniform(0, 1, n),
    'late_payments': np.random.poisson(2, n),
    'credit_age_years': np.random.uniform(1, 20, n),
    'num_accounts': np.random.randint(1, 15, n),
})
data['default'] = ((data['debt_ratio'] > 0.5) | 
                   (data['late_payments'] > 4) | 
                   (data['credit_score'] < 550)).astype(int)

print("Dataset Info:")
print(f"  Samples: {len(data)}")
print(f"  Default rate: {data['default'].mean():.2%}")
print(data.describe().round(2))

# --- Feature engineering ---
data['credit_utilization'] = data['debt_ratio'] * data['num_accounts']
data['risk_score'] = (850 - data['credit_score']) / 50 + data['late_payments'] * 2

# --- Split ---
X = data.drop('default', axis=1)
y = data['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- Train models ---
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42),
}

print("\nModel Comparison:")
print("-" * 60)
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cv = cross_val_score(model, X_train_s, y_train, cv=5).mean()
    print(f"  {name:25s} | Acc: {acc:.4f} | AUC: {auc:.4f} | CV: {cv:.4f}")

# --- Best model details ---
gb = models['Gradient Boosting']
y_pred = gb.predict(X_test_s)
print("\nGradient Boosting Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print("\nKey Features: credit_score, debt_ratio, late_payments are strongest predictors")
