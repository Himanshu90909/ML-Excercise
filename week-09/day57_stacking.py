"""
Day 57: Stacking in Machine Learning
Topic: StackingClassifier, meta-learner with multiple base models
"""
import numpy as np
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=500, n_features=20, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Base models ---
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
]

# --- Stacking with Logistic Regression as meta-learner ---
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1
)

# --- Compare individual models vs stacking ---
print("Individual Model Performance (CV=5):")
print("-" * 50)
for name, model in base_models:
    cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  {name:5s} -> CV: {cv_score:.4f}, Test: {acc:.4f}")

# --- Stacking ---
stacking.fit(X_train, y_train)
stack_pred = stacking.predict(X_test)
stack_cv = cross_val_score(stacking, X_train, y_train, cv=5).mean()
print(f"\n  Stacking -> CV: {stack_cv:.4f}, Test: {accuracy_score(y_test, stack_pred):.4f}")
print(f"  Meta-learner coefficients: {stacking.final_estimator_.coef_[0][:4].round(3)}")

print("\nKey Concepts:")
print("- Stacking trains multiple base models on the full training data")
print("- Base model predictions become features for the meta-learner")
print("- Meta-learner learns optimal weights to combine base model predictions")
print("- Uses cross-validation to prevent data leakage between base and meta levels")
