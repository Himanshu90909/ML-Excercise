"""
Day 41: SVM - Basics & Implementation
Topic: Support Vector Machine fundamentals, linear SVM, margin maximization
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# --- Linear SVM on synthetic data ---
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                          n_informative=2, random_state=42, n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_s, y_train)
y_pred = svm_linear.predict(X_test_s)
print(f"Linear SVM Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Support vectors: {svm_linear.n_support_}")
print(f"W (weights): {svm_linear.coef_}")
print(f"b (intercept): {svm_linear.intercept_}")

# --- SVM on Iris dataset ---
iris = load_iris()
X_iris = iris.data[:, :2]  # first 2 features for visualization
y_iris = iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)
X_tr_s = StandardScaler().fit_transform(X_tr)
X_te_s = StandardScaler().fit(X_tr).transform(X_te)

svm_iris = SVC(kernel='linear', C=1.0, decision_function_shape='ovo')
svm_iris.fit(X_tr_s, y_tr)
print(f"\nIris SVM Accuracy: {accuracy_score(y_te, svm_iris.predict(X_te_s)):.4f}")
print(classification_report(y_te, svm_iris.predict(X_te_s), target_names=iris.target_names))

print("\nKey Concepts:")
print("- SVM finds the hyperplane that maximizes the margin between classes")
print("- Support vectors are the data points closest to the decision boundary")
print("- C parameter controls trade-off between margin width and classification error")
print("- Large C -> narrow margin (may overfit), Small C -> wide margin (may underfit)")
