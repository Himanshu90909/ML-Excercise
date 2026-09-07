"""
Day 42: Non-Linear SVM & Kernels
Topic: Kernel trick, RBF, polynomial, sigmoid kernels
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# --- Non-linear data ---
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- Compare different kernels ---
kernels = {
    'linear': SVC(kernel='linear', C=1.0),
    'poly': SVC(kernel='poly', degree=3, C=1.0, gamma='scale'),
    'rbf': SVC(kernel='rbf', C=1.0, gamma='scale'),
    'sigmoid': SVC(kernel='sigmoid', C=1.0, gamma='scale'),
}

print("Kernel Comparison on Moons Dataset:")
print("-" * 45)
for name, clf in kernels.items():
    clf.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test_s))
    print(f"  {name:12s} -> Accuracy: {acc:.4f}")

# --- Circular data ---
X_c, y_c = make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42)
X_c_s = StandardScaler().fit_transform(X_c)
X_ctr, X_cte, y_ctr, y_cte = train_test_split(X_c_s, y_c, test_size=0.3, random_state=42)

print("\nKernel Comparison on Circles Dataset:")
print("-" * 45)
for name, clf in kernels.items():
    clf.fit(X_ctr, y_ctr)
    acc = accuracy_score(y_cte, clf.predict(X_cte))
    print(f"  {name:12s} -> Accuracy: {acc:.4f}")

# --- Gamma effect on RBF ---
print("\nEffect of Gamma on RBF Kernel:")
print("-" * 35)
for gamma in [0.01, 0.1, 1.0, 10.0]:
    clf = SVC(kernel='rbf', gamma=gamma, C=1.0)
    clf.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test_s))
    print(f"  gamma={gamma:<6} -> Accuracy: {acc:.4f}")

print("\nKey Concepts:")
print("- Kernel trick maps data to higher-dimensional space without explicit computation")
print("- RBF (Gaussian) kernel: K(x,y) = exp(-gamma * ||x-y||^2)")
print("- Polynomial kernel: K(x,y) = (gamma*x*y + r)^degree")
print("- gamma controls the influence range of each training sample")
