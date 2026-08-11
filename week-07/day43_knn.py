"""
===================================================================================
GeeksforGeeks Machine Learning Course
Week 7: Machine Learning Algorithms
Day 43: K-Nearest Neighbors (KNN)

Topics Covered:
- K-Nearest Neighbors algorithm implementation from scratch
- Distance metrics: Euclidean, Manhattan, and Minkowski distance
- Scikit-Learn KNeighborsClassifier and KNeighborsRegressor
- Optimal K selection using cross-validation (Elbow Method)
===================================================================================
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.datasets import make_classification, make_regression

# Set random seed for reproducibility
np.random.seed(42)

# ===================================================================================
# SECTION 1: KNN Implementation from Scratch
# ===================================================================================

def euclidean_distance(x1, x2):
    """Calculate Euclidean distance between two vectors."""
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    """Calculate Manhattan distance between two vectors."""
    return np.sum(np.abs(x1 - x2))

class KNNClassifierScratch:
    """K-Nearest Neighbors Classifier built from scratch."""
    
    def __init__(self, k=3, metric='euclidean'):
        self.k = k
        self.metric = metric
        
    def fit(self, X, y):
        """Fit the model by storing training data."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
    def _predict_single(self, x):
        # Compute distance to all training points
        if self.metric == 'euclidean':
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        elif self.metric == 'manhattan':
            distances = [manhattan_distance(x, x_train) for x_train in self.X_train]
        else:
            raise ValueError("Unsupported metric")
            
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get labels of k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return most common label (majority voting)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
        
    def predict(self, X):
        """Predict labels for all test samples."""
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])


# ===================================================================================
# SECTION 2: Demonstrating KNN from Scratch vs Scikit-Learn
# ===================================================================================

print("=" * 80)
print("DAY 43: K-NEAREST NEIGHBORS (KNN) DEMONSTRATION")
print("=" * 80)

# Generate synthetic binary classification dataset
X, y = make_classification(
    n_samples=200, n_features=4, n_informative=3, n_redundant=0, 
    n_classes=2, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Dataset shape: {X.shape}")
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}\n")

# Scratch implementation
knn_scratch = KNNClassifierScratch(k=5, metric='euclidean')
knn_scratch.fit(X_train, y_train)
y_pred_scratch = knn_scratch.predict(X_test)
acc_scratch = accuracy_score(y_test, y_pred_scratch)

print(f"1. KNN from Scratch Accuracy (k=5, Euclidean): {acc_scratch * 100:.2f}%")

# Scikit-learn implementation
knn_sklearn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_sklearn.fit(X_train, y_train)
y_pred_sklearn = knn_sklearn.predict(X_test)
acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

print(f"2. Sklearn KNeighborsClassifier Accuracy (k=5, Euclidean): {acc_sklearn * 100:.2f}%\n")


# ===================================================================================
# SECTION 3: Distance Metrics Comparison
# ===================================================================================

print("--- Distance Metrics Comparison ---")
for metric in ['euclidean', 'manhattan', 'minkowski']:
    clf = KNeighborsClassifier(n_neighbors=5, metric=metric)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f"Metric: {metric:<10} | Accuracy: {score * 100:.2f}%")


# ===================================================================================
# SECTION 4: Optimal K Selection (Elbow Method / Grid Search)
# ===================================================================================

print("\n--- Optimal K Selection using 5-Fold Cross-Validation ---")
k_values = list(range(1, 21, 2))
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    mean_score = np.mean(scores)
    cv_scores.append(mean_score)
    print(f"k = {k:2d} | 5-Fold CV Accuracy: {mean_score * 100:.2f}%")

best_k = k_values[np.argmax(cv_scores)]
print(f"\nBest K parameter found: k = {best_k} with CV Accuracy: {max(cv_scores) * 100:.2f}%")


# ===================================================================================
# SECTION 5: KNN Regressor Example
# ===================================================================================

print("\n--- KNN Regression Example ---")
X_reg, y_reg = make_regression(n_samples=150, n_features=3, noise=15.0, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

knn_reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn_reg.fit(X_train_r, y_train_r)
y_pred_r = knn_reg.predict(X_test_r)

rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
r2 = knn_reg.score(X_test_r, y_test_r)

print(f"KNN Regressor RMSE: {rmse:.4f}")
print(f"KNN Regressor R² Score: {r2:.4f}")
print("=" * 80)
