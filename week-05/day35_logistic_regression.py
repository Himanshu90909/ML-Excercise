"""
================================================================================
Day 35: Logistic Regression for Binary Classification
================================================================================
Course: GeeksforGeeks Machine Learning (Weeks 5-6)
Topic: Sigmoid Function, Log-Loss & Decision Boundary Implementation

Description:
  This script implements Logistic Regression from scratch and compares it with
  Scikit-Learn's LogisticRegression:
  1. Mathematical Sigmoid Activation: g(z) = 1 / (1 + e^-z)
  2. Log Loss (Binary Cross-Entropy) Cost Function
  3. Custom Vectorized Gradient Descent for Binary Classification
  4. Decision Boundary Linear Equation derivation
  5. Performance Evaluation & Visualization of Decision Boundary

Author: ML Course Instructor / Student
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ==============================================================================
# LOGISTIC REGRESSION FROM SCRATCH
# ==============================================================================
class CustomLogisticRegression:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    @staticmethod
    def _sigmoid(z):
        """Numerically stable sigmoid function"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def _log_loss(self, y_true, y_prob):
        """Binary Cross-Entropy / Log Loss"""
        eps = 1e-15  # Avoid log(0)
        y_prob = np.clip(y_prob, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(self.epochs):
            # Linear combination: z = X*w + b
            linear_model = np.dot(X, self.weights) + self.bias
            # Sigmoid activation: y_hat = g(z)
            y_prob = self._sigmoid(linear_model)

            # Compute Loss
            loss = self._log_loss(y, y_prob)
            self.loss_history.append(loss)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_prob - y))
            db = (1 / n_samples) * np.sum(y_prob - y)

            # Parameter updates
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if (epoch + 1) % (self.epochs // 5) == 0:
                print(f"Epoch {epoch+1:4d}/{self.epochs} | Log-Loss: {loss:.4f}")

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ==============================================================================
# MAIN DEMO
# ==============================================================================
def main():
    print("="*70)
    print("  GEEKSFORGEEKS ML COURSE - DAY 35: LOGISTIC REGRESSION")
    print("="*70)

    # --- Generate Synthetic Binary Classification Dataset ---
    X, y = make_blobs(n_samples=300, centers=2, n_features=2, cluster_std=1.5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    print(f"\nGenerated dataset with {X.shape[0]} samples and 2 features.")
    print(f"Class distribution: {np.bincount(y)}")

    # --- 1. Custom Logistic Regression ---
    print("\n" + "="*50)
    print("1. Custom Logistic Regression (From Scratch)")
    print("="*50)
    scratch_model = CustomLogisticRegression(learning_rate=0.1, epochs=1000)
    scratch_model.fit(X_train, y_train)

    y_pred_scratch = scratch_model.predict(X_test)
    acc_scratch = accuracy_score(y_test, y_pred_scratch)

    print(f"\nScratch Weights: {scratch_model.weights}")
    print(f"Scratch Bias:    {scratch_model.bias:.4f}")
    print(f"Scratch Accuracy: {acc_scratch * 100:.2f}%")
    print("\nConfusion Matrix (Scratch):")
    print(confusion_matrix(y_test, y_pred_scratch))

    # --- 2. Scikit-Learn Logistic Regression ---
    print("\n" + "="*50)
    print("2. Scikit-Learn Logistic Regression")
    print("="*50)
    sk_model = LogisticRegression()
    sk_model.fit(X_train, y_train)

    y_pred_sk = sk_model.predict(X_test)
    acc_sk = accuracy_score(y_test, y_pred_sk)

    print(f"Sklearn Weights: {sk_model.coef_[0]}")
    print(f"Sklearn Bias:    {sk_model.intercept_[0]:.4f}")
    print(f"Sklearn Accuracy: {acc_sk * 100:.2f}%")

    print("\nClassification Report (Sklearn):")
    print(classification_report(y_test, y_pred_sk))

    # --- 3. Visualizations ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Sigmoid Activation Function Curve
    z_vals = np.linspace(-10, 10, 200)
    sigmoid_vals = CustomLogisticRegression._sigmoid(z_vals)
    ax1.plot(z_vals, sigmoid_vals, 'b-', linewidth=2.5, label=r'$\sigma(z) = \frac{1}{1 + e^{-z}}$')
    ax1.axhline(0.5, color='r', linestyle='--', label='Decision Threshold = 0.5')
    ax1.axvline(0, color='gray', linestyle=':')
    ax1.set_title("Sigmoid Activation Function")
    ax1.set_xlabel("z = w^T X + b")
    ax1.set_ylabel("Probability P(Y=1|X)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot 2: 2D Decision Boundary
    ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', edgecolors='k', alpha=0.8)
    
    # Boundary equation: w1*x1 + w2*x2 + b = 0 => x2 = -(w1*x1 + b) / w2
    w1, w2 = scratch_model.weights
    b = scratch_model.bias
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1_vals = np.linspace(x1_min, x1_max, 100)
    x2_vals = -(w1 * x1_vals + b) / w2

    ax2.plot(x1_vals, x2_vals, 'k--', linewidth=2, label='Scratch Decision Boundary')
    ax2.set_title("2D Decision Boundary Line")
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("logistic_regression_boundary.png")
    print("\nPlots saved to 'logistic_regression_boundary.png'.")


if __name__ == "__main__":
    main()
