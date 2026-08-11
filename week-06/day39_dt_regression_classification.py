"""
================================================================================
Day 39: Decision Trees: Regression vs Classification & Overfitting Controls
================================================================================
Course: GeeksforGeeks Machine Learning (Weeks 5-6)
Topic: Classification Trees vs Regression Trees & Pruning Hyperparameters

Description:
  This script provides a detailed comparative study of Decision Trees:
  1. Classification Trees (Discrete predictions, Gini/Entropy criteria)
  2. Regression Trees (Continuous predictions, MSE/MAE criteria, Step-wise fits)
  3. Overfitting Analysis & Regularization via Hyperparameters:
     - max_depth, min_samples_split, min_samples_leaf, ccp_alpha (Pruning)
  4. Visualizing step-wise regression decision boundaries vs true sine wave

Author: ML Course Instructor / Student
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


# ==============================================================================
# 1. DECISION TREE CLASSIFICATION DEMO
# ==============================================================================
def demo_classification_tree():
    print("\n" + "="*50)
    print("1. Decision Tree Classification")
    print("="*50)
    
    X, y = make_classification(
        n_samples=300, n_features=4, n_informative=3, n_redundant=0, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Fully grown (overfitted) tree
    full_tree = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=42)
    full_tree.fit(X_train, y_train)

    # Regularized (pruned) tree
    pruned_tree = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_leaf=5, random_state=42)
    pruned_tree.fit(X_train, y_train)

    print("Fully Grown Tree:")
    print(f"  Depth: {full_tree.get_depth()}, Leaves: {full_tree.get_n_leaves()}")
    print(f"  Train Acc: {accuracy_score(y_train, full_tree.predict(X_train)):.4f}")
    print(f"  Test Acc:  {accuracy_score(y_test, full_tree.predict(X_test)):.4f}")

    print("\nRegularized Tree (max_depth=3, min_samples_leaf=5):")
    print(f"  Depth: {pruned_tree.get_depth()}, Leaves: {pruned_tree.get_n_leaves()}")
    print(f"  Train Acc: {accuracy_score(y_train, pruned_tree.predict(X_train)):.4f}")
    print(f"  Test Acc:  {accuracy_score(y_test, pruned_tree.predict(X_test)):.4f}")


# ==============================================================================
# 2. DECISION TREE REGRESSION DEMO
# ==============================================================================
def demo_regression_tree():
    print("\n" + "="*50)
    print("2. Decision Tree Regression")
    print("="*50)

    np.random.seed(42)
    X = np.sort(5 * np.random.rand(120, 1), axis=0)
    # Sine wave function with noise
    y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Fit Regression Trees at varying depths
    regr_depth_2 = DecisionTreeRegressor(max_depth=2, random_state=42)
    regr_depth_5 = DecisionTreeRegressor(max_depth=5, random_state=42)
    regr_depth_deep = DecisionTreeRegressor(max_depth=15, random_state=42)  # Overfitted

    regr_depth_2.fit(X_train, y_train)
    regr_depth_5.fit(X_train, y_train)
    regr_depth_deep.fit(X_train, y_train)

    models = {'Depth 2 (Underfit)': regr_depth_2, 'Depth 5 (Balanced)': regr_depth_5, 'Depth 15 (Overfit)': regr_depth_deep}
    for name, model in models.items():
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"  {name:20s} | Test MSE: {mse:.4f} | R^2 Score: {r2:.4f}")

    return X, y, regr_depth_2, regr_depth_5, regr_depth_deep


# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================
def main():
    print("="*70)
    print("  GEEKSFORGEEKS ML COURSE - DAY 39: DT REGRESSION VS CLASSIFICATION")
    print("="*70)

    demo_classification_tree()
    X_reg, y_reg, r_d2, r_d5, r_deep = demo_regression_tree()

    # --- Plotting Regression Step Functions ---
    X_grid = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_grid_true = np.sin(X_grid).ravel()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(X_reg, y_reg, s=20, edgecolor="black", c="darkorange", label="Noisy Data")
    ax.plot(X_grid, y_grid_true, color="black", linestyle="--", linewidth=1.5, label="True Sine Function")
    ax.plot(X_grid, r_d2.predict(X_grid), color="cornflowerblue", label="Regr Tree (Depth=2)", linewidth=2)
    ax.plot(X_grid, r_d5.predict(X_grid), color="yellowgreen", label="Regr Tree (Depth=5)", linewidth=2)
    ax.plot(X_grid, r_deep.predict(X_grid), color="red", label="Regr Tree (Depth=15 Overfit)", linewidth=1, alpha=0.7)

    ax.set_title("Decision Tree Regression: Step-Wise Approximations vs Depth")
    ax.set_xlabel("Feature X")
    ax.set_ylabel("Target y")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("dt_regression_vs_classification.png")
    print("\nRegression step-function plot saved to 'dt_regression_vs_classification.png'.")


if __name__ == "__main__":
    main()
