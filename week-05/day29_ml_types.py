"""
================================================================================
Day 29: What is Machine Learning and Its Types
================================================================================
Course: GeeksforGeeks Machine Learning (Weeks 5-6)
Topic: Introduction to Machine Learning & Paradigm Classification

Description:
  This script demonstrates the three primary paradigms of Machine Learning:
  1. Supervised Learning (Classification & Regression)
  2. Unsupervised Learning (Clustering with K-Means & PCA)
  3. Reinforcement Learning (Q-Learning on a simple Grid/Line Environment)

Author: ML Course Instructor / Student
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification, make_blobs, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score


# ==============================================================================
# 1. SUPERVISED LEARNING DEMO
# ==============================================================================
def demo_supervised_learning():
    print("\n" + "="*70)
    print("1. SUPERVISED LEARNING DEMONSTRATION")
    print("="*70)
    print("Supervised learning uses labeled training data (X: features, y: targets).")
    
    # --- Classification ---
    print("\n--- A. Classification (Predicting Discrete Labels) ---")
    X_cls, y_cls = make_classification(
        n_samples=200, n_features=4, n_classes=2, random_state=42
    )
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cls, y_cls, test_size=0.25, random_state=42
    )
    
    clf = LogisticRegression()
    clf.fit(X_train_c, y_train_c)
    preds_c = clf.predict(X_test_c)
    acc = accuracy_score(y_test_c, preds_c)
    print(f"Classification Task: Binary Spam/Ham prediction")
    print(f"Training samples: {X_train_c.shape[0]}, Test samples: {X_test_c.shape[0]}")
    print(f"Test Accuracy: {acc * 100:.2f}%")
    
    # --- Regression ---
    print("\n--- B. Regression (Predicting Continuous Values) ---")
    X_reg, y_reg = make_regression(
        n_samples=200, n_features=2, noise=15.0, random_state=42
    )
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.25, random_state=42
    )
    
    reg = LinearRegression()
    reg.fit(X_train_r, y_train_r)
    preds_r = reg.predict(X_test_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, preds_r))
    print(f"Regression Task: House price prediction")
    print(f"Learned Coefficients: {reg.coef_}, Intercept: {reg.intercept_:.2f}")
    print(f"Test RMSE: {rmse:.2f}")
    
    return X_cls, y_cls, X_reg, y_reg


# ==============================================================================
# 2. UNSUPERVISED LEARNING DEMO
# ==============================================================================
def demo_unsupervised_learning():
    print("\n" + "="*70)
    print("2. UNSUPERVISED LEARNING DEMONSTRATION")
    print("="*70)
    print("Unsupervised learning discovers hidden patterns in unlabeled data (X only).")
    
    # --- Clustering (K-Means) ---
    print("\n--- A. Clustering (K-Means) ---")
    X_blobs, _ = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_blobs)
    sil_score = silhouette_score(X_blobs, cluster_labels)
    
    print(f"Discovered {len(np.unique(cluster_labels))} clusters in unlabeled data.")
    print(f"Cluster Centers:\n{kmeans.cluster_centers_}")
    print(f"Silhouette Score (Cluster Quality): {sil_score:.3f}")
    
    # --- Dimensionality Reduction (PCA) ---
    print("\n--- B. Dimensionality Reduction (PCA) ---")
    X_high_dim, _ = make_blobs(n_samples=100, centers=2, n_features=10, random_state=42)
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_high_dim)
    
    print(f"Original feature dimensions: {X_high_dim.shape[1]}")
    print(f"Reduced feature dimensions: {X_reduced.shape[1]}")
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Total Variance Retained: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
    
    return X_blobs, cluster_labels, kmeans.cluster_centers_


# ==============================================================================
# 3. REINFORCEMENT LEARNING DEMO
# ==============================================================================
def demo_reinforcement_learning():
    print("\n" + "="*70)
    print("3. REINFORCEMENT LEARNING DEMONSTRATION")
    print("="*70)
    print("Reinforcement learning trains an Agent to take Actions in an Environment to maximize Reward.")
    
    # Simple 1D Line Environment: 5 states [0, 1, 2, 3, 4]
    # State 0 is left wall, State 4 is Goal (+10 reward)
    num_states = 5
    num_actions = 2  # 0: Move Left, 1: Move Right
    q_table = np.zeros((num_states, num_actions))
    
    alpha = 0.1    # Learning rate
    gamma = 0.9    # Discount factor
    epsilon = 0.2  # Exploration rate
    episodes = 200
    
    print(f"Environment: 1D Line (States 0 to 4). Goal is State 4.")
    print(f"Training Q-Learning Agent over {episodes} episodes...")
    
    for episode in range(episodes):
        state = 0  # Start state
        done = False
        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.choice(num_actions)
            else:
                action = np.argmax(q_table[state])
            
            # Step transition logic
            if action == 1:  # Move Right
                next_state = min(state + 1, num_states - 1)
            else:           # Move Left
                next_state = max(state - 1, 0)
            
            # Reward scheme
            if next_state == 4:
                reward = 10.0
                done = True
            else:
                reward = -0.1  # Step penalty
            
            # Q-learning update equation
            best_next_q = np.max(q_table[next_state])
            q_table[state, action] += alpha * (reward + gamma * best_next_q - q_table[state, action])
            
            state = next_state
            
    print("\nLearned Q-Table (State x Action [Left, Right]):")
    print("State | Q(Left)  | Q(Right) | Best Action")
    print("-" * 42)
    actions_map = {0: "Left", 1: "Right"}
    for s in range(num_states):
        best_a = actions_map[np.argmax(q_table[s])]
        print(f"  {s}   | {q_table[s,0]:8.2f} | {q_table[s,1]:8.2f} | {best_a}")


# ==============================================================================
# 4. VISUALIZATION
# ==============================================================================
def visualize_ml_types(X_cls, y_cls, X_blobs, cluster_labels, centers):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Supervised Classification
    axes[0].scatter(X_cls[:, 0], X_cls[:, 1], c=y_cls, cmap='coolwarm', edgecolors='k')
    axes[0].set_title("Supervised Learning (Classification with Labels)")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Plot 2: Unsupervised Clustering
    axes[1].scatter(X_blobs[:, 0], X_blobs[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    axes[1].scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')
    axes[1].set_title("Unsupervised Learning (K-Means Clustering)")
    axes[1].set_xlabel("Feature 1")
    axes[1].set_ylabel("Feature 2")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("ml_types_summary.png")
    print("\nVisualization saved to 'ml_types_summary.png'.")
    # plt.show()  # Uncomment when running in GUI interactive environment


if __name__ == "__main__":
    print("="*70)
    print("  GEEKSFORGEEKS ML COURSE - DAY 29: ML TYPES & PARADIGMS")
    print("="*70)
    
    X_c, y_c, X_r, y_r = demo_supervised_learning()
    X_b, labels, centroids = demo_unsupervised_learning()
    demo_reinforcement_learning()
    visualize_ml_types(X_c, y_c, X_b, labels, centroids)
    
    print("\n" + "="*70)
    print("Day 29 Execution Completed Successfully!")
    print("="*70)
