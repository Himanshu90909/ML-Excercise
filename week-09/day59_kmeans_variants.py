"""
Day 59: K-Means Variants: K-Means++, K-Modes, and Fuzzy C-Means
Topic: Advanced Clustering Variants, Smart Initialization, Categorical & Soft Clustering
Description:
    This exercise explores variants beyond standard K-Means:
    1. K-Means++ initialization mechanism vs random initialization
    2. Fuzzy C-Means (Soft Clustering) computing membership probabilities
    3. Categorical clustering concepts (K-Modes dissimilarity)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

plt.switch_backend('Agg')

def fuzzy_c_means(X, c=3, m=2.0, max_iter=100, tol=1e-4):
    """Simple Fuzzy C-Means (FCM) soft clustering algorithm implementation."""
    np.random.seed(42)
    n_samples, n_features = X.shape

    # Randomly initialize membership matrix U (n_samples x c)
    U = np.random.dirichlet(np.ones(c), size=n_samples)

    for iteration in range(max_iter):
        # Update centroids: v_j = sum(u_ij^m * x_i) / sum(u_ij^m)
        um = U ** m
        centroids = (um.T @ X) / um.sum(axis=0)[:, np.newaxis]

        # Compute distance matrix
        dist = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        dist = np.fmax(dist, 1e-10)  # Avoid division by zero

        # Update membership matrix U
        inv_dist = (1.0 / dist) ** (2.0 / (m - 1))
        new_U = inv_dist / inv_dist.sum(axis=1, keepdims=True)

        if np.linalg.norm(new_U - U) < tol:
            break
        U = new_U

    return centroids, U

def main():
    print("=" * 70)
    print("DAY 59: K-Means Variants - K-Means++, Fuzzy C-Means & K-Modes")
    print("=" * 70)

    # 1. Compare K-Means++ vs Random Centroid Initialization
    X, _ = make_blobs(n_samples=600, centers=4, cluster_std=1.2, random_state=42)

    km_random = KMeans(n_clusters=4, init='random', n_init=1, max_iter=300, random_state=10)
    km_random.fit(X)

    km_pp = KMeans(n_clusters=4, init='k-means++', n_init=1, max_iter=300, random_state=10)
    km_pp.fit(X)

    print("--- 1. K-Means++ vs Random Initialization ---")
    print(f"Random Init  -> Final Inertia: {km_random.inertia_:.2f} | Iterations: {km_random.n_iter_}")
    print(f"K-Means++    -> Final Inertia: {km_pp.inertia_:.2f} | Iterations: {km_pp.n_iter_}")

    # 2. Fuzzy C-Means (Soft Clustering)
    print("\n--- 2. Fuzzy C-Means (Soft Clustering) ---")
    centroids_fcm, U_fcm = fuzzy_c_means(X, c=4, m=2.0)
    print(f"FCM Centroids Computed:\n{centroids_fcm.round(3)}")
    print("\nSample Point Membership Probabilities across 4 Clusters (First 5 points):")
    print(U_fcm[:5].round(3))

    # 3. K-Modes Concept (Categorical Clustering)
    print("\n--- 3. Categorical Clustering Concept (K-Modes) ---")
    cat_df = pd.DataFrame({
        'Department': ['IT', 'HR', 'IT', 'Finance', 'HR', 'Finance'],
        'Role': ['Dev', 'Recruiter', 'Dev', 'Analyst', 'Recruiter', 'Manager'],
        'Location': ['NYC', 'NY', 'NYC', 'LN', 'NY', 'LN']
    })
    print("Categorical Transactions DataFrame:")
    print(cat_df)

    def simple_matching_distance(row1, row2):
        """Simple matching dissimilarity measure for categorical attributes."""
        return sum(c1 != c2 for c1, c2 in zip(row1, row2))

    dist_sample = simple_matching_distance(cat_df.iloc[0], cat_df.iloc[2])
    print(f"Matching dissimilarity between Row 0 and Row 2: {dist_sample} differences.")

    # Visualization
    plt.figure(figsize=(8, 6))
    hard_labels = np.argmax(U_fcm, axis=1)
    plt.scatter(X[:, 0], X[:, 1], c=hard_labels, cmap='tab10', alpha=0.7, edgecolors='k')
    plt.scatter(centroids_fcm[:, 0], centroids_fcm[:, 1], c='red', s=200, marker='X', label='FCM Centroids')
    plt.title('Fuzzy C-Means Soft Clustering (Hard Max-Membership View)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

    output_png = "ml-exercise/week-09/day59_fcm_clustering.png"
    plt.savefig(output_png)
    plt.close()
    print(f"\nFuzzy C-Means plot saved to '{output_png}'.")

if __name__ == '__main__':
    main()
