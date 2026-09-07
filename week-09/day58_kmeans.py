"""
Day 58: K-Means Clustering
Topic: K-Means from scratch and sklearn, elbow method, silhouette score
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# --- K-Means from scratch ---
class KMeansScratch:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
    
    def fit(self, X):
        idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[idx].copy()
        for _ in range(self.max_iters):
            distances = np.array([[np.linalg.norm(x - c) for c in self.centroids] for x in X])
            labels = distances.argmin(axis=1)
            new_centroids = np.array([X[labels == k].mean(axis=0) if (labels == k).any() else self.centroids[k] for k in range(self.k)])
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
        self.labels_ = labels
        return self

X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)

km_scratch = KMeansScratch(k=4)
km_scratch.fit(X)
print(f"From-scratch K-Means inertia: {sum(np.min([[np.linalg.norm(x - c) for c in km_scratch.centroids] for x in X], axis=1)):.4f}")

# --- sklearn K-Means ---
km = KMeans(n_clusters=4, random_state=42, n_init=10)
km.fit(X)
print(f"sklearn K-Means inertia: {km.inertia_:.4f}")
print(f"Cluster sizes: {[sum(km.labels_ == i) for i in range(4)]}")

# --- Elbow Method ---
print("\nElbow Method:")
print("-" * 30)
for k in range(1, 11):
    km_k = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_k.fit(X)
    print(f"  k={k:2d} -> inertia: {km_k.inertia_:.2f}")

# --- Silhouette Score ---
print("\nSilhouette Scores:")
print("-" * 30)
for k in range(2, 11):
    km_k = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km_k.fit_predict(X)
    sil = silhouette_score(X, labels)
    print(f"  k={k:2d} -> silhouette: {sil:.4f}")

print("\nKey Concepts:")
print("- K-Means minimizes within-cluster sum of squares (inertia)")
print("- Elbow method: choose k where inertia decrease slows significantly")
print("- Silhouette score ranges from -1 to 1 (higher = better clusters)")
print("- K-Means assumes spherical clusters and is sensitive to initialization")
