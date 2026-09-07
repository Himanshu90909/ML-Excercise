"""
Day 61: Hierarchical Clustering
Topic: Agglomerative clustering, dendrograms, linkage methods
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster

X, y = make_blobs(n_samples=200, centers=4, n_features=2, random_state=42)

# --- Linkage methods comparison ---
linkage_methods = ['ward', 'complete', 'average', 'single']

print("Linkage Methods Comparison:")
print("-" * 55)
for method in linkage_methods:
    ac = AgglomerativeClustering(n_clusters=4, linkage=method)
    labels = ac.fit_predict(X)
    sil = silhouette_score(X, labels)
    print(f"  {method:10s} -> silhouette: {sil:.4f}")

# --- Dendrogram computation ---
Z = linkage(X, method='ward')
print(f"\nLinkage matrix shape: {Z.shape}")
print("First 5 merges:")
for i, row in enumerate(Z[:5]):
    print(f"  Merge {i+1}: clusters {int(row[0])} & {int(row[1])}, distance: {row[2]:.3f}")

# --- Cut dendrogram at different thresholds ---
print("\nCutting dendrogram at different distances:")
print("-" * 50)
for threshold in [5, 10, 15, 20, 30]:
    labels = fcluster(Z, t=threshold, criterion='distance')
    print(f"  threshold={threshold:5.1f} -> {len(set(labels))} clusters")

# --- Optimal number of clusters ---
print("\nSilhouette by number of clusters:")
for n in range(2, 8):
    ac = AgglomerativeClustering(n_clusters=n, linkage='ward')
    labels = ac.fit_predict(X)
    sil = silhouette_score(X, labels)
    print(f"  k={n} -> silhouette: {sil:.4f}")

print("\nKey Concepts:")
print("- Agglomerative (bottom-up): starts with each point as its own cluster")
print("- Ward linkage: minimizes variance within clusters")
print("- Complete linkage: uses maximum distance between clusters")
print("- Average linkage: uses average distance between all pairs")
print("- Single linkage: uses minimum distance (chaining effect)")
print("- Dendrogram visually shows the merging hierarchy")
