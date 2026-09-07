"""
Day 60: Advanced Clustering Algorithms
Topic: Mean Shift, Spectral Clustering, Affinity Propagation
"""
import numpy as np
from sklearn.cluster import MeanShift, SpectralClustering, AffinityPropagation, estimate_bandwidth
from sklearn.datasets import make_moons, make_blobs
from sklearn.metrics import silhouette_score

# --- Mean Shift ---
X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)
bandwidth = estimate_bandwidth(X, quantile=0.2)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms_labels = ms.fit_predict(X)
print(f"Mean Shift:")
print(f"  Bandwidth: {bandwidth:.3f}")
print(f"  Clusters found: {len(np.unique(ms_labels))}")
print(f"  Silhouette: {silhouette_score(X, ms_labels):.4f}")

# --- Spectral Clustering ---
X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)
sc = SpectralClustering(n_clusters=2, affinity='rbf', gamma=1.0, random_state=42)
sc_labels = sc.fit_predict(X_moons)
print(f"\nSpectral Clustering (Moons):")
print(f"  Clusters: {len(np.unique(sc_labels))}")
print(f"  Silhouette: {silhouette_score(X_moons, sc_labels):.4f}")

# --- Affinity Propagation ---
ap = AffinityPropagation(damping=0.7, random_state=42, max_iter=200)
ap_labels = ap.fit_predict(X)
print(f"\nAffinity Propagation:")
print(f"  Clusters found: {len(np.unique(ap_labels))}")
print(f"  Silhouette: {silhouette_score(X, ap_labels):.4f}")
print(f"  Exemplars (cluster centers indices): {ap.cluster_centers_indices_}")

# --- Comparison ---
from sklearn.cluster import KMeans, DBSCAN
algorithms = {
    'KMeans': KMeans(n_clusters=3, random_state=42, n_init=10),
    'MeanShift': MeanShift(bandwidth=bandwidth),
    'Spectral': SpectralClustering(n_clusters=3, random_state=42, affinity='rbf'),
    'DBSCAN': DBSCAN(eps=0.8, min_samples=5),
}

print("\nAlgorithm Comparison on Blobs:")
print("-" * 50)
for name, algo in algorithms.items():
    labels = algo.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    sil = silhouette_score(X, labels) if n_clusters > 1 else 0
    print(f"  {name:12s} -> clusters: {n_clusters}, silhouette: {sil:.4f}")

print("\nKey Concepts:")
print("- Mean Shift: no need to specify k, finds clusters by density modes")
print("- Spectral Clustering: uses graph Laplacian eigenvalues for non-convex clusters")
print("- Affinity Propagation: finds exemplars, no need to set k beforehand")
