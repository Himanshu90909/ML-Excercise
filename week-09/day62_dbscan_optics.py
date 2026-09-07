"""
Day 62: DBSCAN and OPTICS
Topic: Density-based clustering, eps, min_samples
"""
import numpy as np
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.datasets import make_moons, make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# --- DBSCAN on moons ---
X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
X_moons = StandardScaler().fit_transform(X_moons)

dbscan = DBSCAN(eps=0.3, min_samples=5)
db_labels = dbscan.fit_predict(X_moons)
n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = sum(db_labels == -1)
print(f"DBSCAN on Moons:")
print(f"  Clusters: {n_clusters}, Noise points: {n_noise}")
print(f"  Silhouette: {silhouette_score(X_moons, db_labels):.4f}")

# --- Effect of eps ---
print("\nEffect of eps on DBSCAN:")
print("-" * 50)
for eps in [0.1, 0.2, 0.3, 0.5, 1.0]:
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X_moons)
    n_cl = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = sum(labels == -1)
    print(f"  eps={eps:.1f} -> clusters: {n_cl}, noise: {n_noise}")

# --- Effect of min_samples ---
print("\nEffect of min_samples:")
for ms in [3, 5, 10, 15, 20]:
    db = DBSCAN(eps=0.3, min_samples=ms)
    labels = db.fit_predict(X_moons)
    n_cl = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  min_samples={ms:2d} -> clusters: {n_cl}")

# --- OPTICS ---
X_blobs, _ = make_blobs(n_samples=300, centers=3, cluster_std=[0.5, 1.0, 2.0], random_state=42)
optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1)
op_labels = optics.fit_predict(X_blobs)
n_op = len(set(op_labels)) - (1 if -1 in op_labels else 0)
print(f"\nOPTICS:")
print(f"  Clusters: {n_op}")
print(f"  Reachability scores (first 10): {optics.reachability_[optics.ordering_[:10]].round(3)}")

# --- Comparison ---
print("\nDBSCAN vs OPTICS vs KMeans:")
from sklearn.cluster import KMeans
for name, algo in [('DBSCAN', DBSCAN(eps=0.3, min_samples=5)),
                   ('OPTICS', OPTICS(min_samples=5, xi=0.05)),
                   ('KMeans', KMeans(n_clusters=2, random_state=42, n_init=10))]:
    labels = algo.fit_predict(X_moons)
    n_cl = len(set(labels)) - (1 if -1 in labels else 0)
    sil = silhouette_score(X_moons, labels) if n_cl > 1 else 0
    print(f"  {name:8s} -> clusters: {n_cl}, silhouette: {sil:.4f}")

print("\nKey Concepts:")
print("- DBSCAN: density-based, finds core points with min_samples neighbors within eps")
print("- No need to specify k, handles non-spherical clusters")
print("- OPTICS: variable density clustering, creates reachability plot")
print("- OPTICS doesn't need eps parameter, adapts to local density")
