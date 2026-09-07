"""
Day 66: LLE and Manifold Learning Techniques
Topic: Locally Linear Embedding, MDS
"""
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding, MDS
from sklearn.datasets import make_swiss_roll, make_s_curve
from sklearn.decomposition import PCA

# --- Swiss Roll dataset ---
X_roll, t_roll = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)
print(f"Swiss Roll: {X_roll.shape}")

# --- LLE ---
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, method='standard', random_state=42)
X_lle = lle.fit_transform(X_roll)
print(f"\nLLE:")
print(f"  Reduced: {X_lle.shape}")
print(f"  Reconstruction error: {lle.reconstruction_error_:.6f}")

# --- Compare LLE methods ---
methods = ['standard', 'modified', 'hessian', 'ltsa']
print("\nLLE Method Comparison:")
for method in methods:
    try:
        l = LocallyLinearEmbedding(n_components=2, n_neighbors=12, method=method, random_state=42)
        X_ = l.fit_transform(X_roll[:500])
        print(f"  {method:10s} -> error: {l.reconstruction_error_:.6f}")
    except Exception as e:
        print(f"  {method:10s} -> error: {str(e)[:50]}")

# --- Effect of n_neighbors ---
print("\nEffect of n_neighbors:")
for n in [5, 10, 15, 20, 30]:
    l = LocallyLinearEmbedding(n_components=2, n_neighbors=n, random_state=42)
    X_ = l.fit_transform(X_roll[:500])
    print(f"  n_neighbors={n:2d} -> error: {l.reconstruction_error_:.6f}")

# --- MDS ---
mds = MDS(n_components=2, random_state=42, dissimilarity='euclidean')
X_mds = mds.fit_transform(X_roll[:500])
print(f"\nMDS:")
print(f"  Reduced: {X_mds.shape}")
print(f"  Stress: {mds.stress_:.4f}")

# --- S-curve dataset ---
X_s, t_s = make_s_curve(n_samples=800, noise=0.1, random_state=42)

print("\nTechnique Comparison on S-Curve:")
print("-" * 55)
for name, reducer in [
    ('PCA', PCA(n_components=2)),
    ('LLE', LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)),
    ('MDS', MDS(n_components=2, random_state=42)),
]:
    X_red = reducer.fit_transform(X_s)
    print(f"  {name:5s} -> shape: {X_red.shape}")

# --- Compare on classification ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
X_clf, y_clf = make_classification(n_samples=300, n_features=15, random_state=42)

for name, reducer in [('PCA', PCA(n_components=5)), ('LLE', LocallyLinearEmbedding(n_components=5, n_neighbors=10, random_state=42)), ('MDS', MDS(n_components=5, random_state=42))]:
    X_red = reducer.fit_transform(X_clf)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = cross_val_score(rf, X_red, y_clf, cv=5).mean()
    print(f"  {name:5s} + RF CV accuracy: {cv:.4f}")

print("\nKey Concepts:")
print("- LLE: preserves local neighborhoods by reconstructing each point from neighbors")
print("- Modified LLE: regularized for better stability")
print("- MDS: preserves pairwise distances in lower dimensions")
print("- Manifold learning: non-linear dimensionality reduction")
