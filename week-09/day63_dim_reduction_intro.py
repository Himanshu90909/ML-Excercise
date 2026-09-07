"""
Day 63: Introduction to Dimensionality Reduction
Topic: PCA basics, variance explained, scree plot
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# --- Generate high-dimensional data ---
X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                           n_redundant=5, n_classes=3, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# --- PCA ---
pca = PCA()
pca.fit(X_scaled)
print(f"Original dimensions: {X.shape[1]}")
print(f"\nExplained Variance Ratio:")
for i, var in enumerate(pca.explained_variance_ratio_[:10]):
    cumsum = pca.explained_variance_ratio_[:i+1].sum()
    print(f"  PC{i+1:2d}: {var:.4f} ({var*100:.1f}%) -> Cumulative: {cumsum:.4f} ({cumsum*100:.1f}%)")

# --- Scree plot data ---
print(f"\nEigenvalues (first 10):")
for i, eig in enumerate(pca.explained_variance_[:10]):
    print(f"  PC{i+1:2d}: {eig:.4f}")

# --- Number of components for 95% variance ---
pca_95 = PCA(n_components=0.95)
X_reduced = pca_95.fit_transform(X_scaled)
print(f"\nComponents for 95% variance: {pca_95.n_components_}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Total variance retained: {pca_95.explained_variance_ratio_.sum():.4f}")

# --- PCA for 2D visualization ---
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)
print(f"\n2D PCA:")
print(f"  Variance explained: {pca_2d.explained_variance_ratio_.sum():.4f}")
print(f"  First 5 transformed samples:\n{X_2d[:5].round(3)}")

# --- Feature loadings ---
print(f"\nFeature loadings on PC1 (top 5):")
loadings = pca.components_[0]
sorted_idx = np.argsort(np.abs(loadings))[::-1]
for i in sorted_idx[:5]:
    print(f"  Feature {i:2d}: {loadings[i]:.4f}")

# --- PCA before classification ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100, random_state=42)
cv_full = cross_val_score(rf, X_scaled, y, cv=5).mean()
cv_pca = cross_val_score(rf, X_reduced, y, cv=5).mean()
print(f"\nClassification CV Accuracy:")
print(f"  Full features (20): {cv_full:.4f}")
print(f"  PCA reduced ({pca_95.n_components_}): {cv_pca:.4f}")

print("\nKey Concepts:")
print("- PCA finds orthogonal axes of maximum variance")
print("- Scree plot shows variance explained by each component")
print("- Kaiser rule: keep components with eigenvalue > 1")
print("- 95% variance threshold is a common stopping criterion")
