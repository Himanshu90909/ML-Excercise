"""
Day 65: Independent Component Analysis (ICA) & Isomap
Topic: Blind Source Separation (FastICA), Non-Linear Manifold Learning (Isomap)
Description:
    This exercise demonstrates advanced feature decomposition and manifold unrolling:
    - FastICA for unmixing linearly combined independent signals (Cocktail Party Problem)
    - Isomap (Isometric Feature Mapping) preserving geodesic distances on non-linear manifolds
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
from sklearn.manifold import Isomap
from sklearn.datasets import make_swiss_roll

plt.switch_backend('Agg')

def main():
    print("=" * 70)
    print("DAY 65: ICA (Signal Unmixing) & Isomap Manifold Learning")
    print("=" * 70)

    # -------------------------------------------------------------
    # PART 1: FastICA (Independent Component Analysis)
    # -------------------------------------------------------------
    print("\n>>> PART 1: FastICA - Blind Source Signal Separation")
    n_samples = 1000
    time = np.linspace(0, 8, n_samples)

    # Generate 3 independent signals
    s1 = np.sin(2 * time)                           # Signal 1: Sine wave
    s2 = np.sign(np.sin(3 * time))                  # Signal 2: Square wave
    np.random.seed(42)
    s3 = np.random.normal(size=n_samples)            # Signal 3: Gaussian noise

    S = np.c_[s1, s2, s3]  # Real independent sources shape (1000, 3)

    # Mixing matrix A
    A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1, 2.0]])
    X_mixed = np.dot(S, A.T)  # Mixed signals

    # Apply FastICA
    ica = FastICA(n_components=3, random_state=42)
    S_recovered = ica.fit_transform(X_mixed)

    print(f"Recovered {S_recovered.shape[1]} independent components from 3 mixed signal observations.")

    # -------------------------------------------------------------
    # PART 2: Isomap (Isometric Feature Mapping on Swiss Roll)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print(">>> PART 2: Isomap Manifold Learning on 3D Swiss Roll")

    X_roll, color = make_swiss_roll(n_samples=800, noise=0.05, random_state=42)

    # PCA vs Isomap projection comparison
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_roll)

    isomap = Isomap(n_neighbors=10, n_components=2)
    X_isomap = isomap.fit_transform(X_roll)

    print(f"Swiss Roll unrolled from 3D {X_roll.shape} to 2D using Isomap.")

    # Plot comparisons
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=color, cmap='plasma', s=20)
    axes[0].set_title('PCA Projection (Linear - Overlaps Fold)')

    axes[1].scatter(X_isomap[:, 0], X_isomap[:, 1], c=color, cmap='plasma', s=20)
    axes[1].set_title('Isomap Projection (Geodesic Preserving - Unrolled)')

    plt.tight_layout()
    output_png = "ml-exercise/week-10/day65_ica_isomap.png"
    plt.savefig(output_png)
    plt.close()
    print(f"Isomap manifold comparison plot saved to '{output_png}'.")

if __name__ == '__main__':
    main()
