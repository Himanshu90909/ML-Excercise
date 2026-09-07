"""
Day 64: Non-Linear Dimensionality Reduction: t-SNE & NMF
Topic: t-SNE (t-Distributed Stochastic Neighbor Embedding), NMF (Non-Negative Matrix Factorization)
Description:
    This exercise demonstrates non-linear dimensionality reduction and factorization:
    - t-SNE for 2D visual projection of high-dimensional data (impact of perplexity parameter)
    - NMF for non-negative feature extraction and topic modeling on non-negative matrices
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.decomposition import NMF

plt.switch_backend('Agg')

def main():
    print("=" * 70)
    print("DAY 64: Non-Linear Dimensionality Reduction - t-SNE & NMF")
    print("=" * 70)

    # -------------------------------------------------------------
    # PART 1: t-SNE (t-Distributed Stochastic Neighbor Embedding)
    # -------------------------------------------------------------
    print("\n>>> PART 1: t-SNE High-Dimensional Visualization")
    digits = load_digits()
    X_digits = digits.data
    y_digits = digits.target

    print(f"Digits Dataset: {X_digits.shape[0]} samples, {X_digits.shape[1]} pixel features.")

    perplexities = [5, 30, 50]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for idx, perp in enumerate(perplexities):
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
        X_tsne = tsne.fit_transform(X_digits[:500])  # First 500 samples for speed
        
        scatter = axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_digits[:500], cmap='tab10', s=15, alpha=0.8)
        axes[idx].set_title(f't-SNE (Perplexity = {perp})')
        axes[idx].set_xlabel('t-SNE Component 1')
        axes[idx].set_ylabel('t-SNE Component 2')

    plt.tight_layout()
    output_png = "ml-exercise/week-10/day64_tsne_perplexities.png"
    plt.savefig(output_png)
    plt.close()
    print(f"t-SNE perplexity plot saved to '{output_png}'.")

    # -------------------------------------------------------------
    # PART 2: Non-Negative Matrix Factorization (NMF)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print(">>> PART 2: Non-Negative Matrix Factorization (NMF)")

    # Synthetic non-negative document-term matrix
    np.random.seed(42)
    V = np.abs(np.random.normal(loc=5, scale=2, size=(100, 20)))  # 100 documents x 20 terms

    n_topics = 3
    nmf = NMF(n_components=n_topics, init='random', random_state=42, max_iter=500)
    W = nmf.fit_transform(V)  # Document-Topic matrix (100 x 3)
    H = nmf.components_       # Topic-Term matrix (3 x 20)

    print(f"Input Matrix V Shape: {V.shape}")
    print(f"Document-Topic Matrix W Shape: {W.shape}")
    print(f"Topic-Term Matrix H Shape: {H.shape}")
    print(f"NMF Reconstruction Error (Frobenius Norm): {nmf.reconstruction_err_:.4f}")

if __name__ == '__main__':
    main()
