"""
===================================================================================
GeeksforGeeks Machine Learning Course
Week 7: Machine Learning Algorithms
Day 44: Naive Bayes Classification

Topics Covered:
- Bayes Theorem mathematical foundations (P(A|B) = P(B|A)*P(A) / P(B))
- Gaussian Naive Bayes implemented from scratch
- Feature prior, mean, variance, Gaussian likelihood calculations
- Comparison with sklearn.naive_bayes.GaussianNB
===================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification

# Set seed
np.random.seed(42)

# ===================================================================================
# SECTION 1: Bayes Theorem Conceptual Helper
# ===================================================================================

def calculate_bayes_theorem(prior, likelihood, marginal_likelihood):
    """
    Computes posterior probability using Bayes' Theorem:
    Posterior = (Likelihood * Prior) / Marginal Likelihood
    """
    posterior = (likelihood * prior) / marginal_likelihood
    return posterior


# ===================================================================================
# SECTION 2: Gaussian Naive Bayes from Scratch
# ===================================================================================

class GaussianNBScratch:
    """Gaussian Naive Bayes Classifier implemented from scratch."""
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        
        # Initialize mean, variance, and priors for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)
        
        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0) + 1e-9  # Add small epsilon for stability
            self._priors[idx] = X_c.shape[0] / float(n_samples)
            
    def _calculate_gaussian_pdf(self, class_idx, x):
        """Gaussian probability density function."""
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
        
    def _predict_single(self, x):
        posteriors = []
        
        # Calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            likelihood = np.sum(np.log(self._calculate_gaussian_pdf(idx, x)))
            posterior = prior + likelihood
            posteriors.append(posterior)
            
        # Return class with highest posterior probability
        return self._classes[np.argmax(posteriors)]
        
    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])


# ===================================================================================
# SECTION 3: Execution and Benchmark
# ===================================================================================

print("=" * 80)
print("DAY 44: NAIVE BAYES CLASSIFICATION DEMONSTRATION")
print("=" * 80)

# Step 1: Conceptual Bayes Example (Medical Diagnosis)
print("\n--- Step 1: Bayes' Theorem Calculation Example ---")
prior_disease = 0.01          # P(Disease) = 1%
likelihood_pos = 0.95         # P(Positive Test | Disease) = 95%
false_pos_rate = 0.05         # P(Positive Test | No Disease) = 5%
prior_no_disease = 0.99       # P(No Disease) = 99%

# Marginal P(Positive Test)
p_positive = (likelihood_pos * prior_disease) + (false_pos_rate * prior_no_disease)

posterior_disease = calculate_bayes_theorem(prior_disease, likelihood_pos, p_positive)
print(f"P(Disease) = {prior_disease}")
print(f"P(Positive | Disease) = {likelihood_pos}")
print(f"P(Disease | Positive Test) = {posterior_disease * 100:.2f}%\n")

# Step 2: Model Training on Synthetic Continuous Dataset
print("--- Step 2: Gaussian Naive Bayes Model Comparison ---")
X, y = make_classification(
    n_samples=300, n_features=5, n_informative=3, n_redundant=1,
    n_classes=2, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scratch GaussianNB
gnb_scratch = GaussianNBScratch()
gnb_scratch.fit(X_train, y_train)
y_pred_scratch = gnb_scratch.predict(X_test)
acc_scratch = accuracy_score(y_test, y_pred_scratch)

# Sklearn GaussianNB
gnb_sklearn = GaussianNB()
gnb_sklearn.fit(X_train, y_train)
y_pred_sklearn = gnb_sklearn.predict(X_test)
acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

print(f"Gaussian NB from Scratch Accuracy: {acc_scratch * 100:.2f}%")
print(f"Sklearn GaussianNB Accuracy:       {acc_sklearn * 100:.2f}%\n")

print("Classification Report (Sklearn GaussianNB):")
print(classification_report(y_test, y_pred_sklearn))
print("=" * 80)
