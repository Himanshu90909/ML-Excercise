"""
================================================================================
Day 31: Multiple Linear Regression & Vectorized Gradient Descent
================================================================================
Course: GeeksforGeeks Machine Learning (Weeks 5-6)
Topic: Multiple Linear Regression, Feature Scaling & Vectorized Gradient Descent

Description:
  This script implements Multiple Linear Regression for d-dimensional feature vectors:
  1. Feature Standardization (Z-score Scaling)
  2. Vectorized Matrix Gradient Descent from Scratch
  3. Analytical Normal Equation: w = (X^T X)^(-1) X^T y
  4. Performance evaluation: MSE, RMSE, MAE, R^2, and Adjusted R^2

Author: ML Course Instructor / Student
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# ==============================================================================
# FEATURE STANDARDIZER (FROM SCRATCH)
# ==============================================================================
class CustomStandardScaler:
    """Standardize features by removing the mean and scaling to unit variance."""
    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Avoid division by zero
        self.std[self.std == 0] = 1.0
        return (X - self.mean) / self.std

    def transform(self, X):
        return (X - self.mean) / self.std


# ==============================================================================
# MULTIPLE LINEAR REGRESSION (FROM SCRATCH)
# ==============================================================================
class CustomMultipleLinearRegression:
    def __init__(self, learning_rate=0.05, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def fit_gradient_descent(self, X, y):
        """Vectorized Batch Gradient Descent"""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(self.epochs):
            # Predictions: y_hat = X * w + b
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Loss (MSE)
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)
            
            # Vectorized gradients
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)
            
            # Parameter updates
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            if (epoch + 1) % (self.epochs // 5) == 0:
                print(f"Epoch {epoch+1:4d}/{self.epochs} | MSE Loss: {loss:.4f}")

    def fit_normal_equation(self, X, y):
        """Closed-form Normal Equation: w = (X_b^T * X_b)^(-1) * X_b^T * y"""
        n_samples = X.shape[0]
        # Add column of 1s for bias term
        X_b = np.hstack([np.ones((n_samples, 1)), X])
        
        # Calculate (X_b^T * X_b)^(-1) * X_b^T * y
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.bias = theta[0]
        self.weights = theta[1:]

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# ==============================================================================
# EVALUATION METRICS HELPER
# ==============================================================================
def calculate_metrics(y_true, y_pred, n_features):
    n = len(y_true)
    p = n_features
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "Adj_R2": adj_r2}


# ==============================================================================
# MAIN DEMO
# ==============================================================================
def main():
    print("="*70)
    print("  GEEKSFORGEEKS ML COURSE - DAY 31: MULTIPLE LINEAR REGRESSION")
    print("="*70)

    # --- Generate Synthetic Multiple Feature Dataset ---
    np.random.seed(42)
    n_samples = 300
    
    # 3 Features: Size (sqft), Bedrooms, Distance to City Center (km)
    size = np.random.normal(2000, 500, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    distance = np.random.uniform(1, 25, n_samples)
    
    # Target formula: Price ($k) = 150 + 0.12*size + 15*bedrooms - 3.5*distance + noise
    true_bias = 150.0
    true_weights = np.array([0.12, 15.0, -3.5])
    
    X_raw = np.column_stack([size, bedrooms, distance])
    noise = np.random.normal(0, 15, n_samples)
    y = true_bias + np.dot(X_raw, true_weights) + noise
    
    df = pd.DataFrame(X_raw, columns=['Size_sqft', 'Bedrooms', 'Distance_km'])
    df['Price_k'] = y
    print("\nDataset Preview:")
    print(df.head())
    
    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)
    
    # Feature Scaling
    scaler = CustomStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n" + "="*50)
    print("1. Custom Vectorized Gradient Descent")
    print("="*50)
    gd_model = CustomMultipleLinearRegression(learning_rate=0.05, epochs=1000)
    gd_model.fit_gradient_descent(X_train_scaled, y_train)
    y_pred_gd = gd_model.predict(X_test_scaled)
    gd_metrics = calculate_metrics(y_test, y_pred_gd, n_features=3)
    
    print("\nLearned Weights (Standardized Space):", gd_model.weights)
    print(f"Learned Bias: {gd_model.bias:.2f}")
    for k, v in gd_metrics.items():
        print(f"  {k:8s}: {v:.4f}")

    print("\n" + "="*50)
    print("2. Closed-Form Normal Equation")
    print("="*50)
    ne_model = CustomMultipleLinearRegression()
    ne_model.fit_normal_equation(X_train_scaled, y_train)
    y_pred_ne = ne_model.predict(X_test_scaled)
    ne_metrics = calculate_metrics(y_test, y_pred_ne, n_features=3)
    
    print("Learned Weights:", ne_model.weights)
    print(f"Learned Bias: {ne_model.bias:.2f}")
    for k, v in ne_metrics.items():
        print(f"  {k:8s}: {v:.4f}")

    print("\n" + "="*50)
    print("3. Scikit-Learn LinearRegression Baseline")
    print("="*50)
    sk_model = LinearRegression()
    sk_model.fit(X_train_scaled, y_train)
    y_pred_sk = sk_model.predict(X_test_scaled)
    sk_metrics = calculate_metrics(y_test, y_pred_sk, n_features=3)
    
    print("Sklearn Weights:", sk_model.coef_)
    print(f"Sklearn Bias: {sk_model.intercept_:.2f}")
    for k, v in sk_metrics.items():
        print(f"  {k:8s}: {v:.4f}")

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss Convergence
    ax1.plot(gd_model.loss_history, color='tab:blue', linewidth=2)
    ax1.set_title("Gradient Descent MSE Loss Convergence")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Plot 2: Actual vs Predicted
    ax2.scatter(y_test, y_pred_gd, alpha=0.7, color='tab:green', label='Predicted vs Actual')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit (1:1)')
    ax2.set_title(f"Actual vs Predicted Prices (R2 = {gd_metrics['R2']:.3f})")
    ax2.set_xlabel("Actual Price ($k)")
    ax2.set_ylabel("Predicted Price ($k)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("multiple_linear_regression.png")
    print("\nVisualization saved to 'multiple_linear_regression.png'.")


if __name__ == "__main__":
    main()
