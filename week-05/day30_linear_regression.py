"""
================================================================================
Day 30: Introduction to Linear Regression (From Scratch & Sklearn)
================================================================================
Course: GeeksforGeeks Machine Learning (Weeks 5-6)
Topic: Simple Linear Regression & Gradient Descent Optimization

Description:
  This script demonstrates Simple Linear Regression ($y = w x + b$):
  1. Closed-form Solution (Ordinary Least Squares / Analytical Formula)
  2. Gradient Descent Implementation from Scratch
  3. Comparison with Scikit-Learn's LinearRegression module
  4. Cost Function (MSE) tracking over training iterations

Author: ML Course Instructor / Student
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ==============================================================================
# 1. SIMPLE LINEAR REGRESSION FROM SCRATCH
# ==============================================================================
class CustomSimpleLinearRegression:
    """
    Simple Linear Regression model ($y = w*x + b$) implementing both:
    - Analytical Closed-Form Solution (OLS)
    - Gradient Descent Optimization
    """
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.w = 0.0
        self.b = 0.0
        self.loss_history = []

    def fit_ols(self, X, y):
        """Analytical Closed-Form OLS Solution: w = Cov(X,y) / Var(X)"""
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        
        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)
        
        self.w = numerator / denominator
        self.b = y_mean - (self.w * x_mean)
        return self

    def fit_gradient_descent(self, X, y):
        """Gradient Descent Optimization for w and b"""
        n = len(y)
        self.w = 0.0
        self.b = 0.0
        self.loss_history = []

        for epoch in range(self.epochs):
            # Model prediction: y_pred = w*X + b
            y_pred = self.w * X + self.b
            
            # Calculate Mean Squared Error loss
            mse = np.mean((y - y_pred) ** 2)
            self.loss_history.append(mse)
            
            # Compute partial derivatives / gradients
            dw = (-2 / n) * np.sum(X * (y - y_pred))
            db = (-2 / n) * np.sum(y - y_pred)
            
            # Update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            if (epoch + 1) % (self.epochs // 5) == 0:
                print(f"Epoch {epoch+1:4d}/{self.epochs} | MSE Loss: {mse:.4f} | w: {self.w:.4f}, b: {self.b:.4f}")

        return self

    def predict(self, X):
        return self.w * X + self.b


# ==============================================================================
# 2. MAIN DEMONSTRATION & COMPARISON
# ==============================================================================
def main():
    print("="*70)
    print("  GEEKSFORGEEKS ML COURSE - DAY 30: SIMPLE LINEAR REGRESSION")
    print("="*70)
    
    # --- Generate Synthetic Data ---
    np.random.seed(42)
    X = 2 * np.random.rand(100)
    # True relationship: y = 4 + 3*X + Gaussian Noise
    true_w, true_b = 3.0, 4.0
    y = true_b + true_w * X + np.random.randn(100) * 0.5
    
    print(f"\nGenerated 100 sample points.")
    print(f"True underlying model parameters: weight (w) = {true_w}, intercept (b) = {true_b}")

    # --- 1. Closed-form OLS Solution ---
    print("\n" + "="*50)
    print("1. Closed-Form (Analytical OLS) Solution")
    print("="*50)
    model_ols = CustomSimpleLinearRegression()
    model_ols.fit_ols(X, y)
    y_pred_ols = model_ols.predict(X)
    print(f"Estimated w: {model_ols.w:.4f}")
    print(f"Estimated b: {model_ols.b:.4f}")
    print(f"MSE:  {mean_squared_error(y, y_pred_ols):.4f}")
    print(f"R2 Score: {r2_score(y, y_pred_ols):.4f}")

    # --- 2. Gradient Descent Solution ---
    print("\n" + "="*50)
    print("2. Gradient Descent Optimization Solution")
    print("="*50)
    model_gd = CustomSimpleLinearRegression(learning_rate=0.1, epochs=500)
    model_gd.fit_gradient_descent(X, y)
    y_pred_gd = model_gd.predict(X)
    print(f"\nFinal GD w: {model_gd.w:.4f}")
    print(f"Final GD b: {model_gd.b:.4f}")
    print(f"MSE:  {mean_squared_error(y, y_pred_gd):.4f}")
    print(f"R2 Score: {r2_score(y, y_pred_gd):.4f}")

    # --- 3. Scikit-Learn Solution ---
    print("\n" + "="*50)
    print("3. Scikit-Learn LinearRegression Solution")
    print("="*50)
    X_2d = X.reshape(-1, 1)
    sklearn_reg = LinearRegression()
    sklearn_reg.fit(X_2d, y)
    y_pred_sk = sklearn_reg.predict(X_2d)
    print(f"Sklearn w: {sklearn_reg.coef_[0]:.4f}")
    print(f"Sklearn b: {sklearn_reg.intercept_:.4f}")
    print(f"MSE:  {mean_squared_error(y, y_pred_sk):.4f}")
    print(f"R2 Score: {r2_score(y, y_pred_sk):.4f}")

    # --- 4. Plotting Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Fitted Lines vs Data Points
    ax1.scatter(X, y, color='blue', alpha=0.6, label='Data Points')
    X_line = np.linspace(0, 2, 100)
    ax1.plot(X_line, model_ols.predict(X_line), 'r-', linewidth=2, label=f'OLS Line (w={model_ols.w:.2f})')
    ax1.plot(X_line, model_gd.predict(X_line), 'g--', linewidth=2, label=f'GD Line (w={model_gd.w:.2f})')
    ax1.set_title("Linear Regression Fits")
    ax1.set_xlabel("Feature X")
    ax1.set_ylabel("Target y")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Plot 2: Gradient Descent Loss Curve
    ax2.plot(range(1, len(model_gd.loss_history)+1), model_gd.loss_history, color='purple', linewidth=2)
    ax2.set_title("Gradient Descent MSE Loss Convergence")
    ax2.set_xlabel("Epoch / Iteration")
    ax2.set_ylabel("Mean Squared Error (MSE)")
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("linear_regression_fit.png")
    print("\nVisualization saved to 'linear_regression_fit.png'.")
    # plt.show()


if __name__ == "__main__":
    main()
