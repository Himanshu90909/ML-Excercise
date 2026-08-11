"""
================================================================================
Day 33: Multicollinearity & Regularization Techniques
================================================================================
Course: GeeksforGeeks Machine Learning (Weeks 5-6)
Topic: Variance Inflation Factor (VIF), Ridge, Lasso, and ElasticNet

Description:
  This script demonstrates how multicollinearity distorts OLS estimates and how
  regularization techniques stabilize models:
  1. Multicollinearity Detection using Variance Inflation Factor (VIF)
  2. Ordinary Least Squares (OLS) instability under feature correlation
  3. Ridge Regression (L2 regularization)
  4. Lasso Regression (L1 regularization - Sparse Feature Selection)
  5. ElasticNet Regression (L1 + L2 combined penalty)
  6. Coefficient Path Visualization over varying regularization strengths (alpha)

Author: ML Course Instructor / Student
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# ==============================================================================
# 1. CALCULATE VARIANCE INFLATION FACTOR (VIF)
# ==============================================================================
def calculate_vif(df_features):
    """Calculates VIF for each feature in a pandas DataFrame."""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_features.columns
    vif_data["VIF"] = [
        variance_inflation_factor(df_features.values, i)
        for i in range(df_features.shape[1])
    ]
    return vif_data.sort_values(by="VIF", ascending=False)


# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================
def main():
    print("="*70)
    print("  GEEKSFORGEEKS ML COURSE - DAY 33: MULTICOLLINEARITY & REGULARIZATION")
    print("="*70)

    # --- Generate Synthetic Data with High Multicollinearity ---
    np.random.seed(42)
    n_samples = 250
    
    # Base independent features
    x1 = np.random.normal(10, 2, n_samples)
    x2 = np.random.normal(5, 1, n_samples)
    
    # Highly collinear features
    x3 = x1 * 2.0 + np.random.normal(0, 0.05, n_samples)  # Almost 100% correlated with x1
    x4 = x2 * -1.5 + np.random.normal(0, 0.05, n_samples) # Almost 100% correlated with x2
    x5 = np.random.normal(0, 1, n_samples)                # Noise feature
    
    X_raw = np.column_stack([x1, x2, x3, x4, x5])
    feature_names = ['X1_Base', 'X2_Base', 'X3_Collinear1', 'X4_Collinear2', 'X5_Noise']
    
    # Target y is truly dependent on x1 and x2
    y = 5.0 + 3.0*x1 - 2.0*x2 + np.random.normal(0, 1.0, n_samples)
    
    df_X = pd.DataFrame(X_raw, columns=feature_names)
    
    # --- Check Correlation Matrix & VIF ---
    print("\n1. Correlation Matrix of Features:")
    print(df_X.corr().round(3))
    
    print("\n2. Variance Inflation Factors (VIF):")
    print("   (VIF > 10 indicates severe multicollinearity)")
    vif_df = calculate_vif(df_X)
    print(vif_df.to_string(index=False))

    # --- Train / Test Split and Standardization ---
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Compare OLS, Ridge, Lasso, and ElasticNet ---
    print("\n" + "="*50)
    print("3. Model Comparison: OLS vs Ridge vs Lasso vs ElasticNet")
    print("="*50)

    # A. OLS Regression
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train)
    ols_preds = ols.predict(X_test_scaled)

    # B. Ridge Regression (L2)
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train_scaled, y_train)
    ridge_preds = ridge.predict(X_test_scaled)

    # C. Lasso Regression (L1)
    lasso = Lasso(alpha=0.2)
    lasso.fit(X_train_scaled, y_train)
    lasso_preds = lasso.predict(X_test_scaled)

    # D. ElasticNet Regression (L1 + L2)
    elastic = ElasticNet(alpha=0.2, l1_ratio=0.5)
    elastic.fit(X_train_scaled, y_train)
    elastic_preds = elastic.predict(X_test_scaled)

    # Coefficient Comparison Table
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'OLS Coef': ols.coef_,
        'Ridge (L2)': ridge.coef_,
        'Lasso (L1)': lasso.coef_,
        'ElasticNet': elastic.coef_
    })
    print("\nLearned Model Coefficients:")
    print(coef_df.to_string(index=False))

    print("\nTest Performance Comparison:")
    models = {'OLS': ols_preds, 'Ridge': ridge_preds, 'Lasso': lasso_preds, 'ElasticNet': elastic_preds}
    for name, preds in models.items():
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"  {name:10s} | Test MSE: {mse:.4f} | R^2 Score: {r2:.4f}")

    # --- Regularization Path Analysis ---
    alphas = np.logspace(-3, 3, 100)
    ridge_coefs = []
    lasso_coefs = []

    for a in alphas:
        r = Ridge(alpha=a).fit(X_train_scaled, y_train)
        l = Lasso(alpha=a, max_iter=5000).fit(X_train_scaled, y_train)
        ridge_coefs.append(r.coef_)
        lasso_coefs.append(l.coef_)

    ridge_coefs = np.array(ridge_coefs)
    lasso_coefs = np.array(lasso_coefs)

    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Ridge Path
    for i in range(X_train_scaled.shape[1]):
        ax1.plot(alphas, ridge_coefs[:, i], label=feature_names[i])
    ax1.set_xscale('log')
    ax1.set_title("Ridge Trace: Coefficients vs Alpha (L2)")
    ax1.set_xlabel("Regularization Strength Alpha (log scale)")
    ax1.set_ylabel("Coefficients")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # Lasso Path
    for i in range(X_train_scaled.shape[1]):
        ax2.plot(alphas, lasso_coefs[:, i], label=feature_names[i])
    ax2.set_xscale('log')
    ax2.set_title("Lasso Trace: Coefficients vs Alpha (L1 Feature Sparsity)")
    ax2.set_xlabel("Regularization Strength Alpha (log scale)")
    ax2.set_ylabel("Coefficients")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("regularization_paths.png")
    print("\nRegularization path plots saved to 'regularization_paths.png'.")


if __name__ == "__main__":
    main()
