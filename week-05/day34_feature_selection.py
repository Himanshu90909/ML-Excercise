"""
================================================================================
Day 34: Feature Selection Methods in Machine Learning
================================================================================
Course: GeeksforGeeks Machine Learning (Weeks 5-6)
Topic: Filter, Wrapper, and Embedded Feature Selection Techniques

Description:
  This script demonstrates three core feature selection paradigms:
  1. Filter Methods: Correlation thresholding & SelectKBest (ANOVA f_regression / chi2)
  2. Wrapper Methods: Recursive Feature Elimination (RFE) & Cross-Validated RFE (RFECV)
  3. Embedded Methods: Lasso L1 penalty & Tree-based Feature Importance (Random Forest)
  
Author: ML Course Instructor / Student
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest, f_regression, RFE, RFECV
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================
def main():
    print("="*70)
    print("  GEEKSFORGEEKS ML COURSE - DAY 34: FEATURE SELECTION METHODS")
    print("="*70)

    # --- Generate Synthetic Dataset with Informative vs Uninformative Features ---
    np.random.seed(42)
    n_samples = 300
    n_total_features = 15
    n_informative = 5
    
    X_raw, y, true_coefs = make_regression(
        n_samples=n_samples,
        n_features=n_total_features,
        n_informative=n_informative,
        noise=10.0,
        coef=True,
        random_state=42
    )
    
    feature_names = [f"Feat_{i+1:02d}" for i in range(n_total_features)]
    df_X = pd.DataFrame(X_raw, columns=feature_names)
    
    print("\nTrue underlying feature coefficients:")
    for f_name, coef in zip(feature_names, true_coefs):
        is_truly_informative = "★ INFORMATIVE" if abs(coef) > 1.0 else "  Noise"
        print(f"  {f_name}: {coef:7.2f}  [{is_truly_informative}]")

    # ==========================================================================
    # 1. FILTER METHODS
    # ==========================================================================
    print("\n" + "="*50)
    print("1. Filter Methods (SelectKBest with ANOVA F-test)")
    print("="*50)
    
    k_best_selector = SelectKBest(score_func=f_regression, k=5)
    k_best_selector.fit(X_raw, y)
    
    filter_scores = k_best_selector.scores_
    filter_support = k_best_selector.get_support()
    
    df_filter = pd.DataFrame({
        'Feature': feature_names,
        'F_Score': filter_scores,
        'Selected_Top5': filter_support
    }).sort_values(by='F_Score', ascending=False)
    print(df_filter.to_string(index=False))

    # ==========================================================================
    # 2. WRAPPER METHODS
    # ==========================================================================
    print("\n" + "="*50)
    print("2. Wrapper Methods (Recursive Feature Elimination - RFE & RFECV)")
    print("="*50)
    
    estimator = LinearRegression()
    rfe = RFE(estimator=estimator, n_features_to_select=5)
    rfe.fit(X_raw, y)
    
    # RFECV to automatically discover optimal number of features
    rfecv = RFECV(estimator=estimator, step=1, cv=KFold(5), scoring='r2')
    rfecv.fit(X_raw, y)
    
    print(f"RFE Selected Features (top 5): {[feature_names[i] for i in range(n_total_features) if rfe.support_[i]]}")
    print(f"RFECV Optimal Number of Features: {rfecv.n_features_}")
    print(f"RFECV Selected Features: {[feature_names[i] for i in range(n_total_features) if rfecv.support_[i]]}")

    # ==========================================================================
    # 3. EMBEDDED METHODS
    # ==========================================================================
    print("\n" + "="*50)
    print("3. Embedded Methods (Lasso L1 & Random Forest Importance)")
    print("="*50)
    
    # A. Lasso CV
    lasso_cv = LassoCV(cv=5, random_state=42).fit(X_raw, y)
    lasso_selected = np.abs(lasso_cv.coef_) > 1e-3
    
    # B. Random Forest Feature Importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_raw, y)
    rf_importances = rf.feature_importances_
    
    df_embedded = pd.DataFrame({
        'Feature': feature_names,
        'Lasso_Coef': lasso_cv.coef_,
        'Lasso_Selected': lasso_selected,
        'RF_Importance': rf_importances
    }).sort_values(by='RF_Importance', ascending=False)
    print(df_embedded.to_string(index=False))

    # ==========================================================================
    # SUMMARY COMPARISON TABLE
    # ==========================================================================
    print("\n" + "="*50)
    print("Summary Matrix of Selected Features Across Methods")
    print("="*50)
    
    df_summary = pd.DataFrame({
        'Feature': feature_names,
        'Truly_Informative': np.abs(true_coefs) > 1.0,
        'Filter_KBest': filter_support,
        'Wrapper_RFE': rfe.support_,
        'Wrapper_RFECV': rfecv.support_,
        'Embedded_Lasso': lasso_selected,
        'RF_Top5': np.argsort(rf_importances)[::-1] < 5
    })
    print(df_summary.to_string(index=False))

    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Barplot 1: SelectKBest F-Scores
    indices = np.arange(n_total_features)
    colors = ['tab:green' if np.abs(true_coefs[i]) > 1.0 else 'tab:gray' for i in range(n_total_features)]
    ax1.bar(indices, filter_scores, color=colors, edgecolor='k')
    ax1.set_xticks(indices)
    ax1.set_xticklabels(feature_names, rotation=45)
    ax1.set_title("Filter Method: ANOVA F-Scores per Feature")
    ax1.set_ylabel("F-Score")
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot 2: RFECV Performance vs Number of Features
    n_feats_grid = range(1, len(rfecv.cv_results_['mean_test_score']) + 1)
    ax2.plot(n_feats_grid, rfecv.cv_results_['mean_test_score'], marker='o', color='tab:blue')
    ax2.set_title("Wrapper Method: RFECV Score vs Number of Features")
    ax2.set_xlabel("Number of Selected Features")
    ax2.set_ylabel("Cross-Validated R^2 Score")
    ax2.axvline(rfecv.n_features_, color='r', linestyle='--', label=f'Optimal Feats = {rfecv.n_features_}')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("feature_selection_comparison.png")
    print("\nFeature selection plot saved to 'feature_selection_comparison.png'.")


if __name__ == "__main__":
    main()
