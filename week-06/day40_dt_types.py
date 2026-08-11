"""
================================================================================
Day 40: Decision Tree Algorithms: ID3, C4.5, and CART
================================================================================
Course: GeeksforGeeks Machine Learning (Weeks 5-6)
Topic: Algorithmic Evolutionary Variants & Splitting Criteria Comparison

Description:
  This script analyzes the three dominant historical decision tree algorithms:
  1. ID3 (Iterative Dichotomiser 3): Information Gain, multi-way categorical splits
  2. C4.5: Gain Ratio (normalizing high-cardinality bias), continuous features, post-pruning
  3. CART (Classification & Regression Trees): Binary splits, Gini Impurity / MSE, CCP Pruning
  
  Demonstrates custom Gain Ratio calculation in Python vs Information Gain.

Author: ML Course Instructor / Student
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OrdinalEncoder


# ==============================================================================
# 1. MATHEMATICAL FUNCTIONS FOR ID3 vs C4.5 vs CART
# ==============================================================================
def calculate_entropy(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probs if p > 0])


def calculate_split_info(df, feature_name):
    """Calculates Split Information / Split Entropy for C4.5 Gain Ratio denominator."""
    total_samples = len(df)
    feature_counts = df[feature_name].value_counts()
    split_info = 0.0
    for count in feature_counts:
        p = count / total_samples
        if p > 0:
            split_info -= p * np.log2(p)
    return split_info


def calculate_id3_vs_c45_criteria(df, feature_name, target_name):
    """
    Computes Information Gain (ID3) and Gain Ratio (C4.5).
    Formula: GainRatio = InformationGain / SplitInfo
    """
    total_samples = len(df)
    parent_entropy = calculate_entropy(df[target_name])
    
    # Weighted child entropy
    weighted_child_entropy = 0.0
    for val in df[feature_name].unique():
        sub_df = df[df[feature_name] == val]
        p = len(sub_df) / total_samples
        weighted_child_entropy += p * calculate_entropy(sub_df[target_name])
        
    info_gain = parent_entropy - weighted_child_entropy
    split_info = calculate_split_info(df, feature_name)
    
    # Gain Ratio avoids division by zero
    gain_ratio = info_gain / split_info if split_info > 0 else 0.0
    
    return info_gain, split_info, gain_ratio


# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================
def main():
    print("="*70)
    print("  GEEKSFORGEEKS ML COURSE - DAY 40: ID3 vs C4.5 vs CART ALGORITHMS")
    print("="*70)

    # --- Demonstrate ID3 High-Cardinality Bias ---
    # Create dataset with a unique 'Customer_ID' column (high cardinality)
    np.random.seed(42)
    n = 20
    df_demo = pd.DataFrame({
        'Customer_ID': [f"ID_{i:02d}" for i in range(n)],     # Unique per row
        'Credit_Score': np.random.choice(['Low', 'High'], size=n),
        'Income_Level': np.random.choice(['Medium', 'High'], size=n),
        'Loan_Approved': np.random.choice(['No', 'Yes'], size=n)
    })

    print("\nDataset with High-Cardinality ID Column:")
    print(df_demo.head(6))

    print("\n" + "="*60)
    print("Comparison of ID3 (Information Gain) vs C4.5 (Gain Ratio):")
    print("="*60)
    print(f"{'Feature':15s} | {'Info Gain (ID3)':15s} | {'Split Info':12s} | {'Gain Ratio (C4.5)':15s}")
    print("-" * 65)

    features = ['Customer_ID', 'Credit_Score', 'Income_Level']
    ig_list = []
    gr_list = []

    for f in features:
        ig, si, gr = calculate_id3_vs_c45_criteria(df_demo, f, 'Loan_Approved')
        ig_list.append(ig)
        gr_list.append(gr)
        print(f"{f:15s} | {ig:15.4f} | {si:12.4f} | {gr:15.4f}")

    print("\n★ Key Insight:")
    print("  - ID3 falsely prefers 'Customer_ID' (Info Gain = 1.0) because each row becomes a pure node.")
    print("  - C4.5 penalizes high cardinality using SplitInfo, dropping Gain Ratio down to reasonable levels.")

    # --- CART Algorithm Implementation (Scikit-Learn Standard) ---
    print("\n" + "="*60)
    print("CART Algorithm (Binary Splits with Gini Impurity):")
    print("="*60)

    enc = OrdinalEncoder()
    X_encoded = enc.fit_transform(df_demo[features])
    y_encoded = (df_demo['Loan_Approved'] == 'Yes').astype(int)

    cart_tree = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
    cart_tree.fit(X_encoded, y_encoded)

    print("CART Binary Tree Trained.")
    print(f"Tree Depth: {cart_tree.get_depth()}")
    print(f"Feature Importances (Gini): {cart_tree.feature_importances_.round(4)}")

    # --- Comprehensive Comparison Table ---
    print("\n" + "="*70)
    print("SUMMARY COMPARISON OF DECISION TREE ALGORITHMS")
    print("="*70)
    summary_data = [
        {"Property": "Splitting Criterion", "ID3": "Information Gain", "C4.5": "Gain Ratio", "CART": "Gini / MSE"},
        {"Property": "Split Structure", "ID3": "Multi-way splits", "C4.5": "Multi-way splits", "CART": "Binary splits only"},
        {"Property": "Feature Types", "ID3": "Categorical only", "C4.5": "Categorical & Continuous", "CART": "Categorical & Continuous"},
        {"Property": "Task Capability", "ID3": "Classification only", "C4.5": "Classification only", "CART": "Classification & Regression"},
        {"Property": "Missing Values", "ID3": "Cannot handle", "C4.5": "Handles via fractioning", "CART": "Handles via surrogate splits"},
        {"Property": "Pruning Strategy", "ID3": "None", "C4.5": "Subtree raising / Error-based", "CART": "Cost-Complexity Pruning (CCP)"}
    ]
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(8, 4))
    x_indices = np.arange(len(features))
    width = 0.35

    ax.bar(x_indices - width/2, ig_list, width, label='Information Gain (ID3)', color='coral')
    ax.bar(x_indices + width/2, gr_list, width, label='Gain Ratio (C4.5)', color='teal')

    ax.set_xticks(x_indices)
    ax.set_xticklabels(features)
    ax.set_title("ID3 (Information Gain) vs C4.5 (Gain Ratio) Feature Bias")
    ax.set_ylabel("Metric Score")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("dt_types_comparison.png")
    print("\nFeature bias comparison plot saved to 'dt_types_comparison.png'.")


if __name__ == "__main__":
    main()
