"""
================================================================================
Day 38: Decision Tree Fundamentals (Entropy, Information Gain & Gini Impurity)
================================================================================
Course: GeeksforGeeks Machine Learning (Weeks 5-6)
Topic: Information Theory, Node Impurity Metrics & Decision Tree Splitting

Description:
  This script implements decision tree split criteria from scratch:
  1. Entropy: H(S) = - sum(p_i * log2(p_i))
  2. Information Gain: IG(S, A) = H(S) - sum(|S_v|/|S| * H(S_v))
  3. Gini Impurity: Gini(S) = 1 - sum(p_i^2)
  4. Step-by-step split selection on the classic 'PlayTennis' dataset
  5. Scikit-learn DecisionTreeClassifier comparison and tree diagram visualization

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
# 1. IMPURITY & INFORMATION GAIN FUNCTIONS FROM SCRATCH
# ==============================================================================
def calculate_entropy(y):
    """Calculates Shannon Entropy in bits (log2)."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy


def calculate_gini(y):
    """Calculates Gini Impurity."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    gini = 1.0 - np.sum(probabilities ** 2)
    return gini


def calculate_information_gain(df, feature_name, target_name, criterion='entropy'):
    """Calculates Information Gain for splitting df on feature_name."""
    total_samples = len(df)
    parent_y = df[target_name].values
    
    # Parent impurity
    if criterion == 'entropy':
        parent_impurity = calculate_entropy(parent_y)
    else:
        parent_impurity = calculate_gini(parent_y)
        
    # Weighted child impurity
    weighted_child_impurity = 0.0
    feature_values = df[feature_name].unique()
    
    split_details = {}
    for val in feature_values:
        sub_df = df[df[feature_name] == val]
        child_y = sub_df[target_name].values
        weight = len(sub_df) / total_samples
        
        if criterion == 'entropy':
            child_imp = calculate_entropy(child_y)
        else:
            child_imp = calculate_gini(child_y)
            
        weighted_child_impurity += weight * child_imp
        split_details[val] = {'count': len(sub_df), 'impurity': child_imp, 'target_counts': dict(pd.Series(child_y).value_counts())}
        
    info_gain = parent_impurity - weighted_child_impurity
    return info_gain, parent_impurity, weighted_child_impurity, split_details


# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================
def main():
    print("="*70)
    print("  GEEKSFORGEEKS ML COURSE - DAY 38: DECISION TREE BASICS")
    print("="*70)

    # --- Classic PlayTennis Synthetic Dataset ---
    data = {
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
        'Temp':    ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
        'Humidity':['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
        'Wind':    ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
        'Play':    ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    }
    df = pd.DataFrame(data)

    print("\nPlayTennis Dataset (14 Samples):")
    print(df)

    parent_entropy = calculate_entropy(df['Play'])
    parent_gini = calculate_gini(df['Play'])

    print(f"\nRoot Node Dataset Impurity:")
    print(f"  Root Entropy: {parent_entropy:.4f} bits")
    print(f"  Root Gini:    {parent_gini:.4f}")

    # --- Step-by-Step Information Gain Calculation ---
    print("\n" + "="*50)
    print("Information Gain (IG) for Each Feature (Entropy-based):")
    print("="*50)

    features = ['Outlook', 'Temp', 'Humidity', 'Wind']
    best_feature = None
    max_ig = -1.0

    for feat in features:
        ig, p_imp, child_imp, details = calculate_information_gain(df, feat, 'Play', criterion='entropy')
        print(f"Feature: {feat:10s} | Information Gain = {ig:.4f} (Weighted Child Entropy = {child_imp:.4f})")
        if ig > max_ig:
            max_ig = ig
            best_feature = feat

    print(f"\n★ BEST ROOT SPLIT FEATURE: '{best_feature}' with Information Gain = {max_ig:.4f} bits")

    # --- Scikit-Learn DecisionTreeClassifier Comparison ---
    print("\n" + "="*50)
    print("Scikit-Learn DecisionTreeClassifier Comparison:")
    print("="*50)

    encoder = OrdinalEncoder()
    X_encoded = encoder.fit_transform(df[features])
    y_encoded = (df['Play'] == 'Yes').astype(int)

    clf_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
    clf_gini.fit(X_encoded, y_encoded)

    clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
    clf_entropy.fit(X_encoded, y_encoded)

    print("Sklearn Tree (Gini) trained successfully!")
    print("Sklearn Tree (Entropy) trained successfully!")
    print("Tree Depth:", clf_entropy.get_depth())
    print("Number of Leaves:", clf_entropy.get_n_leaves())

    # --- Visualizations ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Entropy vs Gini Curve over Binary Probability p
    p_vals = np.linspace(0.0001, 0.9999, 200)
    entropy_curve = - (p_vals * np.log2(p_vals) + (1 - p_vals) * np.log2(1 - p_vals))
    # Rescaled entropy to compare on same scale (max 0.5)
    entropy_rescaled = 0.5 * entropy_curve
    gini_curve = 1.0 - (p_vals**2 + (1 - p_vals)**2)

    ax1.plot(p_vals, entropy_curve, 'b-', label='Entropy (bits, max=1.0)')
    ax1.plot(p_vals, entropy_rescaled, 'b--', label='Rescaled Entropy (0.5 * H)')
    ax1.plot(p_vals, gini_curve, 'r-', label='Gini Impurity (max=0.5)')
    ax1.set_title("Impurity Measures vs Positive Class Probability (p)")
    ax1.set_xlabel("Probability p of Positive Class")
    ax1.set_ylabel("Impurity Score")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot 2: Decision Tree Structure Visualization
    plot_tree(
        clf_entropy,
        feature_names=features,
        class_names=['No', 'Yes'],
        filled=True,
        ax=ax2,
        fontsize=9
    )
    ax2.set_title("Trained Decision Tree (Entropy/ID3)")

    plt.tight_layout()
    plt.savefig("decision_tree_basics.png")
    print("\nVisualizations saved to 'decision_tree_basics.png'.")


if __name__ == "__main__":
    main()
