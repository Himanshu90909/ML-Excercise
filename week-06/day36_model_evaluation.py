"""
================================================================================
Day 36: Model Evaluation Metrics & Hyperparameter Tuning
================================================================================
Course: GeeksforGeeks Machine Learning (Weeks 5-6)
Topic: Evaluation Metrics, Cross-Validation & Grid/Randomized Search Tuning

Description:
  This script comprehensively covers model evaluation and hyperparameter optimization:
  1. Classification Metrics: Accuracy, Precision, Recall, F1, Specificity, Log-Loss
  2. Confusion Matrix analysis and visual heatmap
  3. ROC-AUC and Precision-Recall curves
  4. Cross-Validation: K-Fold vs Stratified K-Fold
  5. Hyperparameter Tuning: GridSearchCV vs RandomizedSearchCV

Author: ML Course Instructor / Student
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split, KFold, StratifiedKFold, cross_val_score,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve,
    auc
)


# ==============================================================================
# 1. CUSTOM METRICS CALCULATION FROM CONFUSION MATRIX
# ==============================================================================
def calculate_metrics_from_scratch(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "TN": TN, "FP": FP, "FN": FN, "TP": TP,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Specificity": specificity,
        "F1-Score": f1
    }


# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================
def main():
    print("="*70)
    print("  GEEKSFORGEEKS ML COURSE - DAY 36: MODEL EVALUATION & TUNING")
    print("="*70)

    # --- Generate Synthetic Imbalanced Dataset ---
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=6, n_redundant=2,
        weights=[0.8, 0.2], random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    print(f"\nSynthetic Dataset Class Distribution:")
    print(f"  Class 0 (Negative): {np.sum(y == 0)} samples")
    print(f"  Class 1 (Positive): {np.sum(y == 1)} samples (Imbalanced ~20%)")

    # --- Baseline Model Training ---
    base_rf = RandomForestClassifier(random_state=42)
    base_rf.fit(X_train, y_train)

    y_pred = base_rf.predict(X_test)
    y_probs = base_rf.predict_proba(X_test)[:, 1]

    # --- 1. Comprehensive Classification Metrics ---
    print("\n" + "="*50)
    print("1. Evaluation Metrics Breakdown")
    print("="*50)
    metrics_scratch = calculate_metrics_from_scratch(y_test, y_pred)
    for metric_name, val in metrics_scratch.items():
        if isinstance(val, int):
            print(f"  {metric_name:20s}: {val}")
        else:
            print(f"  {metric_name:20s}: {val:.4f}")

    # ROC-AUC and PR-AUC
    roc_auc = roc_auc_score(y_test, y_probs)
    precisions, recalls, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recalls, precisions)
    print(f"  {'ROC-AUC Score':20s}: {roc_auc:.4f}")
    print(f"  {'PR-AUC Score':20s}: {pr_auc:.4f}")

    # --- 2. Cross-Validation Comparison ---
    print("\n" + "="*50)
    print("2. K-Fold vs Stratified K-Fold Cross-Validation")
    print("="*50)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    kf_scores = cross_val_score(base_rf, X, y, cv=kf, scoring='f1')
    skf_scores = cross_val_score(base_rf, X, y, cv=skf, scoring='f1')

    print(f"Standard 5-Fold F1 Scores:   {kf_scores.round(3)} | Mean = {kf_scores.mean():.4f}")
    print(f"Stratified 5-Fold F1 Scores: {skf_scores.round(3)} | Mean = {skf_scores.mean():.4f}")

    # --- 3. Hyperparameter Tuning (GridSearchCV & RandomizedSearchCV) ---
    print("\n" + "="*50)
    print("3. Hyperparameter Tuning (GridSearchCV vs RandomizedSearchCV)")
    print("="*50)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy']
    }

    # A. GridSearchCV
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=skf,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print("\n[GridSearchCV Result]")
    print(f"  Best Parameters: {grid_search.best_params_}")
    print(f"  Best CV F1-Score: {grid_search.best_score_:.4f}")

    # B. RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=param_grid,
        n_iter=10,
        cv=skf,
        scoring='f1',
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    print("\n[RandomizedSearchCV Result (10 iterations)]")
    print(f"  Best Parameters: {random_search.best_params_}")
    print(f"  Best CV F1-Score: {random_search.best_score_:.4f}")

    # Best Tuned Model Test Performance
    best_model = grid_search.best_estimator_
    tuned_preds = best_model.predict(X_test)
    tuned_f1 = f1_score(y_test, tuned_preds)
    print(f"\nTuned Model Test F1-Score: {tuned_f1:.4f}")

    # --- 4. Visualizations ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot A: Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    im = axes[0].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[0].set_title("Confusion Matrix")
    plt.colorbar(im, ax=axes[0])
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(['Pred Negative', 'Pred Positive'])
    axes[0].set_yticklabels(['True Negative', 'True Positive'])

    # Annotate numbers in heatmap
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", color="red", fontsize=14, weight='bold')

    # Plot B: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    axes[1].set_title('Receiver Operating Characteristic (ROC)')
    axes[1].set_xlabel('False Positive Rate (1 - Specificity)')
    axes[1].set_ylabel('True Positive Rate (Recall)')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # Plot C: Precision-Recall Curve
    axes[2].plot(recalls, precisions, color='purple', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    axes[2].set_title('Precision-Recall Curve')
    axes[2].set_xlabel('Recall (Sensitivity)')
    axes[2].set_ylabel('Precision')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("model_evaluation_metrics.png")
    print("\nEvaluation plots saved to 'model_evaluation_metrics.png'.")


if __name__ == "__main__":
    main()
