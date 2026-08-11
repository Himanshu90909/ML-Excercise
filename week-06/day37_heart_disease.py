"""
================================================================================
Day 37: Heart Disease Prediction (End-to-End ML Classification Project)
================================================================================
Course: GeeksforGeeks Machine Learning (Weeks 5-6)
Topic: End-to-End Applied Classification Pipeline: EDA, Preprocessing, Benchmarking & Inference

Description:
  This script implements a complete production-grade machine learning project:
  1. Synthetic Heart Disease Clinical Dataset generation
  2. Exploratory Data Analysis (EDA) & Feature Correlation
  3. Preprocessing (Imputation, Scaling, Categorical One-Hot Encoding)
  4. Benchmarking 5 Classifiers: LogisticRegression, KNN, DecisionTree, RandomForest, SVC
  5. Cross-Validated Model Selection & Hyperparameter Tuning
  6. Clinical Risk Probability Predictions for New Patient Profiles

Author: ML Course Instructor / Student
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve


# ==============================================================================
# 1. SYNTHETIC CLINICAL DATASET GENERATOR
# ==============================================================================
def generate_heart_disease_dataset(n_patients=600):
    np.random.seed(42)
    
    age = np.random.normal(54, 9, n_patients).clip(29, 77).astype(int)
    sex = np.random.choice(['Male', 'Female'], p=[0.68, 0.32], size=n_patients)
    chest_pain = np.random.choice(['Typical Angina', 'Atypical Angina', 'Non-Anginal', 'Asymptomatic'], size=n_patients)
    resting_bp = np.random.normal(130, 17, n_patients).clip(90, 200).astype(int)
    cholesterol = np.random.normal(245, 50, n_patients).clip(120, 500).astype(int)
    fasting_bs = np.random.choice([0, 1], p=[0.85, 0.15], size=n_patients)
    max_hr = np.random.normal(150, 22, n_patients).clip(70, 202).astype(int)
    exercise_angina = np.random.choice(['Yes', 'No'], p=[0.33, 0.67], size=n_patients)
    
    # Calculate physiological risk logit for target creation
    risk_score = (
        0.05 * (age - 50) +
        0.8 * (sex == 'Male') +
        1.2 * (chest_pain == 'Asymptomatic') +
        0.02 * (resting_bp - 120) +
        0.01 * (cholesterol - 200) +
        0.7 * fasting_bs -
        0.03 * (max_hr - 150) +
        1.1 * (exercise_angina == 'Yes') -
        1.5
    )
    prob = 1.0 / (1.0 + np.exp(-risk_score))
    target = (np.random.rand(n_patients) < prob).astype(int)
    
    df = pd.DataFrame({
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'HeartDisease': target
    })
    
    # Introduce small missingness to test imputation
    missing_idx = np.random.choice(n_patients, size=20, replace=False)
    df.loc[missing_idx, 'Cholesterol'] = np.nan
    
    return df


# ==============================================================================
# MAIN PROJECT WORKFLOW
# ==============================================================================
def main():
    print("="*70)
    print("  GEEKSFORGEEKS ML COURSE - DAY 37: HEART DISEASE PREDICTION PROJECT")
    print("="*70)

    # 1. Load Data & EDA
    df = generate_heart_disease_dataset(n_patients=600)
    print("\n1. Clinical Dataset Overview:")
    print(df.info())
    print("\nTarget Distribution (0 = Healthy, 1 = Heart Disease):")
    print(df['HeartDisease'].value_counts(normalize=True).round(3))
    
    print("\nSummary Statistics of Numerical Features:")
    print(df.describe().round(2))

    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']

    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Preprocessing Pipelines
    num_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']
    cat_features = ['Sex', 'ChestPainType', 'FastingBS', 'ExerciseAngina']

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

    # 4. Model Benchmarking
    print("\n" + "="*50)
    print("2. Classifier Benchmarking (5-Fold Cross Validation)")
    print("="*50)

    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42)
    }

    benchmark_results = []
    trained_pipelines = {}

    for name, clf in classifiers.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe
        
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        
        benchmark_results.append({
            'Model': name,
            'CV Accuracy': cv_scores.mean(),
            'Test Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })

    df_bench = pd.DataFrame(benchmark_results).sort_values(by='Test Accuracy', ascending=False)
    print(df_bench.to_string(index=False))

    # 5. Hyperparameter Tuning on Best Model (Random Forest)
    print("\n" + "="*50)
    print("3. Fine-Tuning Champion Model (Random Forest)")
    print("="*50)

    rf_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [3, 5, 8, None],
        'classifier__min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(rf_pipe, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print("Best Hyperparameters:", grid_search.best_params_)
    
    tuned_preds = best_model.predict(X_test)
    tuned_probs = best_model.predict_proba(X_test)[:, 1]
    print(f"Tuned Test F1-Score: {f1_score(y_test, tuned_preds):.4f}")
    print(f"Tuned Test ROC-AUC:  {roc_auc_score(y_test, tuned_probs):.4f}")

    # 6. Inference on New Patient Cases
    print("\n" + "="*50)
    print("4. Clinical Risk Assessment for New Patients")
    print("="*50)

    new_patients = pd.DataFrame([
        {'Age': 62, 'Sex': 'Male',   'ChestPainType': 'Asymptomatic', 'RestingBP': 148, 'Cholesterol': 280, 'FastingBS': 1, 'MaxHR': 115, 'ExerciseAngina': 'Yes'},
        {'Age': 42, 'Sex': 'Female', 'ChestPainType': 'Non-Anginal',  'RestingBP': 118, 'Cholesterol': 190, 'FastingBS': 0, 'MaxHR': 172, 'ExerciseAngina': 'No'}
    ])

    patient_probs = best_model.predict_proba(new_patients)[:, 1]
    for i, p in enumerate(patient_probs):
        risk_level = "HIGH RISK" if p >= 0.5 else "LOW RISK"
        print(f"Patient {i+1} ({new_patients.iloc[i]['Age']}yo {new_patients.iloc[i]['Sex']}): Disease Prob = {p*100:.1f}% -> [{risk_level}]")

    # 7. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curves comparison
    for name, pipe in trained_pipelines.items():
        probs = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val = roc_auc_score(y_test, probs)
        ax1.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.2f})')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_title("ROC Curves across Models")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Feature Importance plot of Random Forest
    rf_clf = best_model.named_steps['classifier']
    feature_names_transformed = best_model.named_steps['preprocessor'].get_feature_names_out()
    importances = rf_clf.feature_importances_

    sorted_idx = np.argsort(importances)
    ax2.barh(range(len(sorted_idx)), importances[sorted_idx], color='teal')
    ax2.set_yticks(range(len(sorted_idx)))
    ax2.set_yticklabels([feature_names_transformed[i].split('__')[-1] for i in sorted_idx])
    ax2.set_title("Random Forest Feature Importances")
    ax2.set_xlabel("Importance Score")
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("heart_disease_project_results.png")
    print("\nProject plots saved to 'heart_disease_project_results.png'.")


if __name__ == "__main__":
    main()
