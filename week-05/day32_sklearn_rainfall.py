"""
================================================================================
Day 32: Scikit-Learn Library & Rainfall Prediction
================================================================================
Course: GeeksforGeeks Machine Learning (Weeks 5-6)
Topic: End-to-End Scikit-Learn Pipeline & Weather/Rainfall Regression

Description:
  This script demonstrates a complete Scikit-Learn machine learning pipeline:
  1. Synthetic Rainfall Dataset creation with missing values and categorical features
  2. Data Preprocessing with ColumnTransformer (Imputation, Scaling, One-Hot Encoding)
  3. Scikit-Learn Model Pipeline Integration
  4. Regression Evaluation Metrics (MAE, MSE, RMSE, R^2)
  5. Residual Analysis and Prediction on new weather observations

Author: ML Course Instructor / Student
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==============================================================================
# DATASET GENERATION
# ==============================================================================
def generate_rainfall_dataset(n_samples=500):
    np.random.seed(42)
    
    # Numerical features
    temp = np.random.uniform(10, 40, n_samples)          # Temperature (°C)
    humidity = np.random.uniform(30, 98, n_samples)      # Relative Humidity (%)
    pressure = np.random.uniform(990, 1030, n_samples)   # Pressure (hPa)
    wind_speed = np.random.uniform(2, 45, n_samples)     # Wind Speed (km/h)
    cloud_cover = np.random.uniform(0, 100, n_samples)   # Cloud Cover (%)
    
    # Categorical feature
    regions = np.random.choice(['Coastal', 'Inland', 'Mountainous'], size=n_samples)
    
    # Target: Rainfall (mm) formula with synthetic physical relationships
    # Higher humidity, cloud cover, lower pressure -> higher rainfall
    rainfall = (
        0.8 * humidity + 
        0.5 * cloud_cover - 
        0.4 * temp - 
        0.3 * (pressure - 1000) + 
        0.2 * wind_speed +
        np.where(regions == 'Coastal', 15, np.where(regions == 'Mountainous', 25, 0)) +
        np.random.normal(0, 8, n_samples)
    )
    # Ensure non-negative rainfall
    rainfall = np.maximum(0, rainfall)
    
    df = pd.DataFrame({
        'Temperature_C': temp,
        'Humidity_pct': humidity,
        'Pressure_hPa': pressure,
        'WindSpeed_kmh': wind_speed,
        'CloudCover_pct': cloud_cover,
        'Region_Type': regions,
        'Rainfall_mm': rainfall
    })
    
    # Introduce random missing values (~5%) to test imputation pipeline
    mask_temp = np.random.rand(n_samples) < 0.05
    mask_hum = np.random.rand(n_samples) < 0.05
    df.loc[mask_temp, 'Temperature_C'] = np.nan
    df.loc[mask_hum, 'Humidity_pct'] = np.nan
    
    return df


# ==============================================================================
# MAIN PIPELINE WORKFLOW
# ==============================================================================
def main():
    print("="*70)
    print("  GEEKSFORGEEKS ML COURSE - DAY 32: SKLEARN RAINFALL PREDICTION")
    print("="*70)

    # 1. Load Data
    df = generate_rainfall_dataset(n_samples=600)
    print("\nDataset Summary:")
    print(df.info())
    print("\nSample Rows:")
    print(df.head())
    
    X = df.drop(columns=['Rainfall_mm'])
    y = df['Rainfall_mm']
    
    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. Define Preprocessing Pipelines
    num_cols = ['Temperature_C', 'Humidity_pct', 'Pressure_hPa', 'WindSpeed_kmh', 'CloudCover_pct']
    cat_cols = ['Region_Type']
    
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])
    
    # 4. Construct Full Machine Learning Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0))
    ])
    
    # 5. Fit Pipeline
    print("\n" + "="*50)
    print("Fitting Pipeline on Training Data...")
    print("="*50)
    model_pipeline.fit(X_train, y_train)
    print("Pipeline training completed!")
    
    # 6. Evaluate Pipeline
    y_pred = model_pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\nTest Evaluation Metrics:")
    print(f"  Mean Absolute Error (MAE) : {mae:.3f} mm")
    print(f"  Mean Squared Error (MSE)  : {mse:.3f}")
    print(f"  Root Mean Sq Error (RMSE) : {rmse:.3f} mm")
    print(f"  R^2 Score                : {r2:.4f}")
    
    # 7. Predict on New Sample Profiles
    print("\n" + "="*50)
    print("Predicting Rainfall for New Weather Forecasts:")
    print("="*50)
    new_data = pd.DataFrame([
        {'Temperature_C': 28.5, 'Humidity_pct': 88.0, 'Pressure_hPa': 1002.0, 'WindSpeed_kmh': 22.0, 'CloudCover_pct': 90.0, 'Region_Type': 'Coastal'},
        {'Temperature_C': 35.0, 'Humidity_pct': 42.0, 'Pressure_hPa': 1020.0, 'WindSpeed_kmh': 8.0,  'CloudCover_pct': 15.0, 'Region_Type': 'Inland'},
        {'Temperature_C': 18.0, 'Humidity_pct': 95.0, 'Pressure_hPa': 995.0,  'WindSpeed_kmh': 35.0, 'CloudCover_pct': 98.0, 'Region_Type': 'Mountainous'}
    ])
    predictions = model_pipeline.predict(new_data)
    for i, pred in enumerate(predictions):
        region = new_data.iloc[i]['Region_Type']
        print(f"Forecast {i+1} ({region}): Predicted Rainfall = {pred:.2f} mm")

    # 8. Visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Actual vs Predicted
    ax1.scatter(y_test, y_pred, color='teal', alpha=0.7, edgecolors='k')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit')
    ax1.set_title(f"Actual vs Predicted Rainfall (R^2 = {r2:.3f})")
    ax1.set_xlabel("Actual Rainfall (mm)")
    ax1.set_ylabel("Predicted Rainfall (mm)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Residual Plot
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, color='coral', alpha=0.7, edgecolors='k')
    ax2.axhline(0, color='black', linestyle='--')
    ax2.set_title("Residual Plot (Errors vs Predictions)")
    ax2.set_xlabel("Predicted Rainfall (mm)")
    ax2.set_ylabel("Residual (Actual - Predicted)")
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("rainfall_prediction_results.png")
    print("\nVisualizations saved to 'rainfall_prediction_results.png'.")


if __name__ == "__main__":
    main()
