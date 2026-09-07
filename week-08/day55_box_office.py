"""
Day 55: Box Office Revenue Prediction - Project
Topic: Regression project predicting movie revenue
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- Generate synthetic movie data ---
np.random.seed(42)
n = 500
data = pd.DataFrame({
    'budget': np.random.lognormal(16, 1, n),  # in dollars
    'runtime': np.random.randint(80, 200, n),
    'imdb_rating': np.random.uniform(3, 10, n),
    'num_theaters': np.random.randint(500, 4500, n),
    'marketing_spend': np.random.lognormal(15, 0.8, n),
    'director_score': np.random.uniform(0, 10, n),  # director popularity
    'star_power': np.random.uniform(0, 10, n),  # cast popularity
    'sequel': np.random.choice([0, 1], n, p=[0.7, 0.3]),
})
data['revenue'] = (data['budget'] * 0.5 + 
                    data['marketing_spend'] * 0.3 +
                    data['num_theaters'] * 5000 +
                    data['imdb_rating'] * 1e6 +
                    data['star_power'] * 5e5 +
                    data['sequel'] * 2e7 +
                    np.random.normal(0, 5e6, n))
data['revenue'] = np.maximum(data['revenue'], 0)

print("Dataset Info:")
print(f"  Movies: {len(data)}")
print(f"  Revenue range: ${data['revenue'].min():,.0f} - ${data['revenue'].max():,.0f}")

# --- Feature engineering ---
data['budget_per_minute'] = data['budget'] / data['runtime']
data['marketing_ratio'] = data['marketing_spend'] / data['budget']

X = data.drop('revenue', axis=1)
y = data['revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- Train models ---
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42),
}

print("\nModel Comparison:")
print("-" * 65)
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"  {name:20s} | RMSE: ${rmse:>12,.0f} | R2: {r2:.4f} | MAE: ${mae:>12,.0f}")

# --- Feature importance ---
rf = models['Random Forest']
importances = sorted(zip(X.columns, rf.feature_importances_), key=lambda x: -x[1])
print("\nTop 5 Feature Importances:")
for name, imp in importances[:5]:
    print(f"  {name:20s}: {imp:.4f}")
