"""
Day 70: Flask, FastAPI and MLOps
Topic: FastAPI model serving endpoint

To run: uvicorn day70_fastapi_mlops:app --reload
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import warnings
warnings.filterwarnings('ignore')

# --- Train model (in production, load from saved artifact) ---
iris = load_iris()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(iris.data, iris.target)

# --- FastAPI app ---
app = FastAPI(title="Iris Classifier API", version="1.0")

# --- Request schema ---
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class BatchRequest(BaseModel):
    features: List[IrisFeatures]

# --- Endpoints ---
@app.get("/")
def root():
    return {"status": "healthy", "model": "RandomForest", "classes": list(iris.target_names)}

@app.get("/info")
def model_info():
    return {
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "features": iris.feature_names.tolist(),
        "classes": iris.target_names.tolist(),
        "training_samples": len(iris.data)
    }

@app.post("/predict")
def predict(features: IrisFeatures):
    X = np.array([[features.sepal_length, features.sepal_width,
                   features.petal_length, features.petal_width]])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    return {
        "prediction": iris.target_names[prediction],
        "prediction_id": int(prediction),
        "probabilities": {iris.target_names[i]: float(prob) for i, prob in enumerate(probabilities)}
    }

@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    if len(request.features) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 samples per request")
    X = np.array([[f.sepal_length, f.sepal_width, f.petal_length, f.petal_width] for f in request.features])
    predictions = model.predict(X)
    return {
        "predictions": [iris.target_names[p] for p in predictions],
        "n_samples": len(predictions)
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# --- MLOps concepts (documentation) ---
print("""
MLOps Pipeline:
1. Data: version with DVC, validate schema
2. Training: MLflow for experiment tracking
   - mlflow.start_run()
   - mlflow.log_metric('accuracy', 0.95)
   - mlflow.sklearn.log_model(model, 'model')
3. CI/CD: GitHub Actions
   - Train on PR, validate metrics
   - Auto-deploy if metrics improve
4. Deployment: FastAPI + Docker
   - docker build -t iris-api .
   - docker run -p 8000:8000 iris-api
5. Monitoring: track drift, latency, accuracy
   - Evidently for data drift
   - Prometheus + Grafana for metrics
6. Retraining: trigger when drift detected

FastAPI advantages:
- Automatic OpenAPI/Swagger docs at /docs
- Async support for high concurrency
- Type validation via Pydantic
- Production-ready with Uvicorn

Example usage:
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
""")
