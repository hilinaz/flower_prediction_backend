from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# ------------------------------
# 1. Initialize FastAPI app
# ------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# 2. Load Models & Label Encoder
# ------------------------------
logistic_model = joblib.load("backend/models/logical_regression/logistic_regression_model.pkl")
decision_tree_model = joblib.load("backend/models/decision_tree/decision_tree_model.pkl")
label_encoder = joblib.load("backend/models/label_encoder.pkl")

# ------------------------------
# 3. Input Schema
# ------------------------------
class PredictionRequest(BaseModel):
    features: list[float]   # [sepal_length, sepal_width, petal_length, petal_width]
    model_type: str         # "logistic" or "tree"

# ------------------------------
# 4. Prediction Endpoint
# ------------------------------
@app.post("/predict")
def predict(data: PredictionRequest):
    X = np.array(data.features).reshape(1, -1)

    if data.model_type == "logistic":
        pred = logistic_model.predict(X)[0]
        probs = logistic_model.predict_proba(X)[0].tolist()

    elif data.model_type == "tree":
        pred = decision_tree_model.predict(X)[0]
        probs = decision_tree_model.predict_proba(X)[0].tolist()

    else:
        raise HTTPException(status_code=400, detail="Invalid model_type")

    # Convert numeric prediction back to species name
    species = label_encoder.inverse_transform([pred])[0]

    return {
        "model_used": data.model_type,
        "prediction": int(pred),
        "species": species,
        "probabilities": probs
    }
