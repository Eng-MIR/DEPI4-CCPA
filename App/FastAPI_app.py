
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('/content/Models/customer_churn_rf_model.pkl')
scaler = joblib.load('/content/Models/scaler.pkl')

# Create FastAPI app
app = FastAPI()

# =========================================================
# Input Schema
# =========================================================

class CustomerData(BaseModel):

    TotalCharges: float
    MonthlyCharges: float
    tenure: int

    Contract: int
    PaymentMethod: int
    OnlineSecurity: int
    TechSupport: int
    gender: int
    InternetService: int
    OnlineBackup: int

# =========================================================
# Home Route
# =========================================================

@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is Running"
    }

# =========================================================
# Prediction Route
# =========================================================

@app.post("/predict")
def predict(data: CustomerData):

    input_data = pd.DataFrame({
        'TotalCharges': [data.TotalCharges],
        'MonthlyCharges': [data.MonthlyCharges],
        'tenure': [data.tenure],
        'Contract': [data.Contract],
        'PaymentMethod': [data.PaymentMethod],
        'OnlineSecurity': [data.OnlineSecurity],
        'TechSupport': [data.TechSupport],
        'gender': [data.gender],
        'InternetService': [data.InternetService],
        'OnlineBackup': [data.OnlineBackup]
    })

    # Scale data
    scaled_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_data)[0]

    # Probability
    probability = model.predict_proba(scaled_data)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": float(probability)
    }
