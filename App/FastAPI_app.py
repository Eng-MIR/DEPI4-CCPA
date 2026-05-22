from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load('/content/Models/customer_churn_rf_model.pkl')
scaler = joblib.load('/content/Models/scaler.pkl')

class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

@app.get('/')
def home():
    return {'message': 'Customer Churn Prediction API'}

@app.post('/predict')
def predict(data: CustomerData):

    input_data = pd.DataFrame([data.dict()])

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    return {
        'prediction': int(prediction),
        'churn_probability': float(probability)
    }
