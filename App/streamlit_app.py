import streamlit as st
import json
import pandas as pd
import xgboost as xgb

# Load feature names
with open('app/model/feature_names.json') as f:
    feature_names = json.load(f)

# Load XGBoost Booster model
model = xgb.Booster()
model.load_model('app/model/churn_model.json')

st.title("Customer Churn Prediction App")

st.write("""
This app predicts **customer churn** based on input features.  
Adjust the features below and see the predicted risk!
""")

st.sidebar.header("Customer Information")

# Numerical features
tenure = st.sidebar.slider('Tenure (months)', 0, 72, 12)
monthly_charges = st.sidebar.slider('Monthly Charges ($)', 18, 120, 70)
total_charges = st.sidebar.slider('Total Charges ($)', 0, 9000, 1000)

# Contract type
contract = st.sidebar.selectbox("Contract Type", ("Month-to-month", "One year", "Two year"))

# Internet service
internet_service = st.sidebar.selectbox("Internet Service", ("Fiber optic", "DSL", "No"))

# Online security
online_security = st.sidebar.selectbox("Online Security", ("Yes", "No"))

# Tech support
tech_support = st.sidebar.selectbox("Tech Support", ("Yes", "No"))

# Payment method
payment_method = st.sidebar.selectbox("Payment Method", ("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"))

# Base numeric features
input_data = {
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'gender': 1,  # default example (could be extended)
    'SeniorCitizen': 0,
    'Partner': 1,
    'Dependents': 0,
    'PhoneService': 1,
    'PaperlessBilling': 1,
}

# One-hot features
input_data['Contract_Month-to-month'] = 1 if contract == "Month-to-month" else 0
input_data['Contract_One year'] = 1 if contract == "One year" else 0
input_data['Contract_Two year'] = 1 if contract == "Two year" else 0

input_data['InternetService_Fiber optic'] = 1 if internet_service == "Fiber optic" else 0
input_data['InternetService_DSL'] = 1 if internet_service == "DSL" else 0
input_data['InternetService_No'] = 1 if internet_service == "No" else 0

input_data['OnlineSecurity_Yes'] = 1 if online_security == "Yes" else 0
input_data['OnlineSecurity_No'] = 1 if online_security == "No" else 0

input_data['TechSupport_Yes'] = 1 if tech_support == "Yes" else 0
input_data['TechSupport_No'] = 1 if tech_support == "No" else 0

input_data['PaymentMethod_Electronic check'] = 1 if payment_method == "Electronic check" else 0
input_data['PaymentMethod_Mailed check'] = 1 if payment_method == "Mailed check" else 0
input_data['PaymentMethod_Bank transfer (automatic)'] = 1 if payment_method == "Bank transfer (automatic)" else 0
input_data['PaymentMethod_Credit card (automatic)'] = 1 if payment_method == "Credit card (automatic)" else 0

# Fill remaining columns as 0 if needed
for col in feature_names:
    if col not in input_data:
        input_data[col] = 0

# Convert to DataFrame with correct feature order
input_df_pd = pd.DataFrame([input_data])[feature_names]

# Convert to DMatrix
dtest = xgb.DMatrix(input_df_pd)

# Predict churn probability
churn_prob = model.predict(dtest)[0]

# Display prediction
st.subheader("Predicted Churn Probability")
st.write(f"**{churn_prob:.2%} chance this customer will churn.**")

if churn_prob > 0.6:
    st.error("⚠️ High churn risk — consider immediate retention actions.")
elif churn_prob > 0.3:
    st.warning("🟠 Medium risk — consider engagement strategies.")
else:
    st.success("✅ Low churn risk.")
