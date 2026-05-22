import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('customer_churn_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Customer Churn Prediction")

st.title("Customer Churn Prediction Dashboard")

st.sidebar.header("Customer Information")

# -----------------------------
# User Inputs
# -----------------------------

gender = st.sidebar.selectbox("Gender", [0, 1])

SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])

Partner = st.sidebar.selectbox("Partner", [0, 1])

Dependents = st.sidebar.selectbox("Dependents", [0, 1])

tenure = st.sidebar.slider("Tenure", 0, 72, 12)

PhoneService = st.sidebar.selectbox("Phone Service", [0, 1])

MultipleLines = st.sidebar.selectbox("Multiple Lines", [0, 1, 2])

InternetService = st.sidebar.selectbox("Internet Service", [0, 1, 2])

OnlineSecurity = st.sidebar.selectbox("Online Security", [0, 1, 2])

OnlineBackup = st.sidebar.selectbox("Online Backup", [0, 1, 2])

DeviceProtection = st.sidebar.selectbox("Device Protection", [0, 1, 2])

TechSupport = st.sidebar.selectbox("Tech Support", [0, 1, 2])

StreamingTV = st.sidebar.selectbox("Streaming TV", [0, 1, 2])

StreamingMovies = st.sidebar.selectbox("Streaming Movies", [0, 1, 2])

Contract = st.sidebar.selectbox("Contract", [0, 1, 2])

PaperlessBilling = st.sidebar.selectbox("Paperless Billing", [0, 1])

PaymentMethod = st.sidebar.selectbox("Payment Method", [0, 1, 2, 3])

MonthlyCharges = st.sidebar.slider("Monthly Charges", 0.0, 200.0, 70.0)

TotalCharges = st.sidebar.slider("Total Charges", 0.0, 10000.0, 2000.0)

# customerID feature placeholder
customerID = st.sidebar.number_input("Customer ID", value=1)

# -----------------------------
# Create Input DataFrame
# -----------------------------

input_data = pd.DataFrame({
    'customerID': [customerID],
    'gender': [gender],
    'SeniorCitizen': [SeniorCitizen],
    'Partner': [Partner],
    'Dependents': [Dependents],
    'tenure': [tenure],
    'PhoneService': [PhoneService],
    'MultipleLines': [MultipleLines],
    'InternetService': [InternetService],
    'OnlineSecurity': [OnlineSecurity],
    'OnlineBackup': [OnlineBackup],
    'DeviceProtection': [DeviceProtection],
    'TechSupport': [TechSupport],
    'StreamingTV': [StreamingTV],
    'StreamingMovies': [StreamingMovies],
    'Contract': [Contract],
    'PaperlessBilling': [PaperlessBilling],
    'PaymentMethod': [PaymentMethod],
    'MonthlyCharges': [MonthlyCharges],
    'TotalCharges': [TotalCharges]
})

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Churn"):

    # Scale data
    scaled_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_data)[0]

    # Probability
    probability = model.predict_proba(scaled_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"Customer is likely to churn.\n\n"
            f"Churn Probability: {probability:.2%}"
        )
    else:
        st.success(
            f"Customer is not likely to churn.\n\n"
            f"Churn Probability: {probability:.2%}"
        )

    # Display input data
    st.subheader("Customer Input Data")
    st.dataframe(input_data)
