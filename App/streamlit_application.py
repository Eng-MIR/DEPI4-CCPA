import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load('Models/customer_churn_model.pkl')
scaler = joblib.load('Models/scaler.pkl')

st.title('Customer Churn Prediction Dashboard')

st.sidebar.header('Customer Information')

monthly_charges = st.sidebar.slider('Monthly Charges', 0, 200, 70)
tenure = st.sidebar.slider('Tenure', 0, 72, 24)
total_charges = st.sidebar.slider('Total Charges', 0, 10000, 2000)

input_data = pd.DataFrame({
    'MonthlyCharges': [monthly_charges],
    'tenure': [tenure],
    'TotalCharges': [total_charges]
})

# Reshape to 2D: (1 sample, n_features)
#input_array = np.array(input_data).reshape(1, -1) 

if st.button('Predict Churn'):
    
    #scaled_data = scaler.transform(input_array)
    scaled_data = scaler.transform(input_data.to_numpy())
    
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    st.subheader('Prediction Result')

    if prediction == 1:
        st.error(f'Customer is likely to churn. Probability: {probability:.2f}')
    else:
        st.success(f'Customer is not likely to churn. Probability: {probability:.2f}')
