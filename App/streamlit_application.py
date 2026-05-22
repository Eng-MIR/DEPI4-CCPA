import streamlit as st
import pandas as pd
import joblib

# =========================
# Load Model and Scaler
# =========================

model = joblib.load('Models/customer_churn_rf_model.pkl')
scaler = joblib.load('Models/scaler.pkl')
top_features = joblib.load('Models/top_10_features.pkl')

# =========================
# Streamlit Page Config
# =========================

st.set_page_config(
    page_title='Customer Churn Prediction',
    page_icon='📊',
    layout='centered'
)

st.title('📊 Customer Churn Prediction Dashboard')

st.markdown("""
Predict whether a customer is likely to churn based on customer information.
""")

# =========================
# Sidebar Inputs
# =========================

st.sidebar.header('Customer Information')

# Gender
gender_option = st.sidebar.selectbox(
    'Gender',
    ['Female', 'Male']
)

gender = 0 if gender_option == 'Female' else 1

# Senior Citizen
senior_option = st.sidebar.selectbox(
    'Senior Citizen',
    ['No', 'Yes']
)

SeniorCitizen = 1 if senior_option == 'Yes' else 0

# Partner
partner_option = st.sidebar.selectbox(
    'Partner',
    ['No', 'Yes']
)

Partner = 1 if partner_option == 'Yes' else 0

# Dependents
dependents_option = st.sidebar.selectbox(
    'Dependents',
    ['No', 'Yes']
)

Dependents = 1 if dependents_option == 'Yes' else 0

# Tenure
tenure = st.sidebar.slider(
    'Tenure (Months)',
    min_value=0,
    max_value=72,
    value=12
)

# Online Security
security_option = st.sidebar.selectbox(
    'Online Security',
    ['No', 'Yes']
)

OnlineSecurity = 1 if security_option == 'Yes' else 0

# Tech Support
tech_support_option = st.sidebar.selectbox(
    'Tech Support',
    ['No', 'Yes']
)

TechSupport = 1 if tech_support_option == 'Yes' else 0

# Contract Type
contract_option = st.sidebar.selectbox(
    'Contract Type',
    ['Month-to-Month', 'One Year', 'Two Year']
)

contract_mapping = {
    'Month-to-Month': 0,
    'One Year': 1,
    'Two Year': 2
}

Contract = contract_mapping[contract_option]

# Monthly Charges
MonthlyCharges = st.sidebar.slider(
    'Monthly Charges',
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

# Total Charges
TotalCharges = st.sidebar.slider(
    'Total Charges',
    min_value=0.0,
    max_value=10000.0,
    value=2000.0
)

# =========================
# Create Input DataFrame
# =========================

input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [SeniorCitizen],
    'Partner': [Partner],
    'Dependents': [Dependents],
    'tenure': [tenure],
    'OnlineSecurity': [OnlineSecurity],
    'TechSupport': [TechSupport],
    'Contract': [Contract],
    'MonthlyCharges': [MonthlyCharges],
    'TotalCharges': [TotalCharges]
})

# Ensure correct column order
#input_data = input_data[top_features]

# =========================
# Display Input Summary
# =========================

st.subheader('Customer Information Summary')

st.dataframe(input_data)

# =========================
# Prediction Button
# =========================

if st.button('Predict Churn'):

    # Scale data
    scaled_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_data)[0]

    # Prediction probability
    probability = model.predict_proba(scaled_data)[0][1]

    st.subheader('Prediction Result')

    # Display result
    if prediction == 1:

        st.error(
            f'⚠️ Customer is likely to churn.\n\n'
            f'Churn Probability: {probability:.2%}'
        )

    else:

        st.success(
            f'✅ Customer is not likely to churn.\n\n'
            f'Churn Probability: {probability:.2%}'
        )

    # Progress Bar
    st.progress(float(probability))

# =========================
# Footer
# =========================

st.markdown('---')
st.markdown(
    'Built using Streamlit, Scikit-learn, and MLflow'
)
