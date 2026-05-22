import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load('Models/customer_churn_rf_model.pkl')
scaler = joblib.load('Models/scaler.pkl')

st.set_page_config(
    page_title='Customer Churn Prediction Application',
    layout='centered'
)

# Display Title
st.title('Customer Churn Prediction Application')
st.markdown('This app predicts Customer Churn based on Top features.')
st.markdown('Enter customer information to predict churn probability.')

# Sidebar Inputs
st.sidebar.header('Customer Information')

# Numerical Features
TotalCharges = st.sidebar.slider(
    'Total Charges',
    min_value=0.0,
    max_value=10000.0,
    value=2000.0
)

MonthlyCharges = st.sidebar.slider(
    'Monthly Charges',
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

tenure = st.sidebar.slider(
    'Tenure (Months)',
    min_value=0,
    max_value=72,
    value=24
)

# Categorical Features
contract_option = st.sidebar.selectbox(
    'Contract Type',
    ['Month-to-month', 'One year', 'Two year']
)
payment_option = st.sidebar.selectbox(
    'Payment Method',
    [
        'Electronic check',
        'Mailed check',
        'Bank transfer (automatic)',
        'Credit card (automatic)'
    ]
)
online_security_option = st.sidebar.selectbox(
    'Online Security',
    ['No', 'Yes']
)
tech_support_option = st.sidebar.selectbox(
    'Tech Support',
    ['No', 'Yes']
)
gender_option = st.sidebar.selectbox(
    'Gender',
    ['Female', 'Male']
)
internet_option = st.sidebar.selectbox(
    'Internet Service',
    ['DSL', 'Fiber optic', 'No']
)
online_backup_option = st.sidebar.selectbox(
    'Online Backup',
    ['No', 'Yes']
)

# Encoding Mappings
contract_mapping = {
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2
}
payment_mapping = {
    'Electronic check': 0,
    'Mailed check': 1,
    'Bank transfer (automatic)': 2,
    'Credit card (automatic)': 3
}
binary_mapping = {
    'No': 0,
    'Yes': 1
}
gender_mapping = {
    'Female': 0,
    'Male': 1
}
internet_mapping = {
    'DSL': 0,
    'Fiber optic': 1,
    'No': 2
}

# Convert Inputs to Encoded Values
Contract = contract_mapping[contract_option]
PaymentMethod = payment_mapping[payment_option]
OnlineSecurity = binary_mapping[online_security_option]
TechSupport = binary_mapping[tech_support_option]
gender = gender_mapping[gender_option]
InternetService = internet_mapping[internet_option]
OnlineBackup = binary_mapping[online_backup_option]

# Create Input DataFrame
# IMPORTANT: Must match exact training feature order
input_data = pd.DataFrame({
    'TotalCharges': [TotalCharges],
    'MonthlyCharges': [MonthlyCharges],
    'tenure': [tenure],
    'Contract': [Contract],
    'PaymentMethod': [PaymentMethod],
    'OnlineSecurity': [OnlineSecurity],
    'TechSupport': [TechSupport],
    'gender': [gender],
    'InternetService': [InternetService],
    'OnlineBackup': [OnlineBackup]
})

# Display Entered Data
st.subheader('Input Summary')
st.dataframe(input_data)

# Prediction
if st.button('Predict Churn'):
    # Scale input data
    scaled_data = scaler.transform(input_data)
    # Predict
    prediction = model.predict(scaled_data)[0]
    # Predict probability
    probability = model.predict_proba(scaled_data)[0][1]
    st.subheader('Prediction Result')

    if prediction == 1:
        st.error(
            f'Customer is likely to churn.\n\n'
            f'Churn Probability: {probability:.2%}'
        )
    else:
        st.success(
            f'Customer is not likely to churn.\n\n'
            f'Churn Probability: {probability:.2%}'
        )

    # Probability Progress Bar
    st.progress(float(probability))
