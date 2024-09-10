import streamlit as st
import pandas as pd
import joblib

st.title('Churn Prediction App')

# Input fields
CreditScore = st.number_input('Credit Score', min_value=0)
Age = st.number_input('Age', min_value=18, max_value=100)
Balance = st.number_input('Balance', min_value=0.0)
EstimatedSalary = st.number_input('Estimated Salary', min_value=0.0)
NumOfProducts = st.slider('Number of Products', min_value=1, max_value=4, step=1)
Tenure = st.slider('Tenure', min_value=0, max_value=10, step=1)

# Categorical features
Geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])  
Gender = st.selectbox('Gender', ['Male', 'Female'])  
HasCrCard = st.selectbox('Has Credit Card?', ['Yes', 'No']) 
IsActiveMember = st.selectbox('Is Active Member?', ['Yes', 'No']) 

mapping = {'Yes': 1, 'No': 0}
HasCrCard_num = mapping[HasCrCard]
IsActiveMember_num = mapping[IsActiveMember]

# Create a DataFrame from the inputs
input_df = pd.DataFrame({
    'CreditScore': [CreditScore],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard_num],
    'IsActiveMember': [IsActiveMember_num],
    'EstimatedSalary': [EstimatedSalary],
    'Geography': [Geography],
    'Gender': [Gender],
})

# Load the pre-fitted encoders, scaler and model
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')
model = joblib.load('trained_model.pkl')

# Encode categorical features
encoded_features = encoder.transform(input_df[['Geography', 'Gender']])

# Convert encoded features to DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

# Concatenate encoded features with numerical features
preprocessed_df = pd.concat([
    input_df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']],
    encoded_df
], axis=1)

scaled_features = scaler.transform(preprocessed_df)
preprocessed_df = pd.DataFrame(scaled_features, columns=preprocessed_df.columns)

# Add a button for prediction
if st.button('Predict'):
    prediction = model.predict(preprocessed_df)

    if prediction == 0:
        output_text = "No"
    else:
        output_text = "Yes"

    # Display the prediction result with a success message
    st.success(f'The prediction is: {output_text}')