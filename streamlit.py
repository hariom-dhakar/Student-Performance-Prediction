import streamlit as st
import pickle

#load the model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)    

#streamlit app
st.title('Student Test Score Prediction')
st.write('Enter the number of hours studied to predict the test score.')

# User Input
hours = st.number_input('Hours Studied', min_value=0.0, max_value=24.0, step=0.1)

if st.button('Predict'):
    try:
        data = [[hours]]
        scaled_data = scaler.transform(data)
        prediction = model.predict(scaled_data)
        st.success(f'Predicted Test Score: {prediction[0]:.2f}')
    except Exception as e:
        st.error(f'Error in prediction: {e}')
        
        