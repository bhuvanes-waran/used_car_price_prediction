import streamlit as st
import numpy as np
import pandas as pd
from sklearn.externals import joblib

# Load the trained model from the file
model = joblib.load('car_price_model.pkl')

# List of columns used during training
columns = ['km', 'ft_Diesel', 'ft_Electric', 'ft_Lpg', 'ft_Petrol',
       'bt_Convertibles', 'bt_Coupe', 'bt_Hatchback', 'bt_Hybrids', 'bt_MUV',
       'bt_Minivans', 'bt_Pickup Trucks', 'bt_SUV', 'bt_Sedan', 'bt_Wagon',
       'transmission_Manual', 'city_chennai', 'city_delhi', 'city_hyderabad',
       'city_jaipur', 'city_kolkata']

# Title and description for the Streamlit app
st.title("Used Car Price Prediction App")
st.write("This application predicts the price of a used car based on its features. Provide the car details below.")

# Create input fields for user input
city = st.selectbox("Select City", ["Chennai", "Delhi", "Hyderabad", "Jaipur", "Kolkata"])
kilometers_driven = st.number_input("Enter Kilometers Driven", value=50000)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Lpg"])
body_type = st.selectbox("Body Type", ["Convertibles", "Coupe", "Hatchback", "Hybrids", "MUV", "Minivans", "Pickup Trucks", "SUV", "Sedan", "Wagon"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])

# Prepare input data for prediction
input_data = pd.DataFrame({
    'ft_' + fuel_type : [1],
    'bt_' + body_type : [1],
    'km' : [kilometers_driven],
    'transmission_' + transmission : [1],
    'city_' + city : [1]
})

# One-hot encode the 'City', 'Fuel Type', and 'Body Type' columns
input_data = pd.get_dummies(input_data)

# Add missing columns to match the model's training format
missing_cols = set(columns) - set(input_data.columns)
for col in missing_cols:
    input_data[col] = 0

# Reorder columns to match the training data
input_data = input_data[columns]

# Make predictions using the trained model
predicted_price = model.predict(input_data)

# Display the predicted price to the user
st.write(f"Predicted Price: ₹{predicted_price[0]:,.2f}")
