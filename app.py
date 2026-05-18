
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model
with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load model columns
with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# Streamlit App Title
st.title("🚗 Car Price Prediction App")

st.write("Enter car details below:")

# ---------------------------------------------------
# USER INPUTS
# ---------------------------------------------------

present_price = st.number_input(
    "Present Price (in lakhs)",
    min_value=0.0,
    max_value=100.0,
    value=5.0
)

kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    max_value=500000,
    value=30000
)

owner = st.number_input(
    "Number of Previous Owners",
    min_value=0,
    max_value=5,
    value=0
)

car_age = st.number_input(
    "Car Age (Years)",
    min_value=0,
    max_value=30,
    value=5
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG"]
)

seller_type = st.selectbox(
    "Seller Type",
    ["Dealer", "Individual"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

# ---------------------------------------------------
# CREATE INPUT DATAFRAME
# ---------------------------------------------------

input_dict = {
    'Present_Price': present_price,
    'Kms_Driven': kms_driven,
    'Owner': owner,
    'Car_Age': car_age,
}

# Create dataframe
input_df = pd.DataFrame([input_dict])

# ---------------------------------------------------
# HANDLE ONE-HOT ENCODING
# ---------------------------------------------------

# Add categorical columns manually
input_df['Fuel_Type_Diesel'] = 1 if fuel_type == 'Diesel' else 0
input_df['Fuel_Type_Petrol'] = 1 if fuel_type == 'Petrol' else 0

input_df['Seller_Type_Individual'] = (
    1 if seller_type == 'Individual' else 0
)

input_df['Transmission_Manual'] = (
    1 if transmission == 'Manual' else 0
)

# ---------------------------------------------------
# MATCH TRAINING COLUMNS
# ---------------------------------------------------

for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Ensure correct column order
input_df = input_df[model_columns]

# ---------------------------------------------------
# SCALE FEATURES
# ---------------------------------------------------

scaled_features = scaler.transform(input_df)

# ---------------------------------------------------
# PREDICT
# ---------------------------------------------------

if st.button("Predict Car Price"):

    prediction = model.predict(scaled_features)

    st.success(
        f"🚘 Estimated Car Selling Price: {prediction[0]:,.2f} Lakhs"
    )
