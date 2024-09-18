# used_car_price_prediction

This project involves the development of a machine learning model that predicts the price of used cars based on various features. The model has been deployed as a Streamlit web app where users can input details of a car and get a real-time price prediction.

## Project Overview
The main objective of this project is to streamline the pricing process for used cars by leveraging machine learning. The model predicts the price based on features such as the city, kilometers driven, fuel type, body type, and transmission type.

## Features
- **User Inputs**: City, Kilometers Driven, Fuel Type, Body Type, and Transmission Type.
- **Real-time Predictions**: The app provides an instant price prediction based on the input car details.

## Data
The dataset used for this project was obtained from CarDekho. It includes various attributes of used cars from different cities.

### Data Preprocessing
- Missing values were handled using mean imputation for numerical columns and mode for categorical columns.
- One-hot encoding was used for categorical variables (City, Fuel Type, Body Type).
- Numerical features such as kilometers driven were normalized.

## Model Development
- The model was trained using a **Linear Regression**.
- Model evaluation metrics:
  - **Mean Absolute Error (MAE)**: 25.65008960573479
  - **Mean Squared Error (MSE)**: 76220.96748811219
  - **R-squared (R2)**: 0.9999320763225981

## How to Run the App
1. Clone this repository:
   >>>bash git clone https://github.com/bhuvanes-waran/used_car_price_prediction.git
