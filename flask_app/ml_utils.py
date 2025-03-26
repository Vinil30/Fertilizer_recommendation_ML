import pickle
import pandas as pd
import logging
import numpy as np
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
# Load ML Model and Preprocessing Tools
model_path = os.path.join(base_dir,"models","xgboost_irrigation_model.pkl")
scaler_path = os.path.join(base_dir,"models","minmax_scaler.pkl")
encoder_path = os.path.join(base_dir,"models","onehot_encoder.pkl")
label_encoder_path = os.path.join(base_dir,"models","label_encoder.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
with open(encoder_path, "rb") as f:
    encoder = pickle.load(f)
with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

def predict_irrigation(input_df):
    try:
        logging.info(f"Received input for prediction:\n{input_df}")

        # Define your features
        numerical_features = ["Rainfall (mm)", "Soil Moisture (%)", "Temperature ", "Evaporation Rate (mm/day)"]
        categorical_features = ["Crop Type", "Water Availability", "Soil Type"]

        # One-hot encode categorical features
        input_encoded = encoder.transform(input_df[categorical_features])
        input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(categorical_features))

        logging.info(f"One-hot encoded categorical features:\n{input_encoded_df}")

        # Combine with numerical columns
        input_combined = pd.concat([input_encoded_df, input_df[numerical_features].copy()], axis=1)

        logging.info(f"Combined features before scaling:\n{input_combined}")

        # Scale numerical columns
        input_combined[numerical_features] = scaler.transform(input_combined[numerical_features])

        logging.info(f"Final input for prediction after scaling:\n{input_combined}")

        # Prediction (it returns the encoded label)
        prediction_encoded = model.predict(input_combined)

        logging.info(f"Encoded prediction result: {prediction_encoded}")

        # Inverse transform to get the original label
        prediction_label = label_encoder.inverse_transform(prediction_encoded)

        logging.info(f"Decoded prediction label: {prediction_label}")

        # Return the label as string
        return prediction_label[0]

    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        logging.error(error_msg)
        return error_msg