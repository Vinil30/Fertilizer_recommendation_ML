from flask import Blueprint, request, jsonify
import pandas as pd
import logging
import traceback
from flask_app.ml_utils import predict_irrigation

# Setup logging
log_file = "notebooks/api.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Create Blueprint
routes = Blueprint('routes', __name__)

@routes.route('/predict-irrigation', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request body
        data = request.get_json()
        logging.info(f"POST /predict-irrigation - Raw JSON Data: {data}")

        # Convert JSON data to DataFrame with expected columns
        input_data = pd.DataFrame([{
    'Crop Type': data.get('crop'),
    'Water Availability': data.get('water'), # Make sure it's a string like 'Low', 'Medium', etc.
    'Rainfall (mm)': float(data.get('rainfall')),
    'Soil Type': data.get('soil'),
    'Soil Moisture (%)': float(data.get('soil_moisture')),
    'Temperature ': float(data.get('temperature')),
    'Evaporation Rate (mm/day)': float(data.get('evaporation'))
}])

        logging.info(f"POST /predict-irrigation - Input DataFrame:\n{input_data}")

        # Call the prediction function from ml_utils.py
        prediction = predict_irrigation(input_data)

        logging.info(f"POST /predict-irrigation - Prediction Result: {prediction}")

        # Return the prediction as JSON
        return jsonify({
            "success": True,
            "prediction": prediction
        })

    except Exception as e:
        error_message = traceback.format_exc()
        logging.error(f"POST /predict-irrigation - Error:\n{error_message}")

        # Return an error message as JSON
        return jsonify({
            "success": False,
            "error": "Prediction failed. Please check your input values."
        }), 500
