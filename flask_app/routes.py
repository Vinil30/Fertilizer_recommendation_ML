from flask import Blueprint, request, jsonify
import pandas as pd
import logging
import traceback
from flask_app.ml_utils import predictFertilizer

# Setup logging
log_file = "notebooks/api.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Create Blueprint
routes = Blueprint('routes', __name__)

def safe_float_convert(value, default=0.0):
    """Safely convert a value to float with default fallback"""
    try:
        return float(value) if value not in [None, ''] else default
    except (ValueError, TypeError):
        return default

@routes.route('/predict-fertilizer', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request body
        data = request.get_json()
        logging.info(f"POST /predict-fertilizer - Raw JSON Data: {data}")

        # Convert JSON data to DataFrame with expected columns
        input_data = pd.DataFrame([{
            'Crop': data.get('Crop'),
            'Soil Type': data.get('Soil_Type'), # Make sure it's a string like 'Low', 'Medium', etc.
            'Nitrogen (N)': float(data.get('Nitrogen')),
            'Phosphorus (P)': float(data.get('Phosphorus')),
            'Potassium (K)': float(data.get('Potassium')),
            'Crop Growth Stage': data.get('Crop_Growth_Stage'),
            'Rainfall (mm)': float(data.get('Rainfall')),
            'Temperature (°C)': safe_float_convert(data.get('Temperature')), # type: ignore
            'Irrigation Availability': data.get('Irrigation_Availability'),
            'Past Yield (tons/ha)': float(data.get('Past_Yield')),
            'Pest/Disease': data.get('Pest_Disease'),
            'Region': data.get('Region')
        }])

        logging.info(f"POST /predict-fertilizer - Input DataFrame:\n{input_data}")

        # Call the prediction function from ml_utils.py
        prediction = predictFertilizer(input_data)

        logging.info(f"POST /predict-fertilizer - Prediction Result: {prediction}")

        # Return the prediction as JSON
        return jsonify({
            "success": True,
            "prediction": prediction
        })

    except Exception as e:
        error_message = traceback.format_exc()
        logging.error(f"POST /predict-fertilizer - Error:\n{error_message}")

        # Return an error message as JSON
        return jsonify({
            "success": False,
            "error": "Prediction failed. Please check your input values."
        }), 500