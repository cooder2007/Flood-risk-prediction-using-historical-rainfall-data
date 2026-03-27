import logging
import pickle
import numpy as np
from flask import Flask, request, render_template

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Global model references
lr_model = None
knn_model = None
log_reg_model = None
scaler = None

def load_models():
    """Load pre-trained models from disk."""
    global lr_model, knn_model, log_reg_model, scaler
    try:
        with open(config.MODEL_FILE, 'rb') as f:
            lr_model, knn_model, log_reg_model, scaler = pickle.load(f)
        logging.info("Models loaded successfully from %s", config.MODEL_FILE)
    except FileNotFoundError:
        logging.error("Model file not found. Please run train.py first.")
        raise RuntimeError("Models not found. Train them using train.py.")
    except Exception as e:
        logging.error("Error loading models: %s", e)
        raise

@app.route('/')
def home():
    """Render the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request."""
    # Get form data
    try:
        rainfall = float(request.form['rainfall'])
        river_level = float(request.form['river_level'])
    except (KeyError, ValueError):
        return render_template('index.html', error="Please enter valid numeric values for rainfall and river level.")

    # Prepare input
    input_data = np.array([[rainfall, river_level]])
    try:
        input_scaled = scaler.transform(input_data)
    except Exception as e:
        logging.error("Scaling error: %s", e)
        return render_template('index.html', error="Internal error during input scaling.")

    # Predictions
    try:
        flood_level = lr_model.predict(input_scaled)[0]
        flood_risk = knn_model.predict(input_scaled)[0]
        flood_occurrence = log_reg_model.predict(input_scaled)[0]
    except Exception as e:
        logging.error("Prediction error: %s", e)
        return render_template('index.html', error="Error during prediction.")

    # Convert risk code to text
    risk_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
    flood_risk_text = risk_mapping.get(flood_risk, 'Unknown')
    occurrence_text = 'Yes' if flood_occurrence == 1 else 'No'

    return render_template('index.html',
                           rainfall=rainfall,
                           river_level=river_level,
                           flood_level=round(flood_level, 2),
                           flood_risk=flood_risk_text,
                           flood_occurrence=occurrence_text)

if __name__ == '__main__':
    load_models()
    app.run(debug=config.DEBUG)