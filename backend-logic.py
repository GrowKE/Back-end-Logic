"""
Flask Backend Server for Housing Market Prediction

This script creates a simple web server with an API endpoint that
receives housing data, uses a pre-trained model to make a prediction,
and returns the prediction.
"""

import pandas as pd
import joblib  # For loading the trained model and scaler
from flask import Flask, request, jsonify
from flask_cors import CORS # To handle requests from the React UI

# 1. Initialize the Flask App
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) to allow our React app
# to make requests to this server.
CORS(app)

# 2. Load the Pre-trained Model and Scaler
# In a real application, these files would be created by your training script.
# We'll assume they are saved in the same directory as this server script.
try:
    model = joblib.load('housing_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl') # List of columns the model was trained on
    print("Model, scaler, and columns loaded successfully.")
except FileNotFoundError:
    print("Error: Model or scaler files not found. Please run the training script first.")
    model = None
    scaler = None
    model_columns = None

# 3. Define the Prediction API Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives house data from the UI, preprocesses it, makes a prediction,
    and returns it in JSON format.
    """
    if not model or not scaler or not model_columns:
        return jsonify({'error': 'Model not loaded, cannot make predictions.'}), 500

    try:
        # Get the JSON data sent from the React UI
        json_data = request.get_json()
        
        # Convert the incoming JSON to a pandas DataFrame
        # The [0] is because we send a single object in a list
        input_df = pd.DataFrame([json_data])

        # --- Preprocessing ---
        # This must be the *exact* same preprocessing as in the training script.
        
        # 1. One-hot encode categorical features
        # We use reindex to ensure the columns match the training data exactly,
        # filling any missing new columns with 0.
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
        
        # 2. Scale the numerical features using the *same* scaler from training
        input_scaled = scaler.transform(input_df)
        
        # --- Prediction ---
        # Use the loaded model to make a prediction
        prediction = model.predict(input_scaled)
        
        # The model returns a numpy array, so we get the first element
        output = prediction[0]

        # Return the prediction as a JSON response
        return jsonify({'predicted_price': output})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 400

# 4. Run the Flask App
if __name__ == '__main__':
    # This makes the server accessible on your local network
    # The default port is 5000, which matches our React UI's fetch request.
    app.run(host='0.0.0.0', port=5000, debug=True)

