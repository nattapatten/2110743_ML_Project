from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd
import numpy as np

# Define the folder where your model and scaler are saved (update as needed)
base_folder = r'D:\CU_Homework\Semester_2023_2\2110743 Machine Learning\2110743_ML_Project\Algorithms\Trained_Model\Logistic_Regression\2024-10-04\0954'

# Load the saved model and scaler
model_path = os.path.join(base_folder, 'best_logistic_model.pkl')
scaler_path = os.path.join(base_folder, 'scaler.pkl')

loaded_model = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)

# Initialize the Flask app
app = Flask(__name__)

# Define the API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.json['data']

    # Convert the input data to a numpy array and reshape for a single sample
    data_array = np.array(data).reshape(1, -1)

    # Apply the saved scaler to the new data
    scaled_data = loaded_scaler.transform(data_array)

    # Use the loaded model to make predictions
    prediction = loaded_model.predict(scaled_data)
    probability = loaded_model.predict_proba(scaled_data)[:, 1]  # Probability for positive class

    # Return the prediction and probability as a JSON response
    return jsonify({
        'prediction': int(prediction[0]),
        'probability': float(probability[0])
    })

# Run the Flask app on localhost with the custom port 848484
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8484, debug=True)
