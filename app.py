# app.py
# Import Flask for creating a web application, and import NumPy and the machine 
# learning model (model) created in model.py.
from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS  # Import CORS from flask_cors
from model import model

#Create a Flask web application.
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        # Retrieve the input data from the POST request's JSON body and convert it into a 
        # NumPy array.
        data = request.get_json(force=True)
        #The reshape(1, -1) is used to convert the 1D array into a 2D array with one row and 
        # as many columns as needed. The -1 in the second dimension means that NumPy should 
        # automatically calculate the number of columns based on the size of the original array. 
        features = np.array(data['features']).reshape(1, -1)
        # Make prediction using the model
        # Use the trained machine learning model (model) to make a prediction 
        # based on the input data.
#         It uses the predict method of the trained machine learning model (model) to make 
#         predictions based on the input features. The result is an array of predictions.

# [0]: It extracts the first element of the prediction array. In the context of this example, 
# the assumption is that the model produces a single prediction
        prediction = model.predict(features)[0]

        # Return the result as JSON
        #Return the prediction result as a JSON response.

        result = {'prediction': int(prediction)}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000)
