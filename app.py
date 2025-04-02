
from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS  # Import CORS from flask_cors
from model import model

#Create a Flask web application.
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Mapping class labels to flower names
flower_names = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)[0]

        # Return the result as JSON
        #Return the prediction result as a JSON response.

                # Map class label to flower name
        flower_name = flower_names.get(int(prediction), "Unknown")

        result = {'prediction': int(prediction), 'flower_name': flower_name}

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000)
