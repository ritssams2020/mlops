#Prediction Utility Code

from flask import Flask, request, jsonify
import mlflow.pyfunc
import logging

app = Flask(__name__)

# Load the model
model_name = "iris_classifier_production"
model_version = "1"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

# Set up logging
logging.basicConfig(filename='prediction.log', level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data
        input_data = request.json

        # Make prediction
        prediction = model.predict(input_data)

        # Log the prediction
        logging.info(f'Prediction made: {prediction}')

        # Return the prediction
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        # Log the error
        logging.error(f'Error making prediction: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
