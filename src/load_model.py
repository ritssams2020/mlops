# load_model.py
import joblib
import numpy as np

# Load the saved model
model = joblib.load("/home/ritesh/new_MLOPs/iris_classifier.joblib")

# Define a function to make predictions
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
        # Create a numpy array with the input features
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
              
         # Make a prediction
            prediction = model.predict(features)
            species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}                   
         # Return the predicted class label
            return species_map[prediction[0]]

# Example usage
sepal_length = 5.1
sepal_width = 3.5
petal_length = 1.4
petal_width = 0.2

predicted_class = predict_iris(sepal_length, sepal_width, petal_length, petal_width)
print(f"Predicted class: {predicted_class}")
