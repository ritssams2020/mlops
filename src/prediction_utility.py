#prediction_utility.py
import joblib
import numpy as np
import os
from dotenv import load_dotenv
import json

load_dotenv()

env = os.getenv("ENVIRONMENT")
config_file = f"config/{env.lower()}_config.json"

with open(config_file, "r") as f:
        config = json.load(f)   

class IrisPredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

    def predict(self, sepal_length, sepal_width, petal_length, petal_width):
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = self.model.predict(features)
        return self.species_map[prediction[0]]

def main():
    model_path = "iris_classifier.joblib"
    predictor = IrisPredictor(model_path)

    print("Iris Species Prediction Utility")
    print("--------------------------------")

    while True:
        sepal_length = float(input("Enter sepal length (cm): "))
        sepal_width = float(input("Enter sepal width (cm): "))
        petal_length = float(input("Enter petal length (cm): "))
        petal_width = float(input("Enter petal width (cm): "))

        predicted_species = predictor.predict(sepal_length, sepal_width, petal_length, petal_width)
        print(f"Predicted species: {predicted_species}")

        cont = input("Do you want to continue? (y/n): ")
        if cont.lower() != "y":
            break

if __name__ == "__main__":
    main()
