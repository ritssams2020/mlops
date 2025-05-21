import json
import mlflow
import numpy as np
import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_config(config_file):
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON data in '{config_file}': {e}")
        return None

def train_model(config):
    if config is None:
        print("Error: Unable to load configuration.")
        return

    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment("Iris Classifier")

    with mlflow.start_run():
        iris = load_iris()
        X = iris.data
        y = iris.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)

        # Create an input example
        input_example = X_train[:1]

        # Log model with input example
        mlflow.sklearn.log_model(model, "model", input_example=input_example)

        # Validate model
        validate_model(model, X_test, y_test)

        print("Model training completed.")

def validate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print("Validation accuracy:", accuracy)
    mlflow.log_metric("val_accuracy", accuracy)

    print("Classification report:")
    report = classification_report(y_val, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv("classification_report.csv", index=True)
    mlflow.log_artifact("classification_report.csv")
    print(report_df)

    print("Confusion matrix:")
    conf_mat = confusion_matrix(y_val, y_pred)
    conf_mat_df = pd.DataFrame(conf_mat)
    conf_mat_df.to_csv("confusion_matrix.csv", index=False, header=False)
    mlflow.log_artifact("confusion_matrix.csv")
    print(conf_mat_df)

def main():
    env = "DEV"
    config_file = f"config/{env.lower()}_config.json"
    config = load_config(config_file)
    print(config)
    train_model(config)

if __name__ == "__main__":
    main()
