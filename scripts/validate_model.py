import json
import mlflow
import argparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import subprocess

def load_config(env):
    config_file = f"config/{env}_config.json"
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

def load_model(model_name):
    latest_version = mlflow.tracking.MlflowClient().get_latest_versions(model_name)[0].version
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{latest_version}")
    return model

def validate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print("Validation accuracy:", accuracy)

    report = classification_report(y_val, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print(report_df)

    conf_mat = confusion_matrix(y_val, y_pred)
    conf_mat_df = pd.DataFrame(conf_mat)
    print(conf_mat_df)

    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Model")
    parser.add_argument("--env", type=str, required=True, help="Environment (dev, staging, prod)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to validation data")
    args = parser.parse_args()

    config = load_config(args.env)
    if config:
        model_name = "iris_classifier_staging"

        model = load_model(model_name)

        # Load validation data
        iris = pd.read_csv(args.data_path)
        X_val = iris.drop("target", axis=1)
        y_val = iris["target"]

        accuracy = validate_model(model, X_val, y_val)

        if accuracy > 0.9:
            print("Model accuracy is more than 0.9, promoting to production.")
            new_model_name = "iris_classifier_production"
            subprocess.run(["python", "scripts/promote_to_production.py", "--model_name", new_model_name])
        else:
            print("Model accuracy is less than 0.9, not promoting to production.")
