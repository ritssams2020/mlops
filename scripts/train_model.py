import json
import mlflow
import argparse
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

def train_model(config):
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["experiment_name"])

    with mlflow.start_run() as run:
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

        # Save the run ID
        run_id = run.info.run_id

        return run_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Iris Classifier")
    parser.add_argument("--env", type=str, required=True, help="Environment (dev, staging, prod)")
    args = parser.parse_args()

    config = load_config(args.env)
    if config:
        run_id = train_model(config)
        print(f"Run ID: {run_id}")
