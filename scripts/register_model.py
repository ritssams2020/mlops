import mlflow
import argparse

def register_model(run_id, model_name):
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register Model")
    parser.add_argument("--run_id", type=str, required=True, help="Run ID")
    args = parser.parse_args()

    model_name = "iris_classifier_production"
    register_model(args.run_id, model_name)
