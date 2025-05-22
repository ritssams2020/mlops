import mlflow
import argparse

def deploy_model(model_name, model_version):
    # Deploy the model
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

    # Transition the model to the "Production" stage
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Production"
    )

    print(f"Model {model_name} version {model_version} deployed to production")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy Model")
    parser.add_argument("--model_name", type=str, required=True, help="Model Name")
    parser.add_argument("--model_version", type=str, required=True, help="Model Version")
    args = parser.parse_args()

    deploy_model(args.model_name, args.model_version)
