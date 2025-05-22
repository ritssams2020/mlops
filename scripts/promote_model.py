import mlflow

client = mlflow.tracking.MlflowClient()
source_model_name = "iris_classifier"
target_model_name = "iris_classifier_staging"

# Get the latest version of the source model
latest_version = client.get_latest_versions(source_model_name)[0].version

# Create a new registered model if it doesn't exist
try:
    client.get_registered_model(target_model_name)
except mlflow.exceptions.MlflowException:
    client.create_registered_model(target_model_name)

# Create a new model version
mv = client.create_model_version(
    name=target_model_name,
    source=f"models:/{source_model_name}/{latest_version}",
    run_id=client.get_model_version(source_model_name, latest_version).run_id
)

# Transition the model version to staging
client.transition_model_version_stage(
    name=target_model_name,
    version=mv.version,
    stage="Staging"
)
