import mlflow
import argparse

def promote_model_to_production(staging_model_name, production_model_name):
    client = mlflow.tracking.MlflowClient()
    latest_version = client.search_model_versions(f"name='{staging_model_name}'")[0].version
    model_uri = f"models:/{staging_model_name}/{latest_version}"

    # Create a new registered model if it doesn't exist
    try:
        client.get_registered_model(production_model_name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(production_model_name)

    # Create a new model version
    mv = client.create_model_version(
        name=production_model_name,
        source=model_uri,
        run_id=client.get_model_version(staging_model_name, latest_version).run_id
    )

    # Get all versions of the production model
    versions = client.search_model_versions(f"name='{production_model_name}'")

    # Set alias to archive for previous champion version and champion for new version
    for version in versions:
        if version.version == mv.version:
            # Set champion alias for new version
            try:
                existing_champion_version = [v for v in versions if 'champion' in v.aliases]
                if existing_champion_version:
                    client.delete_registered_model_alias(
                        name=production_model_name,
                        alias="champion"
                    )
            except:
                pass
            client.set_registered_model_alias(
                name=production_model_name,
                alias="champion",
                version=version.version
            )
        else:
            # Set archive alias for previous versions
            alias_text = f"archive_{version.version}" 
            client.set_registered_model_alias(
                name=production_model_name,
                alias=alias_text,
                version=version.version
            )

    print(f"Model {production_model_name} version {mv.version} promoted to production and set as champion.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote Model to Production")
    parser.add_argument("--staging_model_name", type=str, required=True, help="Name of the staging model")
    parser.add_argument("--production_model_name", type=str, required=True, help="Name of the production model")
    args = parser.parse_args()

    promote_model_to_production(args.staging_model_name, args.production_model_name)

