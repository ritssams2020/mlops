import mlflow

def validate_model():
    # Define the model URI
    model_uri = 'runs:/ea3abbc8b0224ebcb1a9eed8cb04149b/model'

    # Load the model
    pyfunc_model = mlflow.pyfunc.load_model(model_uri)

    # Get the input example
    input_data = pyfunc_model.input_example

    # Verify the model with the provided input data
    result = mlflow.models.predict(
        model_uri=model_uri,
        input_data=input_data,
        env_manager="uv",
    )

    print(result)

if __name__ == "__main__":
    validate_model()
