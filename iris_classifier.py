import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Start MLflow experiment
mlflow.set_experiment("Iris Classifier")

with mlflow.start_run():
        # Load Iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target

        # Split dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Logistic Regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Make predictions and evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log model performance and parameters
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("max_iter", model.max_iter)

        # Save trained model
        joblib.dump(model, "iris_classifier.joblib")
        mlflow.log_artifact("iris_classifier.joblib")

        print(f"Model Accuracy: {accuracy:.2f}")
