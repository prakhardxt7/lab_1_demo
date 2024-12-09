import mlflow

# Set the tracking URI to the local MLFlow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Now you can log metrics, models, etc.
