from mlflow import MlflowClient, get_artifact_uri

from koregraph.params import MLFLOW_TRACKING_URI


print("Initializing mlflow client...")
MLFLOW_CLIENT = MlflowClient(MLFLOW_TRACKING_URI)

if __name__ == "__main__":
    print(MLFLOW_CLIENT.search_experiments())
    print(get_artifact_uri())
