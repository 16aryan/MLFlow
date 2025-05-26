import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import logging
import json

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Set the DagsHub MLflow tracking URI
    mlflow.set_tracking_uri("https://dagshub.com/16aryan/MLFlow.mlflow")

    # Authentication (Optional but recommended)
    os.environ['MLFLOW_TRACKING_USERNAME'] = "<16aryan"
    os.environ['MLFLOW_TRACKING_PASSWORD'] = "8c0b77ee9ef83414e9cb22e106b904ad9c957f77"

    # Load dataset
    csv_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception("Dataset loading failed: %s", e)
        sys.exit(1)

    # Split dataset
    train, test = train_test_split(data)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # Begin MLflow run with a custom name
    with mlflow.start_run(run_name="ElasticNet-Wine-Model"):
        mlflow.set_tag("project", "Wine Quality Regression")
        mlflow.set_tag("developer", "<your-name>")

        # Train model
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(train_x, train_y)
        predictions = model.predict(test_x)

        # Evaluate model
        rmse, mae, r2 = eval_metrics(test_y, predictions)

        # Log parameters & metrics
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log model
        input_example = train_x.head(5)
        tracking_type = urlparse(mlflow.get_tracking_uri()).scheme

        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="ElasticnetWineModel",
            input_example=input_example,
        )

        # Log artifacts for the UI
        sample_data_path = "sample_input.csv"
        train_x.head(10).to_csv(sample_data_path, index=False)
        mlflow.log_artifact(sample_data_path)

        metrics_path = "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({"rmse": rmse, "mae": mae, "r2": r2}, f)
        mlflow.log_artifact(metrics_path)

        print(f"Logged to MLflow tracking URI: {mlflow.get_tracking_uri()}")