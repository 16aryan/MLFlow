# MLFlow

# DAGSHUB
import dagshub
dagshub.init(repo_owner='16aryan', repo_name='MLFlow', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)


  export MLFLOW_TRACKING_URI : "https://dagshub.com/16aryan/MLFlow.mlflow"
  export MLFLOW_TRACKING_USERNAME=16aryan
  export MLFLOW_TRACKING_PASSWORD=8c0b77ee9ef83414e9cb22e106b904ad9c957f77

  Region: us-east-1
  Endpoint URL: https://dagshub.com/api/v1/repo-