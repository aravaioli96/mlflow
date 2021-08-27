import numpy as np
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn
import pandas as pd



if __name__ == "__main__":
    file = "mlflow/data.csv"
    df = pd.read_csv(file, sep = ',') 

    X = np.array(df.X).reshape(-1, 1)
    y = np.array(df.Y)
    lr = LogisticRegression()
    lr.fit(X, y)
    score = lr.score(X, y)
    print("Score: %s" % score)
    mlflow.log_metric("score", score)
    mlflow.log_param("alberto", 1)
    mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="sklearn-model",
        registered_model_name="modello_prova"
    )
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
