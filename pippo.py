import mlflow
logged_model = 'runs:/300ca9fc82924fa1808d683af2852ee9/sklearn-model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(pd.DataFrame(data))
