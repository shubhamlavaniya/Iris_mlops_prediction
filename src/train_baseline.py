import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# Load processed dataset (from notebook output)
df = pd.read_csv("data/processed/retail_clean.csv")  # You’ll create this in the next notebook step

# Simple features
df = df[["Quantity", "UnitPrice", "TotalPrice"]]
X = df[["Quantity", "UnitPrice"]]
y = df["TotalPrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

mlflow.set_experiment("baseline-model")

# Start MLflow run
with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")

    print("Logged to MLflow with MSE:", mse)
