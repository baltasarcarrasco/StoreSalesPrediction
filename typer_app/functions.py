import pandas as pd
import sys

sys.path.append("..")
from store_sales_prediction.db_utilities import read_table, write_table
import pandas as pd
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt


def train_model():
    # Read the processed sales data
    train_df = read_table("sales_train")

    # Define the features and target variable
    X_train = train_df.drop(["id", "date", "sales"], axis=1)
    y_train = train_df["sales"]

    # Define the model
    model = XGBRegressor()

    # Fit the grid search to the training data
    model.fit(X_train, y_train)

    joblib.dump(model, "../models/user_model.pkl")


def predict(initial_date, n_days):
    # Load the saved model
    model = joblib.load("../models/user_model.pkl")

    # Load the test data
    test = read_table("sales_test")

    # Filter the test data to include only the specified date range
    test = test[
        (test["date"] >= initial_date)
        & (test["date"] < initial_date + pd.Timedelta(days=n_days))
    ]

    # Define the features and target variable
    X_test = test.drop(["id", "date", "sales"], axis=1)
    y_test = test["sales"]

    # Make predictions
    y_pred = model.predict(X_test)

    # Save predictions and actual values to the database
    results = pd.DataFrame({"sales_pred": y_pred, "sales_actual": y_test})
    write_table(results, "user_sales_predictions")


def plot_predictions():
    # Load the predictions
    results = read_table("user_sales_predictions")

    # Plot the predictions and actual values
    plt.plot(results["sales_pred"], label="Predicted Sales")
    plt.plot(results["sales_actual"], label="Actual Sales")
    plt.xlabel("Predicted Sales")
    plt.ylabel("Actual Sales")
    plt.legend()

    plt.show()
