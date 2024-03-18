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
    # Parse the input date
    initial_date = pd.to_datetime(initial_date)

    # Load the saved model
    model = joblib.load("../models/user_model.pkl")

    # Load the test data
    test = read_table("sales_test")

    # Convert the date column to datetime format
    test["date"] = pd.to_datetime(test["date"])

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
    results = pd.DataFrame({"predicted_sales": y_pred, "actual_sales": y_test})
    write_table(results, "user_sales_predictions")


def plot_predictions():
    # Load the predictions
    results_df = read_table("user_sales_predictions")

    # Creating the scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(results_df["predicted_sales"], results_df["actual_sales"], alpha=0.7)

    # Adding a line for perfect predictions
    max_val = max(results_df["predicted_sales"].max(), results_df["actual_sales"].max())
    plt.plot([0, max_val], [0, max_val], "r--")  # Red dashed line for reference

    # Labels and titles
    plt.xlabel("Predicted Sales")
    plt.ylabel("Actual Sales")
    plt.title("Predicted vs. Actual Sales")

    # Show plot
    plt.grid(True)
    plt.axis(
        "equal"
    )  # Equal aspect ratio ensures that the line of perfect prediction is at 45 degrees
    plt.show()
