import pandas as pd
from store_sales_prediction.db_utilities import write_table, read_table
from sklearn.metrics import (
    mean_squared_error,
    # root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
import joblib


def evaluate_model():
    """
    This function evaluates the model using the test data and saves the results to the database.
    """
    eval_metrics = {
        "mse": mean_squared_error,
        # "rmse": root_mean_squared_error,
        "mae": mean_absolute_error,
        "mape": mean_absolute_percentage_error,
    }

    # Load the saved model
    model = joblib.load("./models/store_sales_model.pkl")

    # Load the test data
    test = read_table("sales_test")
    train = read_table("sales_train")

    # Define the features and target variable
    X_test = test.drop(["id", "date", "sales"], axis=1)
    y_test = test["sales"]
    X_train = train.drop(["id", "date", "sales"], axis=1)
    y_train = train["sales"]

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Evaluate the model

    metrics_list = []
    train_scores = []
    test_scores = []
    for metric, function in eval_metrics.items():
        train_metric = function(y_train, y_pred_train)
        test_metric = function(y_test, y_pred)
        metrics_list.append(metric)
        train_scores.append(train_metric)
        test_scores.append(test_metric)

    metrics_df = pd.DataFrame(
        {"metric": metrics_list, "train": train_scores, "test": test_scores}
    )

    # Save predictions, actual values and metrics to the database
    results = pd.DataFrame(
        {"id": test["id"], "sales_pred": y_pred, "sales_actual": y_test}
    )
    results_train = pd.DataFrame(
        {"id": train["id"], "sales_pred": y_pred_train, "sales_actual": y_train}
    )
    write_table(results, "sales_predictions")
    write_table(results_train, "sales_predictions_train")
    write_table(metrics_df, "metrics_summary")


def predict(initial_date, n_days):
    """
    This function makes predictions for a given date range and saves the results to the database.
    """
    # Parse the input date
    initial_date = pd.to_datetime(initial_date)

    # If the user has trained a model, load the user model, else load the store sales model
    try:
        model = joblib.load("./models/user_model.pkl")
    except FileNotFoundError:
        model = joblib.load("./models/store_sales_model.pkl")

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
    results = pd.DataFrame(
        {"date": test["date"], "predicted_sales": y_pred, "actual_sales": y_test}
    )
    write_table(results, "user_sales_predictions")
