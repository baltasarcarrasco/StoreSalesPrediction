from store_sales_prediction.db_utilities import read_table
import matplotlib.pyplot as plt
import pandas as pd


def plot_predictions():
    """
    This function plots the predictions and actual values.
    """
    # If table there were no predictions made by the user, read the predictions from the database
    try:
        results_df = read_table("user_sales_predictions")
    except FileNotFoundError:
        results_df = read_table("sales_predictions")

    # Aggregate the predictions and actual values by date
    results_df["date"] = pd.to_datetime(results_df["date"])
    results_df = results_df.groupby("date").agg(
        {"predicted_sales": "sum", "actual_sales": "sum"}
    )

    # Plot the predictions and actual values
    plt.figure(figsize=(10, 5))
    plt.plot(results_df.index, results_df["predicted_sales"], label="Predictions")
    plt.plot(results_df.index, results_df["actual_sales"], label="Actual values")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title("Predictions vs Actual Values")
    plt.legend()

    plt.show()
