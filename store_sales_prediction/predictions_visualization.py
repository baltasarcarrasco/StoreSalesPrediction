from store_sales_prediction.db_utilities import read_table
import matplotlib.pyplot as plt


def plot_predictions():
    """
    This function plots the predictions and actual values.
    """
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
