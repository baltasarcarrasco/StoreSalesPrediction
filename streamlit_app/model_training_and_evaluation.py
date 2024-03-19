import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from store_sales_prediction.db_utilities import read_table
import numpy as np
import joblib


def load_data():
    test = read_table("sales_test")
    predictions = read_table("sales_predictions")
    sales_raw_data = read_table("sales")
    stores_data = read_table("stores")
    general_metrics = read_table("metrics_summary")
    return test, predictions, sales_raw_data, stores_data, general_metrics


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_percentage_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return mse, rmse, mae, mape


def show_model_training_evaluation_page(
    store_nbr="All", product_family="All", apply_changes_btn=False
):
    st.title("Model Training and Evaluation")

    st.markdown(
        """
    ## Model Training and Evaluation

    This page details our approach to training the sales forecasting model and evaluates the model's performance on the test set. We used an XGBoost regressor model, optimized through grid search cross-validation with a TimeSeriesSplit.

    Below are the main regression metrics from the model evaluation, an interactive plot that compares actual vs. predicted sales, and a bar plot showing RMSE by product family.
    """
    )

    # Load the data
    test, predictions, sales_raw_data, stores_data, general_metrics = load_data()
    general_metrics = general_metrics[(general_metrics["metric"] != "mape")]
    test["date"] = pd.to_datetime(test["date"])

    # Map back the original 'family' categories using join on 'id'
    test = test.merge(sales_raw_data[["id", "family"]], on="id", how="left")

    # Merge test data with predictions
    test_merged = pd.merge(test, predictions, on="id")

    # Adjust for global sidebar control
    filtered_test = test_merged.copy()
    if store_nbr != "All":
        filtered_test = filtered_test[filtered_test["store_nbr"] == store_nbr]
    if product_family != "All":
        filtered_test = filtered_test[filtered_test["family"] == product_family]

    # Display regression metrics using the general_metrics df
    st.markdown("### Regression Metrics")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Training Set Metrics")
        for index, row in general_metrics.iterrows():
            st.write(f"{row['metric'].upper()}: {row['train']:,.2f}")

    with col2:
        st.markdown("#### Testing Set Metrics")
        for index, row in general_metrics.iterrows():
            st.write(f"{row['metric'].upper()}: {row['test']:,.2f}")

    # Plotting actual vs. predicted sales
    st.markdown("### Actual vs. Predicted Sales")
    # Aggregate sales by date
    agg_sales = (
        filtered_test.groupby("date")
        .agg(sales_actual=("sales_actual", "sum"), sales_pred=("sales_pred", "sum"))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        agg_sales["date"],
        agg_sales["sales_actual"],
        label="Actual Sales",
        color="blue",
        linestyle="-",
        linewidth=2,
    )
    ax.plot(
        agg_sales["date"],
        agg_sales["sales_pred"],
        label="Predicted Sales",
        color="red",
        linestyle="-",
        linewidth=2,
    )
    ax.set_title(
        f'Actual vs. Predicted Sales {"- Store: " + store_nbr if store_nbr != "All" else ""} {"- Family: " + product_family if product_family != "All" else ""}'
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # RMSE by Product Family
    test_merged["rmse"] = (test_merged["sales_actual"] - test_merged["sales_pred"]) ** 2
    rmse_by_family = test_merged.groupby("family")["rmse"].mean().reset_index()
    rmse_by_family["rmse"] = np.sqrt(rmse_by_family["rmse"])

    st.markdown("### RMSE by Product Family")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x="rmse",
        y="family",
        data=rmse_by_family.sort_values("rmse", ascending=False),
        ax=ax,
    )
    ax.set_title("RMSE by Product Family")
    ax.set_xlabel("RMSE")
    ax.set_ylabel("Product Family")
    st.pyplot(fig)


if __name__ == "__main__":
    show_model_training_evaluation_page()
