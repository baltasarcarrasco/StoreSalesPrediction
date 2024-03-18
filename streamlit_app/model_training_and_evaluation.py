import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from store_sales_prediction.db_utilities import read_table
import numpy as np
import joblib


def load_data():
    test = read_table("sales_test")
    predictions = read_table("sales_predictions")
    sales_raw_data = read_table("sales")
    stores_data = read_table("stores")
    return test, predictions, sales_raw_data, stores_data


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mse, rmse, mae, mape


def show_model_training_evaluation_page():
    st.title("Model Training and Evaluation")

    st.markdown("""
    ## Model Training and Evaluation

    This page details our approach to training the sales forecasting model and evaluates the model's performance on the test set. We used an XGBoost regressor model, optimized through grid search cross-validation with a TimeSeriesSplit.

    Below are the main regression metrics from the model evaluation, an interactive plot that compares actual vs. predicted sales, and a bar plot showing RMSE by product family.
    """)

    # Load the data
    test, predictions, sales_raw_data, stores_data = load_data()
    test['date'] = pd.to_datetime(test['date'])

    # Map back the original 'family' categories using join on 'id'
    test = test.merge(sales_raw_data[['id', 'family']], on='id', how='left')

    # Merge test data with predictions
    test_merged = pd.merge(test, predictions, on="id")

    # Calculate regression metrics
    mse, rmse, mae, mape = calculate_metrics(test_merged['sales_actual'], test_merged['sales_pred'])

    # Display regression metrics
    st.markdown("### Regression Metrics")
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")
    st.write(f"Mean Absolute Error (MAE): {mae}")
    st.write(f"Mean Absolute Percentage Error (MAPE): {mape}%")

    # Sidebar controls for user input
    st.sidebar.header('User Input Parameters')
    store_options = ['All'] + sorted(stores_data['store_nbr'].unique())
    product_family_options = ['All'] + sorted(sales_raw_data['family'].unique())

    store_nbr = st.sidebar.selectbox('Store Number', options=store_options)
    product_family = st.sidebar.selectbox('Product Family', options=product_family_options)

    # Apply filters based on sidebar selections
    filtered_test = test
    if store_nbr != 'All':
        filtered_test = filtered_test[filtered_test['store_nbr'] == store_nbr]
    if product_family != 'All':
        filtered_test = filtered_test[filtered_test['family'] == product_family]

    # Merge filtered test data with predictions
    filtered_test_merged = pd.merge(filtered_test, predictions, on="id")

    # Plotting actual vs. predicted sales
    st.markdown("### Actual vs. Predicted Sales")
    # Aggregate sales by date
    agg_sales = filtered_test_merged.groupby('date').agg(sales_actual=('sales_actual', 'sum'),
                                                         sales_pred=('sales_pred', 'sum')).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(agg_sales['date'], agg_sales['sales_actual'], label='Actual Sales', color='blue', linestyle='-',
            linewidth=2)
    ax.plot(agg_sales['date'], agg_sales['sales_pred'], label='Predicted Sales', color='red',
            linestyle='-', linewidth=2)
    ax.set_title(
        f'Actual vs. Predicted Sales {"- Store: " + store_nbr if store_nbr != "All" else ""} {"- Family: " + product_family if product_family != "All" else ""}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # RMSE by Product Family
    test_merged['rmse'] = (test_merged['sales_actual'] - test_merged['sales_pred']) ** 2
    rmse_by_family = test_merged.groupby('family')['rmse'].mean().reset_index()
    rmse_by_family['rmse'] = np.sqrt(rmse_by_family['rmse'])

    st.markdown("### RMSE by Product Family")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='rmse', y='family', data=rmse_by_family.sort_values('rmse', ascending=False), ax=ax)
    ax.set_title('RMSE by Product Family')
    ax.set_xlabel('RMSE')
    ax.set_ylabel('Product Family')
    st.pyplot(fig)


if __name__ == "__main__":
    show_model_training_evaluation_page()
