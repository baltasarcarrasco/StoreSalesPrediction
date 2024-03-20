import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from store_sales_prediction.db_utilities import read_table
from sidebar_controls import create_sidebar_controls

def show_predictions_and_insights(store_nbr, product_family, apply_changes_btn=False):
    st.title("Predictions and Insights")
    st.markdown("""
    This section visualizes both actual sales data for the last 30 days and predicted sales up to December 31, 2017. Use the sidebar controls to filter data by store number and product family.
    """)
    # Loading model
    model = joblib.load("./models/store_sales_model.pkl")

    #Loading data
    test = read_table("sales_test")
    predictions = read_table("sales_predictions")
    sales_raw_data = read_table("sales")
    actual_sales_df = read_table('sales_train')
    # Map back the original 'family' categories using join on 'id'
    test = test.merge(sales_raw_data[["id", "family"]], on="id", how="left")
    actual_sales_df = actual_sales_df.merge(sales_raw_data[["id", "family"]], on="id", how="left")
    # Merge test data with predictions
    test_merged = pd.merge(test, predictions, on="id")
    #predicted_sales_df = read_table('predicted_sales')
    predicted_sales_df = test_merged
    predicted_sales_df['type'] = 'Predicted'
    predicted_sales_df = predicted_sales_df[['id', 'date', 'store_nbr', 'family', 'sales_pred','type']]
    predicted_sales_df.rename(columns={'sales_pred':'sales'}, inplace=True)

    #st.dataframe(predicted_sales_df)

    actual_sales_df = actual_sales_df[['id', 'date', 'store_nbr', 'family','sales']]
    actual_sales_df['type'] = 'Actual'
    actual_sales_df['date'] = pd.to_datetime(actual_sales_df['date'])
    predicted_sales_df['date'] = pd.to_datetime(predicted_sales_df['date'])

    df = pd.concat([actual_sales_df, predicted_sales_df])
    df['date'] = pd.to_datetime(df['date'])


    # Date input for date selection
    prediction_date = st.date_input("Select a date for prediction", value = actual_sales_df['date'].max() + pd.Timedelta(days=30),min_value=actual_sales_df['date'].max() + pd.Timedelta(days=1),
                                    max_value=predicted_sales_df['date'].max())

    prediction_date = pd.to_datetime(prediction_date)

    if apply_changes_btn or True:
        # Time series plot
        fig, ax = plt.subplots()

        # Filter and aggregate if necessary
        if store_nbr == 'All':
            actual_sales = df[
                (df['type'] == 'Actual') & (df['date'] >= actual_sales_df['date'].max() - pd.Timedelta(days=30))]
            predicted_sales = df[(df['type'] == 'Predicted') & (df['date'] <= prediction_date)]
            if product_family != 'All':
                actual_sales = actual_sales[actual_sales['family'] == product_family]
                predicted_sales = predicted_sales[predicted_sales['family'] == product_family]
            # Aggregate by date since all stores or all families are selected
            actual_sales = actual_sales.groupby('date')['sales'].sum().reset_index()
            predicted_sales = predicted_sales.groupby('date')['sales'].sum().reset_index()

        elif product_family == 'All':
            actual_sales = df[
                (df['type'] == 'Actual') & (df['date'] >= actual_sales_df['date'].max() - pd.Timedelta(days=30))]
            predicted_sales = df[(df['type'] == 'Predicted') & (df['date'] <= prediction_date)]
            if store_nbr != 'All':
                actual_sales = actual_sales[actual_sales['store_nbr'] == store_nbr]
                predicted_sales = predicted_sales[predicted_sales['store_nbr'] == store_nbr]
            # Aggregate by date since all stores or all families are selected
            actual_sales = actual_sales.groupby('date')['sales'].sum().reset_index()
            predicted_sales = predicted_sales.groupby('date')['sales'].sum().reset_index()

        else:
            # Filter based on sidebar controls without aggregating
            actual_sales = df[
                (df['type'] == 'Actual') & (df['date'] >= actual_sales_df['date'].max() - pd.Timedelta(days=30))]
            predicted_sales = df[(df['type'] == 'Predicted') & (df['date'] <= prediction_date)]
            if store_nbr != 'All':
                actual_sales = actual_sales[actual_sales['store_nbr'] == store_nbr]
                predicted_sales = predicted_sales[predicted_sales['store_nbr'] == store_nbr]
            if product_family != 'All':
                actual_sales = actual_sales[actual_sales['family'] == product_family]
                predicted_sales = predicted_sales[predicted_sales['family'] == product_family]

        # Plotting
        actual_sales['date'] = pd.to_datetime(actual_sales['date'])
        predicted_sales['date'] = pd.to_datetime(predicted_sales['date'])

        sns.lineplot(data=actual_sales, x='date', y='sales', ax=ax, label='Actual Sales', color='blue')
        sns.lineplot(data=predicted_sales, x='date', y='sales', ax=ax, label='Predicted Sales', color='red')

        plt.title('Sales Forecast')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        st.pyplot(fig)

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            st.markdown("### Feature Importance")
            feature_names = model.get_booster().feature_names
            feature_importances = model.feature_importances_
            features = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importances
            })

            # Filter features with importance greater than 0
            features = features[features['Importance'] > 0.0019]

            features.sort_values(by='Importance', ascending=False, inplace=True)

            # Create a figure and axis for the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=features, ax=ax)
            ax.set_title('Feature Importance')
            ax.set_xlabel('Importance')
            ax.set_ylabel('Features')
            st.pyplot(fig)

if __name__ == "__main__":
    # Assuming create_sidebar_controls returns (store_nbr, product_family, apply_changes_btn)
    store_nbr, product_family, apply_changes_btn = create_sidebar_controls()
    show_predictions_and_insights(store_nbr, product_family, apply_changes_btn)


