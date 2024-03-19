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

    model = joblib.load("./models/store_sales_model.pkl")
    print('model_loaded')
    actual_sales_df = read_table('sales')
    print('sales_read')
    predicted_sales_df = read_table('predicted_sales')
    print('predictions_read')
    actual_sales_df = actual_sales_df[['id', 'date', 'store_nbr', 'family']]
    actual_sales_df['type'] = 'Actual'
    print('select columns')
    actual_sales_df['date'] = pd.to_datetime(actual_sales_df['date'])
    predicted_sales_df['date'] = pd.to_datetime(predicted_sales_df['date'])
    df = pd.concat([actual_sales_df, predicted_sales_df])
    print('concat done')


    # Sidebar input for date selection
    prediction_date = st.date_input("Select a date for prediction", value = actual_sales_df['date'].max() + pd.Timedelta(days=1),min_value=actual_sales_df['date'].max() + pd.Timedelta(days=1),
                                    max_value=pd.to_datetime('2017-08-18'))

    if apply_changes_btn or True:
        # Filtering based on sidebar controls
        if store_nbr != 'All':
            df = df[df['store_nbr'] == store_nbr]
        if product_family != 'All':
            df = df[df['family'] == product_family]

        # Time series plot
        fig, ax = plt.subplots()
        # Actual sales
        actual_sales = df[df['type'] == 'Actual']
        sns.lineplot(data=actual_sales, x='date', y='sales', ax=ax, label='Actual Sales', color='blue')
        # Predicted sales
        predicted_sales = df[(df['type'] == 'Predicted') & (df['date'] <= prediction_date)]
        sns.lineplot(data=predicted_sales, x='date', y='sales', ax=ax, label='Predicted Sales', color='red', linestyle='--')
        plt.title('Sales Forecast')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        st.pyplot(fig)

        # Feature importance (if applicable)
        if hasattr(model, 'feature_importances_'):
            st.markdown("### Feature Importance")
            features = pd.DataFrame({'Feature': df.columns.drop(['id', 'date', 'store_nbr', 'family', 'sales', 'type']),
                                     'Importance': model.feature_importances_})
            features.sort_values(by='Importance', ascending=True, inplace=True)
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=features)
            plt.title('Feature Importance')
            plt.xlabel('Importance')
            plt.ylabel('Features')
            st.pyplot()

if __name__ == "__main__":
    # Assuming create_sidebar_controls returns (store_nbr, product_family, apply_changes_btn)
    store_nbr, product_family, apply_changes_btn = create_sidebar_controls()
    show_predictions_and_insights(store_nbr, product_family, apply_changes_btn)


