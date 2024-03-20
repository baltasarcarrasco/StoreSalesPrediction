from datetime import datetime, timedelta, date
import pandas as pd
from store_sales_prediction.db_utilities import read_table
from store_sales_prediction.data_processing_utilities import prepare_data, encode_data
import joblib

def iterative_predictions(model, sales_df, input_date, store="All", family="All"):
    """'
    This function works with a user-input model to iteratively predict sales of desired store
    and/or family desired until user input_date.
    """
    stores_df = read_table("stores")
    holidays_df = read_table("holidays")
    sales_df["date"] = pd.to_datetime(sales_df["date"])

    # Convert input date to datetime format
    input_date = pd.to_datetime(input_date)
    last_date = sales_df['date'].max()
    sales_df = sales_df[(sales_df["date"] >= last_date + timedelta(days=-30))]

    # Generate a date range for the prediction period
    prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=input_date, freq='D')

    # Assuming you have a way to determine the maximum number of store-family combinations beforehand
    max_combinations = len(stores_df['store_nbr'].unique()) * len(sales_df['family'].unique())
    predicted_rows = []

    # Iterate through each prediction date
    for pred_date in prediction_dates:
        # Generate rows for every store-family combination for this prediction date
        for store_nbr in stores_df['store_nbr'].unique():
            for family in sales_df['family'].unique():
                # Simulate a row for prediction
                simulated_row = {
                    "date": pred_date,
                    "store_nbr": store_nbr,
                    "family": family,
                    "sales": 0,
                    "onpromotion": 0,
                }

                # Append the simulated row for this date-store-family combination
                predicted_rows.append(simulated_row)

        # Convert predicted rows to a DataFrame
        df_predicted_rows = pd.DataFrame(predicted_rows)
        df_combined = pd.concat([sales_df, df_predicted_rows])

        # Prepare the DataFrame for prediction (use your existing logic here)
        df_predicted_rows_prepared = prepare_data(df_combined, stores_df, holidays_df)
        df_predicted_rows_encoded = encode_data(df_predicted_rows_prepared)
        df_predicted_rows_encoded = df_predicted_rows_encoded[(df_predicted_rows_encoded['date'] == pred_date)]
        df_predicted_rows = df_predicted_rows[(df_predicted_rows['date'] == pred_date)]
        # Predict sales for these rows
        # Ensure your model and feature extraction logic aligns with the DataFrame structure
        X = df_predicted_rows_encoded.drop(['date','id','sales'], axis=1)
        predictions = model.predict(X)

        # Assign the predictions back to the DataFrame
        df_predicted_rows['sales'] = predictions

        # Concatenate the newly predicted rows with the sales_df for the next iteration
        sales_df = pd.concat([sales_df, df_predicted_rows])
        print(pred_date)

    # After all predictions are made, return the updated sales DataFrame
    df_predicted_rows['id'] = df_predicted_rows.index
    df_predicted_rows = df_predicted_rows[['id', 'date', 'store_nbr', 'family']]
    df_predicted_rows['type'] = 'Predicted'
    return df_predicted_rows