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
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    last_date = sales_df["date"].max()
    sales_df = sales_df[(sales_df["date"] >= last_date + timedelta(days=-30))]
    input_date = datetime.strptime(
        input_date, "%Y-%m-%d"
    )  # Assuming input_date is a string

    df_temporary = sales_df.copy()
    if store != "All" or family != "All":
        # Filter by store and family if specified
        if store != "All":
            df_temporary = df_temporary[df_temporary["store_nbr"] == store]
        if family != "All":
            df_temporary = df_temporary[df_temporary["family"] == family]

    # Prepare a new row for each combination of store and family
    new_rows = []
    if store == "All" and family == "All":
        store_family_combinations = [
            (store, family)
            for store in stores_df["store_nbr"].unique()
            for family in sales_df["family"].unique()
        ]
    elif store == "All":
        store_family_combinations = [
            (store, family) for store in sales_df["store_nbr"].unique()
        ]
    elif family == "All":
        store_family_combinations = [
            (store, family) for family in sales_df["family"].unique()
        ]
    else:
        store_family_combinations = [(store, family)]

    while last_date < input_date:
        last_date += timedelta(days=1)  # Predict for the next day
        df_temporary = sales_df.copy()

        # Filter the temporary DataFrame for the last 15 days for feature generation
        df_temporary = df_temporary[
            df_temporary["date"] >= last_date + timedelta(days=-30)
            ]

        for store, family in store_family_combinations:
            new_rows.append(
                {
                    "date": last_date,
                    "store_nbr": store,
                    "family": family,
                    "sales": 0,
                    "onpromotion": 0,
                }
            )

        df_new_rows = pd.DataFrame(new_rows)
        df_temporary = pd.concat([df_temporary, df_new_rows], ignore_index=True)
        df_temporary['id'] = df_temporary.index
        df_temporary = prepare_data(
            df_sales=df_temporary, df_stores=stores_df, df_holidays=holidays_df
        )
        family_column = df_temporary["family"]
        df_temporary = encode_data(df_temporary)
        df_temporary["family"] = family_column

        for _, row in df_new_rows.iterrows():
            input_ready = df_temporary[
                (df_temporary['date'] == last_date) & (df_temporary['store_nbr'] == row['store_nbr']) & (
                        df_temporary['family'] == row['family'])]
            input_ready = input_ready.drop(['id', 'date', 'sales', 'family'],
                                           axis=1)  # Drop non-feature columns
            predicted_sales = model.predict(input_ready)
            # Update the original sales_df with the predicted sales
            new_row = {"date": last_date, "store_nbr": row["store_nbr"], "family": row["family"],
                       "sales": predicted_sales[0], 'onpromotion': 0}
            new_rows.append(new_row)

        sales_df = pd.concat([sales_df, pd.DataFrame(new_rows)], ignore_index=True)
        print(last_date)
    predictions = pd.DataFrame(new_rows)
    predictions['id'] = predictions.index
    predictions = predictions[['id', 'date', 'store_nbr', 'family']]
    predictions['type'] = 'Predicted'

    # Return the predictions
    return predictions


def iterative_predictions_optimized(model, sales_df, input_date, store="All", family="All"):
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