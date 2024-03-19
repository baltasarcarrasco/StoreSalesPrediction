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