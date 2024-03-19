import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
from StoreSalesPrediction.db_utilities import read_table
from StoreSalesPrediction.data_processing_utilities import prepare_data, encode_data
import matplotlib.pyplot as plt
import joblib


def iterative_predictions(model, sales_df, input_date, store="All", family="All"):
    stores_df = read_table("stores")
    holidays_df = read_table("holidays")

    last_date = sales_df["date"].max()
    input_date = datetime.strptime(
        input_date, "%Y-%m-%d"
    )  # Assuming input_date is a string

    while last_date < input_date:
        last_date += timedelta(days=1)  # Predict for the next day
        df_temporary = sales_df.copy()

        # Filter the temporary DataFrame for the last 15 days for feature generation
        df_temporary = df_temporary[
            df_temporary["date"] >= last_date + timedelta(days=-15)
        ]

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
                for store in sales_df["store_nbr"].unique()
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

        for store, family in store_family_combinations:
            new_rows.append(
                {
                    "date": last_date,
                    "store_nbr": store,
                    "family": family,
                    "sales": 0,
                    "onpromotion": 0,
                }
            )  # Assume 'onpromotion' is a required feature; adjust as necessary

        df_new_rows = pd.DataFrame(new_rows)
        df_temporary = pd.concat([df_temporary, df_new_rows], ignore_index=True)

        df_temporary = prepare_data(
            df_sales=df_temporary, df_stores=stores_df, df_holidays=holidays_df
        )
        family_column = df_temporary["family"]
        df_temporary = encode_data(df_temporary)
        df_temporary["family"] = family_column

        for _, row in df_new_rows.iterrows():
            input_ready = df_temporary[
                (df_temporary["date"] == last_date)
                & (df_temporary["store_nbr"] == row["store_nbr"])
                & (df_temporary["family"] == row["family"])
            ]
            input_ready = input_ready.drop(
                ["date", "sales", "family"], axis=1
            )  # Drop non-feature columns; adjust as needed
            predicted_sales = model.predict(input_ready)
            # Update the original sales_df with the predicted sales
            sales_df = sales_df.append(
                {
                    "date": last_date,
                    "store_nbr": row["store_nbr"],
                    "family": row["family"],
                    "sales": predicted_sales[0],
                },
                ignore_index=True,
            )

    # Return the updated sales_df with predictions
    return sales_df


def show_predictions_and_insights(
    store_nbr="All", product_family="All", apply_changes_btn=False
):
    st.title("Predictions and Insights")

    # Load necessary data
    model = joblib.load("./models/store_sales_model.pkl")
    sales_df = read_table("sales")
    sales_df["date"] = pd.to_datetime(sales_df["date"])
    last_date = sales_df["date"].max()

    # User input for prediction date, placed directly in the page
    prediction_date = st.date_input(
        "Enter prediction end date",
        value=(last_date + timedelta(days=15)),
        min_value=(last_date + timedelta(days=1)),
    )

    if apply_changes_btn or True:  # Check if "Apply Changes" is clicked
        # Ensure user has selected a future date
        if prediction_date <= last_date.date():
            st.warning("Please select a date after the last available data date.")
            return

        # Call the iterative prediction function
        predicted_sales_df = iterative_predictions(
            model,
            sales_df,
            prediction_date.strftime("%Y-%m-%d"),
            store_nbr,
            product_family,
        )

        # Filter the DataFrame for the last 15 days and the prediction period
        plot_data = predicted_sales_df[
            (predicted_sales_df["date"] > last_date - timedelta(days=15))
            & (predicted_sales_df["date"] <= prediction_date)
        ]

        if store_nbr != "All":
            plot_data = plot_data[plot_data["store_nbr"] == store_nbr]
        if product_family != "All":
            plot_data = plot_data[plot_data["family"] == product_family]

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            plot_data[plot_data["date"] <= last_date]["date"],
            plot_data[plot_data["date"] <= last_date]["sales"],
            label="Actual Sales",
            color="blue",
        )
        ax.plot(
            plot_data[plot_data["date"] > last_date]["date"],
            plot_data[plot_data["date"] > last_date]["sales"],
            label="Predicted Sales",
            color="red",
            linestyle="--",
        )
        ax.set_title("Sales Predictions")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.legend()
        st.pyplot(fig)


if __name__ == "__main__":
    show_predictions_and_insights()  # These parameters should be set based on the global sidebar inputs
