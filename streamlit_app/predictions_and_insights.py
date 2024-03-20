import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from store_sales_prediction.db_utilities import read_table
from sidebar_controls import create_sidebar_controls


def show_predictions_and_insights(store_nbr, product_family):
    st.title("Predictions and Insights")
    st.markdown("""
    This section offers a visualization tool that allows for the comparison of actual sales data from the last 30 days with sales predictions for future dates, extending up to the user-selected end date. Utilize the sidebar controls to refine your view by selecting specific store numbers and product families, thereby customizing your comparison of historical data against forecasts for new, unseen days.
    """)
    # Loading model
    model = joblib.load("./models/store_sales_model.pkl")

    # Loading data
    test = read_table("sales_test")
    predictions = read_table("sales_predictions")
    sales_raw_data = read_table("sales")
    actual_sales_df = read_table("sales_train")
    # Map back the original 'family' categories using join on 'id'
    test = test.merge(sales_raw_data[["id", "family"]], on="id", how="left")
    actual_sales_df = actual_sales_df.merge(
        sales_raw_data[["id", "family"]], on="id", how="left"
    )
    # Merge test data with predictions
    test_merged = pd.merge(test, predictions, on="id")
    # predicted_sales_df = read_table('predicted_sales')
    predicted_sales_df = test_merged
    predicted_sales_df["type"] = "Predicted"
    predicted_sales_df = predicted_sales_df[
        ["id", "date", "store_nbr", "family", "sales_pred", "type"]
    ]
    predicted_sales_df.rename(columns={"sales_pred": "sales"}, inplace=True)

    # st.dataframe(predicted_sales_df)

    actual_sales_df = actual_sales_df[["id", "date", "store_nbr", "family", "sales"]]
    actual_sales_df["type"] = "Actual"
    actual_sales_df["date"] = pd.to_datetime(actual_sales_df["date"])
    predicted_sales_df["date"] = pd.to_datetime(predicted_sales_df["date"])

    df = pd.concat([actual_sales_df, predicted_sales_df])
    df["date"] = pd.to_datetime(df["date"])

    # Date input for date selection
    prediction_date = st.date_input(
        "Select a date for prediction",
        value=actual_sales_df["date"].max() + pd.Timedelta(days=30),
        min_value=actual_sales_df["date"].max() + pd.Timedelta(days=1),
        max_value=predicted_sales_df["date"].max(),
    )

    prediction_date = pd.to_datetime(prediction_date)

    if st.session_state["apply_changes"] or st.session_state["first_load"]:
        # Time series plot
        fig, ax = plt.subplots()

        # Filter and aggregate if necessary
        if store_nbr == "All":
            actual_sales = df[
                (df["type"] == "Actual")
                & (df["date"] >= actual_sales_df["date"].max() - pd.Timedelta(days=30))
            ]
            predicted_sales = df[
                (df["type"] == "Predicted") & (df["date"] <= prediction_date)
            ]
            if product_family != "All":
                actual_sales = actual_sales[actual_sales["family"] == product_family]
                predicted_sales = predicted_sales[
                    predicted_sales["family"] == product_family
                ]
            # Aggregate by date since all stores or all families are selected
            actual_sales = actual_sales.groupby("date")["sales"].sum().reset_index()
            predicted_sales = (
                predicted_sales.groupby("date")["sales"].sum().reset_index()
            )

        elif product_family == "All":
            actual_sales = df[
                (df["type"] == "Actual")
                & (df["date"] >= actual_sales_df["date"].max() - pd.Timedelta(days=30))
            ]
            predicted_sales = df[
                (df["type"] == "Predicted") & (df["date"] <= prediction_date)
            ]
            if store_nbr != "All":
                actual_sales = actual_sales[actual_sales["store_nbr"] == store_nbr]
                predicted_sales = predicted_sales[
                    predicted_sales["store_nbr"] == store_nbr
                ]
            # Aggregate by date since all stores or all families are selected
            actual_sales = actual_sales.groupby("date")["sales"].sum().reset_index()
            predicted_sales = (
                predicted_sales.groupby("date")["sales"].sum().reset_index()
            )

        else:
            # Filter based on sidebar controls without aggregating
            actual_sales = df[
                (df["type"] == "Actual")
                & (df["date"] >= actual_sales_df["date"].max() - pd.Timedelta(days=30))
            ]
            predicted_sales = df[
                (df["type"] == "Predicted") & (df["date"] <= prediction_date)
            ]
            if store_nbr != "All":
                actual_sales = actual_sales[actual_sales["store_nbr"] == store_nbr]
                predicted_sales = predicted_sales[
                    predicted_sales["store_nbr"] == store_nbr
                ]
            if product_family != "All":
                actual_sales = actual_sales[actual_sales["family"] == product_family]
                predicted_sales = predicted_sales[
                    predicted_sales["family"] == product_family
                ]

        # Plotting
        actual_sales["date"] = pd.to_datetime(actual_sales["date"])
        predicted_sales["date"] = pd.to_datetime(predicted_sales["date"])

        sns.lineplot(
            data=actual_sales,
            x="date",
            y="sales",
            ax=ax,
            label="Actual Sales",
            color="blue",
        )
        sns.lineplot(
            data=predicted_sales,
            x="date",
            y="sales",
            ax=ax,
            label="Predicted Sales",
            color="red",
        )

        plt.title("Sales Forecast")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.legend()
        st.pyplot(fig)

        # Feature importance
        if hasattr(model, "feature_importances_"):
            st.markdown(
                """### Feature Importance
                
                - The **`sales_seasonal_rolling_mean_7`** emerges as the most influential feature, significantly outpacing others. This highlights the importance of short-term historical sales trends and their impact on forecasting accuracy. The 7-day rolling mean effectively captures weekly sales patterns, which are crucial for understanding consumer behavior and demand fluctuations.
                - **Past sales variables** like **`sales_lag_1`**, **`sales_seasonal_lag_1`**, **`rolling mean 3`**, and **`rolling mean 2`** are among the top influencers. This underscores the predictive power of recent sales data in forecasting future sales. These features help the model capture immediate sales dynamics and trends, which are predictive of near-future performance.
                - The significance of **`lag 5`** suggests that sales data from five days ago also have a meaningful impact on future sales predictions, pointing to specific weekly patterns or consumer purchasing cycles that affect sales.
                - **`Store`** and **`on promotion`** features indicate that geographical location and promotional activities are also key drivers of sales. These factors account for external influences on consumer purchasing decisions, emphasizing the need for targeted marketing and inventory planning.
                - The **`locale`** feature's importance reveals the effect of holidays on consumer purchasing behavior. This suggests that sales are not only influenced by regular patterns but also by seasonal events and holidays, which can significantly alter consumer demand.
                       """ )
            feature_names = model.get_booster().feature_names
            feature_importances = model.feature_importances_
            features = pd.DataFrame(
                {"Feature": feature_names, "Importance": feature_importances}
            )

            # Filter features with importance greater than 0
            features = features[features["Importance"] > 0.0019]

            features.sort_values(by="Importance", ascending=False, inplace=True)

            # Create a figure and axis for the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x="Importance", y="Feature", data=features, ax=ax)
            ax.set_title("Feature Importance")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Features")
            st.pyplot(fig)

            st.markdown(
                """### Main Insights

                - **Recency Effect**: Recent sales data are closely related to the current market conditions, customer preferences, and inventory levels, making them highly relevant for short-term predictions.
                - **Pattern Recognition**: Machine learning models excel at identifying patterns. Past sales data, especially when structured as lags and rolling means, present clear patterns that models can learn and extrapolate into the future.
                - **Consumer Behavior**: Purchasing habits tend to be consistent over short periods. By analyzing recent sales, models can predict future behavior based on established trends.
                - **Impact of External Factors**: While variables like promotions and holidays do impact sales, their effects are often mediated through changes in recent sales patterns, which the model captures through lagged sales features.
                       """)


if __name__ == "__main__":
    # Assuming create_sidebar_controls returns (store_nbr, product_family, apply_changes_btn)
    store_nbr, product_family, apply_changes_btn = create_sidebar_controls()
    show_predictions_and_insights(store_nbr, product_family, apply_changes_btn)
