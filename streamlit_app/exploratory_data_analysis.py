import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from store_sales_prediction.db_utilities import read_table


# Function to remove outliers from a given DataFrame
def remove_outliers(df, column="sales"):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def show_eda(store_nbr="All", product_family="All"):
    st.title("Exploratory Data Analysis")
    st.markdown(
        """
        This section focuses on exploring the sales distribution and trends over time. Use the sidebar controls to filter the sales data by store number and product family. Changes will be applied after clicking the "Apply Changes" button.
    """
    )
    df_sales = read_table("sales")
    df_stores = read_table("stores")
    df_holidays = read_table("holidays")

    df_sales["date"] = pd.to_datetime(df_sales["date"])
    df_holidays["date"] = pd.to_datetime(df_holidays["date"])

    # Initialize df_sales_filtered with the original df_sales DataFrame
    df_sales_filtered = df_sales.copy()

    # Apply filters based on the sidebar selections
    if st.session_state['apply_changes'] or st.session_state['first_load']:  # Assuming you want to filter on load as well; adjust as needed
        if store_nbr != "All":
            df_sales_filtered = df_sales_filtered[
                df_sales_filtered["store_nbr"] == store_nbr
            ]
        if product_family != "All":
            df_sales_filtered = df_sales_filtered[
                df_sales_filtered["family"] == product_family
            ]

    # Removing outliers only for the sales distribution visualization
    df_sales_filtered_no_outliers = remove_outliers(df_sales_filtered)

    # Distribution Visualization with Outlier Removal
    st.markdown("## Sales Distribution")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Sales Histogram")
        fig, ax = plt.subplots()
        sns.histplot(df_sales_filtered_no_outliers["sales"], bins=50, kde=True)
        plt.title("Distribution of Sales (Without Outliers)")
        st.pyplot(fig)

    with col2:
        st.markdown("#### Sales Box Plot")
        fig, ax = plt.subplots()
        sns.boxplot(x=df_sales_filtered_no_outliers["sales"])
        plt.title("Sales Box Plot (Without Outliers)")
        st.pyplot(fig)

    # Time Series Analysis (Full Data, Without Outlier Removal)
    # if apply_changes_btn:
    st.markdown("## Time Series Analysis of Sales")
    st.markdown("### Sales Trends Over Time")
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots()
    total_sales_by_date = df_sales_filtered.groupby("date")["sales"].sum()
    sns.lineplot(data=total_sales_by_date, ax=ax, color="royalblue", linewidth=2.5)
    ax.set_title("Daily Sales Over Time", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Total Sales", fontsize=14)
    ax.tick_params(labelsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Average Sales by Store Type
    st.markdown("## Store Characteristics")
    st.markdown("### Average Sales by Store Type")
    df_sales_stores = pd.merge(df_sales_filtered, df_stores, on="store_nbr", how="left")
    sales_by_store_type = df_sales_stores.groupby("type")["sales"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="type", y="sales", data=sales_by_store_type, ax=ax)
    ax.set_title("Average Sales by Store Type")
    st.pyplot(fig)

    # Average Sales by Cluster
    st.markdown("### Average Sales by Cluster")
    sales_by_cluster = df_sales_stores.groupby("cluster")["sales"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="cluster", y="sales", data=sales_by_cluster, ax=ax)
    ax.set_title("Average Sales by Cluster")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Holidays Impact on Sales
    st.markdown("## Holidays Impact on Sales")
    st.markdown("### Sales on Holidays vs Non-Holidays (Without Outliers)")
    # Merge holidays information
    df_sales_holidays = pd.merge(
        df_sales_filtered,
        df_holidays[["date", "locale", "transferred"]],
        on="date",
        how="left",
    )
    df_sales_holidays.fillna({"locale": "No Holiday"}, inplace=True)

    # Aggregate sales by date considering holidays
    daily_sales_holidays = (
        df_sales_holidays.groupby(["date", "locale"])["sales"].sum().reset_index()
    )

    # Remove outliers for this specific analysis
    daily_sales_holidays_filtered = remove_outliers(daily_sales_holidays, "sales")

    # Plotting
    fig, ax = plt.subplots()
    sns.boxplot(x="locale", y="sales", data=daily_sales_holidays_filtered, ax=ax)
    ax.set_title("Sales on Holidays vs Non-Holidays (Without Outliers)")
    st.pyplot(fig)


if __name__ == "__main__":
    show_eda()
