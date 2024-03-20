import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from store_sales_prediction.db_utilities import read_table


def show_data_overview():
    st.title("Data Overview")

    # Introduction to the dataset
    st.markdown(
        """
    Our journey begins with an exploration of the dataset from Favorita, capturing daily sales across various stores in Ecuador. This dataset not only reflects the purchasing patterns of thousands of products but also illustrates the influence of promotions, store characteristics, and even external factors like holidays on sales dynamics.

    **Sales data features**:
    - **date**: The day of sales records, offering insights into daily, weekly, and seasonal trends.
    - **store_nbr**: Identifies the store, enabling analysis of geographical and store-specific patterns.
    - **family**: The category of the product, crucial for understanding category-wise sales performance.
    - **sales**: The number of units sold, the primary target variable for our forecasting models.
    - **onpromotion**: Indicates the number of products of the family that were on promotion that day.
    
    **Stores features**:
    - **`state`**: Indicates the store's location within Ecuador.
    - **`cluster`**: Defines internal store clusters based on certain characteristics.
    - **`type`**: Provides an internal categorization of stores.
    
    **Holidays:***
    An **integrated calendar** details all holidays celebrated in Ecuador, including the dates and types of holidays, offering additional insights into potential sales impacts.
    """
    )

    # Load data
    df_sales = read_table("sales")
    df_stores = read_table("stores")
    df_sales["date"] = pd.to_datetime(df_sales["date"])

    # Display sample data
    st.markdown("### Sample Data")
    st.dataframe(df_sales.head())

    # Compute basic metrics
    total_sales = df_sales["sales"].sum()
    number_of_stores = df_sales["store_nbr"].nunique()
    number_of_families = df_sales["family"].nunique()
    number_of_time_series = number_of_stores * number_of_families

    # Show basic metrics
    st.markdown("### Key Metrics")
    st.metric(label="Total Sales", value=f"{total_sales:,.0f} units")
    st.metric(label="Number of Stores", value=f"{number_of_stores}")
    st.metric(label="Product Families", value=f"{number_of_families}")
    st.metric(label="Time Series", value=f"{number_of_time_series}")

    # Counting the number of stores per state
    stores_count = (
        df_stores.groupby("state")["store_nbr"].nunique().reset_index(name="NUM_STORES")
    )
    stores_count = stores_count.sort_values("NUM_STORES", ascending=False)

    # Visualizing the number of stores by state
    st.markdown("### Number of Stores by State")
    fig, ax = plt.subplots()
    sns.barplot(data=stores_count, x="NUM_STORES", y="state", ax=ax, palette="viridis")
    ax.set_title("Number of Stores by State")
    ax.set_xlabel("Number of Stores")
    st.pyplot(fig)

    # Total Sales Time Series
    sns.set_theme(style="whitegrid")
    st.markdown("### Sales Trends Over Time")
    fig, ax = plt.subplots()
    total_sales_by_date = df_sales.groupby("date")["sales"].sum()
    # Use Seaborn's lineplot for a smoother line and better aesthetics
    sns.lineplot(data=total_sales_by_date, ax=ax, color="royalblue", linewidth=2.5)

    # Customizing the plot
    ax.set_title("Daily Sales Over Time", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Total Sales", fontsize=14)
    ax.tick_params(labelsize=12)  # Adjust to make tick labels larger

    plt.xticks(rotation=45)  # Rotate date labels for better readability
    plt.tight_layout()  # Adjust layout to not cut off labels

    st.pyplot(fig)

    # Conclusion and transition to next section
    st.markdown(
        """
    This overview provides a glimpse into the dataset's richness and the potential insights to be uncovered through our analysis. As we transition to exploratory data analysis, we'll dive deeper into these patterns and uncover the stories hidden within the sales data.
    """
    )


if __name__ == "__main__":
    show_data_overview()
