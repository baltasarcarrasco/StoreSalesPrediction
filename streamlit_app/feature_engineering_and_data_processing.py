import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from store_sales_prediction.db_utilities import read_table, custom_query

df_sales = read_table("sales")
df_sales["date"] = pd.to_datetime(df_sales["date"])
sales_processed_data = custom_query(query="SELECT * FROM sales_processed LIMIT 5")


def show_feature_engineering_page():
    st.title("Feature Engineering and Data Processing")

    st.markdown(
        """
    This page outlines the steps taken to prepare our dataset for machine learning modeling. Feature engineering is a crucial step in the data science workflow. It involves creating new features from existing ones to better highlight underlying patterns in the data for predictive modeling.

    Our approach includes the following steps:
    - **Lagged Features**: To capture sales trends and seasonality.
    - **Rolling Window Features**: To smooth out short-term fluctuations and highlight longer-term trends.
    - **Encoding Categorical Variables**: To transform categorical variables into a format that can be provided to machine learning algorithms.
    - **Handling Holidays**: As holidays have a significant impact on sales, we incorporate holiday information into our model.
    - **Store Information**: Store attributes like type and cluster are included as they can affect sales patterns.
    """
    )

    st.markdown("## Feature Engineering Pipeline")
    st.code(
        """
# Sample code snippet from our pipeline
def prepare_data(df_sales, df_stores, df_holidays, df_oil=None):
    # Merging, encoding, and creating lagged features...

def encodings(df):
    # Encoding categorical variables...
    """,
        language="python",
    )

    st.markdown("### Lagged and Rolling Window Features")
    st.markdown(
        """
    We decided on specific lagged and rolling window features based on Autocorrelation (ACF) and Partial Autocorrelation (PACF) analyses, which are essential components in time series forecasting. These features help our model understand and predict based on past sales data.
    """
    )

    # Aggregate sales by date to get the total daily sales
    daily_sales = df_sales.groupby("date")["sales"].sum().reset_index()
    # First, ensure 'DATE' is the DataFrame index for easier differencing
    daily_sales.set_index("date", inplace=True)
    # Apply first difference to the daily sales series
    daily_sales["sales_diff"] = daily_sales["sales"].diff()
    # Drop the first row as it will be NaN after differencing
    daily_sales_diff = daily_sales.dropna()

    # Plotting the first-differenced sales series
    st.markdown("#### First-Differenced Daily Sales Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_sales_diff.index, daily_sales_diff["sales_diff"])
    ax.set_title("First-Differenced Daily Sales Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("First Difference of Sales")
    st.pyplot(fig)

    # ACF and PACF plots
    st.markdown("#### ACF and PACF for First-Differenced Sales")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Autocorrelation Function (ACF)")
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_acf(daily_sales_diff["sales_diff"], lags=20, ax=ax)
        st.pyplot(fig)

    with col2:
        st.markdown("##### Partial Autocorrelation Function (PACF)")
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_pacf(daily_sales_diff["sales_diff"], lags=20, ax=ax)
        st.pyplot(fig)

    st.markdown("### Handling Categorical Data")
    st.markdown(
        """
    Categorical data such as store type, holidays, and product families are encoded to numerical values. This process is crucial for machine learning models as they require numerical input.
    """
    )
    # Listing the encodings used for each categorical variable
    st.markdown(
        """
        - **Store Type**: Ordinal Encoding. Different store types (A, B, C, D, E) are encoded with ordinal values reflecting their inherent order or importance.
        - **Holiday Type**: Ordinal Encoding. Holidays are categorized (e.g., No Holiday, Local, Regional, National) and encoded with ordinal values to capture the potential different impacts on sales.
        - **Product Family**: One-Hot Encoding. To capture the uniqueness of each product family without implying any order, one-hot encoding is utilized, creating a binary column for each category.
        - **Cluster**: Original Encoding Kept. The cluster information, which groups stores based on similar characteristics, is kept as is without additional encoding, assuming it's already in a numerical format suitable for modeling.
        """
    )

    # Display preview of the processed data
    st.markdown("## Processed Data Preview")
    st.write(sales_processed_data.head())


if __name__ == "__main__":
    show_feature_engineering_page()
