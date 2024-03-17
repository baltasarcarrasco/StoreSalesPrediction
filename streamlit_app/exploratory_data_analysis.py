import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from store_sales_prediction.db_utilities import read_table

df_sales = read_table('sales')
df_stores = read_table('stores')
df_sales['date'] = pd.to_datetime(df_sales['date'])

# Sidebar controls for user input
st.sidebar.header('User Input Parameters')
store_options = ['All'] + sorted(df_sales['store_nbr'].unique())
product_family_options = ['All'] + sorted(df_sales['family'].unique())

store_nbr = st.sidebar.selectbox('Store Number', options=store_options)
product_family = st.sidebar.selectbox('Product Family', options=product_family_options)

apply_changes_btn = st.sidebar.button('Apply Changes')


# Function to remove outliers from a given DataFrame
def remove_outliers(df, column='sales'):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply filters and remove outliers when "Apply Changes" is clicked
df_sales_filtered = df_sales.copy()
if apply_changes_btn:
    if store_nbr != 'All' and product_family != 'All':
        df_sales_filtered = df_sales[(df_sales['store_nbr'] == store_nbr) & (df_sales['family'] == product_family)]
    elif store_nbr != 'All':
        df_sales_filtered = df_sales[df_sales['store_nbr'] == store_nbr]
    elif product_family != 'All':
        df_sales_filtered = df_sales[df_sales['family'] == product_family]
    else:
        df_sales_filtered = df_sales.copy()

# Removing outliers only for the sales distribution visualization
df_sales_filtered_no_outliers = remove_outliers(df_sales_filtered)

def show_eda():
    st.title("Exploratory Data Analysis")
    st.markdown("""
        This section focuses on exploring the sales distribution and trends over time. Use the sidebar controls to filter the sales data by store number and product family. Changes will be applied after clicking the "Apply Changes" button.
    """)

    # Distribution Visualization with Outlier Removal
    st.markdown("## Sales Distribution")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Sales Histogram")
        fig, ax = plt.subplots()
        sns.histplot(df_sales_filtered_no_outliers['sales'], bins=50, kde=True)
        plt.title('Distribution of Sales (Without Outliers)')
        st.pyplot(fig)

    with col2:
        st.markdown("#### Sales Box Plot")
        fig, ax = plt.subplots()
        sns.boxplot(x=df_sales_filtered_no_outliers['sales'])
        plt.title('Sales Box Plot (Without Outliers)')
        st.pyplot(fig)

    # Time Series Analysis (Full Data, Without Outlier Removal)
    #if apply_changes_btn:
    st.markdown("## Time Series Analysis of Sales")
    st.markdown("### Sales Trends Over Time")
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots()
    total_sales_by_date = df_sales_filtered.groupby('date')['sales'].sum()
    sns.lineplot(data=total_sales_by_date, ax=ax, color="royalblue", linewidth=2.5)
    ax.set_title("Daily Sales Over Time", fontsize=16, fontweight='bold')
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Total Sales", fontsize=14)
    ax.tick_params(labelsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    show_eda()