import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from store_sales_prediction.db_utilities import read_table

df_sales = read_table('sales')

# Sidebar controls for user input
st.sidebar.header('User Input Parameters')
store_nbr = st.sidebar.text_input('Store Number', '')
product_family = st.sidebar.selectbox('Product Family', ['All'] + sorted(df_sales['family'].unique()))

# Filter data based on user input
if store_nbr and product_family != 'All':
    df_sales_filtered = df_sales[(df_sales['store_nbr'] == store_nbr) & (df_sales['family'] == product_family)]
elif store_nbr:
    df_sales_filtered = df_sales[df_sales['store_nbr'] == store_nbr]
elif product_family != 'All':
    df_sales_filtered = df_sales[df_sales['family'] == product_family]
else:
    df_sales_filtered = df_sales

# Remove outliers
Q1 = df_sales_filtered['sales'].quantile(0.25)
Q3 = df_sales_filtered['sales'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_sales_filtered = df_sales_filtered[(df_sales_filtered['sales'] >= lower_bound) & (df_sales_filtered['sales'] <= upper_bound)]

def show_eda():
    st.title("Exploratory Data Analysis")

    st.markdown("""
        This section of our application focuses on exploring the distribution of sales across various stores and product families. By examining the sales distribution, we aim to identify patterns, outliers, and overall trends that could inform our forecasting models. 

        Use the sidebar controls to filter the sales data by store number and product family.
    """)

    # Visualizations
    st.markdown("### Distribution of Sales")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Sales Histogram")
        fig, ax = plt.subplots()
        sns.histplot(df_sales_filtered['sales'], bins=50, kde=True, ax=ax)
        ax.set_title('Distribution of Sales (Without Outliers)')
        st.pyplot(fig)

    with col2:
        st.markdown("#### Sales Box Plot")
        fig, ax = plt.subplots()
        sns.boxplot(x=df_sales_filtered['sales'], ax=ax)
        ax.set_title('Sales Box Plot (Without Outliers)')
        st.pyplot(fig)

if __name__ == "__main__":
    show_eda()
