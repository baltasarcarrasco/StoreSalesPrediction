import streamlit as st
from store_sales_prediction.db_utilities import read_table
df_stores = read_table('stores')
df_sales = read_table('sales')

def create_sidebar_controls():
    # Generate store options for the selectbox
    store_options = ['All'] + sorted(df_stores['store_nbr'].unique())
    # Generate product family options for the selectbox
    product_family_options = ['All'] + sorted(df_sales['family'].unique())

    # Create sidebar controls
    st.sidebar.header('User Input Parameters')

    # Store number selection
    store_nbr = st.sidebar.selectbox('Store Number', options=store_options, index=0, key='store_number_selectbox')

    # Product family selection
    product_family = st.sidebar.selectbox('Product Family', options=product_family_options, index=0,
                                          key='product_family_selectbox')

    # Apply changes button
    apply_changes_btn = st.sidebar.button('Apply Changes', key='apply_changes_button')

    return store_nbr, product_family, apply_changes_btn
