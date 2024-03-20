import streamlit as st
from store_sales_prediction.db_utilities import read_table

df_stores = read_table("stores")
df_sales = read_table("sales")


def create_sidebar_controls():
    # Generate store options for the selectbox
    store_options = ["All"] + sorted(df_stores["store_nbr"].unique())
    # Generate product family options for the selectbox
    product_family_options = ["All"] + sorted(df_sales["family"].unique())

    # Create sidebar controls
    st.sidebar.header("User Input Parameters")

    # Store number selection
    store_nbr = st.sidebar.selectbox(
        "Store Number", options=store_options, index=0, key="store_number_selectbox"
    )

    # Product family selection
    product_family = st.sidebar.selectbox(
        "Product Family",
        options=product_family_options,
        index=0,
        key="product_family_selectbox",
    )

    # Apply changes button with session state handling
    if st.sidebar.button("Apply Changes", key="apply_changes_button"):
        st.session_state["apply_changes"] = True
    else:
        if "apply_changes" not in st.session_state:
            # Default value before any interactions
            st.session_state["apply_changes"] = False

    # Initialize 'first_load' in session state if not already present
    if "first_load" not in st.session_state:
        st.session_state["first_load"] = True
    elif st.session_state["apply_changes"]:
        # Reset 'first_load' if 'Apply Changes' is clicked
        st.session_state["first_load"] = False

    return store_nbr, product_family
