import streamlit as st

def show_introduction():
    st.title("Introduction")
    st.write("""
    Welcome to the Sales Prediction Dashboard. This application provides insights into sales data, 
    explores various factors influencing sales, and predicts future sales based on historical data.

    Please use the navigation on the left to explore different sections of the analysis.
    """)
