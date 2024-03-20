import streamlit as st
from introduction import show_introduction
from data_overview import show_data_overview
from exploratory_data_analysis import show_eda
from feature_engineering_and_data_processing import show_feature_engineering_page
from model_training_and_evaluation import show_model_training_evaluation_page
from predictions_and_insights import show_predictions_and_insights
from sidebar_controls import create_sidebar_controls

# Call the centralized sidebar function to get user input values
store_nbr, product_family = create_sidebar_controls()

# Create tabs for each section of the app
tab1, tab2, tab3, tab4, tab5, tab6= st.tabs(
    [
        "Introduction",
        "Data Overview",
        "Exploratory Data Analysis",
        "Feature Engineering & Data Processing",
        "Model Training & Evaluation",
        "Predictions & Insights"
    ]
)

with tab1:
    show_introduction()

with tab2:
    show_data_overview()

with tab3:
    show_eda(store_nbr, product_family)

with tab4:
    show_feature_engineering_page()

with tab5:
    show_model_training_evaluation_page(store_nbr, product_family)

with tab6:
    show_predictions_and_insights(store_nbr, product_family)
