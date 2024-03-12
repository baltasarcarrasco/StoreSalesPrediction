import streamlit as st
from introduction import show_introduction
from data_overview import show_data_overview
#from exploratory_data_analysis import show_eda
#from feature_engineering_and_data_processing import show_feature_engineering
#from model_training_and_evaluation import show_model_training
#from predictions_and_insights import show_predictions
#from conclusion import show_conclusion

# Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ('Introduction', 'Data Overview', 'Exploratory Data Analysis',
                                    'Feature Engineering & Data Processing', 'Model Training & Evaluation',
                                    'Predictions & Insights', 'Conclusion'))

if section == 'Introduction':
    show_introduction()
elif section == 'Data Overview':
    show_data_overview()
