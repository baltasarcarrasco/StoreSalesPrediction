import streamlit as st


def show_introduction():
    st.title("Favorita Store Sales Forecasting")

    # Introduction to the project
    st.markdown("""
    Welcome to our time-series forecasting project, where we delve into predicting the unit sales for thousands of items across the Favorita stores located in Ecuador. Corporación Favorita, a leading grocery retailer in Ecuador, presents a rich dataset comprising dates, store and product information, promotional activity, and unit sales.

    This web application serves a dual purpose:
    - **Presentation of Results**: Offering an in-depth look at our exploratory data analysis, feature engineering efforts, and the performance of our machine learning model.
    - **Interactive User Application**: Allowing users to interact with the model, explore predictions, and gain insights into sales forecasting for Favorita's diverse product range.

    Our challenge encompasses not just forecasting sales but also understanding the intricate relationships between various factors influencing sales across multiple store locations and product families.
    """)

    # Visuals
    st.markdown("### Project Overview")
    st.image("streamlit_app/images/corporacion-la-favorita.jpg", caption="Retail Environment at Favorita Stores")

    #st.markdown("### Exploratory Data Analysis Highlights")
    #st.image("path_to_your_eda_visual.jpg", caption="Sample EDA Visual")

    # App Sections Overview
    st.markdown("### App Sections")
    st.markdown("""
        Navigate through the app using the sidebar to explore:
        - **Introduction**: A warm welcome and overview of the project.
        - **Data Overview**: Insights into the raw data feeding our analyses.
        - **Exploratory Data Analysis**: Visual explorations of sales trends, seasonality, and more.
        - **Feature Engineering and Data Processing**: A look into how we've prepared the data for modeling.
        - **Model Training and Evaluation**: Our model's performance and the insights we've gathered.
        - **Predictions and Insights**: Interactive predictions and key findings from our analysis.
        """)

    # Introduction to the team
    st.markdown("### Meet the Team")
    st.markdown("""
    - **Iran Benitez**
    - **Abigail Carpenter**
    - **Baltasar Carrasco**
    - **Leonardo Dulcetti**
    - **Victor Gonzalez**
    - **Diego Leon**
    """)


    st.markdown("""
    We invite you to navigate through the app using the sidebar to explore our data, methodology, model insights, and interactive predictions. Enjoy the journey through data to actionable insights!
    """)


# This ensures the function runs when the script is executed directly, useful for testing
#if __name__ == "__main__":
 #   show_introduction()

