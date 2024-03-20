# Store Sales Prediction
[GitHub Repository](https://github.com/baltasarcarrasco/StoreSalesPrediction.git)


This project aims to predict the sales of a store and a family of products using historical sales data. The project uses a dataset from the [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview) competition on Kaggle. The dataset contains sales data for Favorita stores located in Ecuador. The training data includes dates, store and product information, whether that item was being promoted, as well as the sales numbers. The project uses a time series model to predict the sales of a store and a family of products for the next days. The project also includes a Streamlit app for data visualization and model predictions, and a Typer CLI app for personalized model training and predictions.

## Data Processing
The dataset was preprocessed in order to capture the most relevant information for the time series model. 
Our approach includes the following steps:  
    - **Lagged Features**: To capture sales trends and seasonality.  
    - **Rolling Window Features**: To smooth out short-term fluctuations and highlight longer-term trends.  
    - **Encoding Categorical Variables**: To transform categorical variables into a format that can be provided to machine learning algorithms.  
    - **Handling Holidays**: As holidays have a significant impact on sales, we incorporate holiday information into our model.  
    - **Store Information**: Store attributes like type and cluster are included as they can affect sales patterns.  

## Model Overview

The model used is an XGBRegressor from the XGBoost library. The model is trained using the first 80% of the data (chronologically ordered) and the sales numbers for the next days are predicted. The model hyperparameters were tuned using a grid search and TimeSeriesSplit cross-validation. The pipeline for creating the model is defined in `store_sales_prediction/store_model_pipeline.py`. The model is saved to the `models/store_sales_model.pkl` file using the joblib library.

## Project Structure

- `data/`: Contains data files like `holidays_events.csv`, `oil.csv`, `stores.csv`, `transactions.csv`.
- `models/`: Contains the trained models `store_sales_model.pkl` and the user defined model.
- `notebooks/`: Contains Jupyter notebooks for exploratory data analysis.
- `store_sales_prediction/`: Contains Python scripts for data ingestion, processing, model training, and evaluation.
- `streamlit_app/`: Contains Streamlit app for data visualization and model predictions.
- `cli/`: Contains Typer CLI app for model training and predictions.

## Dependencies

The project uses several Python libraries like pandas, sqlalchemy, jupyter, matplotlib, seaborn, statsmodels, scikit-learn, streamlit, xgboost, joblib, and typer. The exact versions of these libraries are specified in the `pyproject.toml` and `poetry.lock` files.

### Setup

To set up the project, first install the dependencies using Poetry:

```sh
poetry install
```
    
Then, activate the virtual environment:

```sh
poetry shell
```
## Initial Data Setup for Cloned Repositories

If you've cloned this repository directly from GitHub, please note that the content of the database (`store_sales.db`) and the raw data file (`train.csv`) used for data ingestion and processing are not included due to file size limitations on GitHub. To fully utilize this project, including running the CLI and Streamlit app, you'll need to manually add these files to your local setup.

### Adding the Raw Data File

1. Download the `train.csv` file from this [link](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).
2. Place the downloaded `train.csv` file into the `data/` directory of your local project.

### Populating the Database

Once you have the `train.csv` file in place, you need to populate the `store_sales.db` database:

1. Run the `store_model_pipeline.py` script to process the raw data and populate the database:

```sh
python store_sales_prediction/store_model_pipeline.py
```

This will fill the `store_sales.db` database with the necessary tables and data required for the project.

### Running the CLI and Streamlit App

After completing the above steps, the database should be populated with the necessary data. You're now ready to use the Typer CLI app and the Streamlit web app. Follow the setup instructions in the following sections of this README to get started.

## Usage

The Streamlit app allows users to visualize the data and make predictions using the trained model. The Typer app allows users to train the model and make predictions using the command line.

### Streamlit App

The Streamlit app provides an interactive interface for exploring the data and visualizing the model's predictions. It is structured into various pages, each focusing on a different aspect of the data and predictions:

- **Introduction**: Gives a brief overview of the project and its objectives.
- **Data Overview**: Displays the raw data and its characteristics, allowing users to understand the dataset used for modeling.
- **Exploratory Data Analysis (EDA)** *(Interactive)*: Offers detailed insights into the sales data, enabling users to explore trends, patterns, and relationships.
- **Feature Engineering & Data Processing**: Explains the steps taken to prepare the data for modeling, including feature engineering and encoding.
- **Model Training & Evaluation** *(Interactive)*: Details the modeling process, including hyperparameter tuning and model evaluation, with metrics to assess the model's performance.
- **Predictions & Insights** *(Interactive)*: Enables users to input a future date and visualize the model's sales predictions up to that date, alongside the actual sales data for the last 30 days. This page uses a combination of user input from sidebar controls and a date input widget to filter data and generate predictions.

***Note:*** The app also integrates global sidebar controls for filtering data based on store number and product family across different pages in the *Interactive* pages, enhancing the user experience and providing consistent analysis criteria throughout the application. User changes are reflected upon clicking the "Apply Changes" button in the sidebar.

To run the Streamlit app, execute the command:

```sh
streamlit run streamlit_app/app.py
```
### Typer App

The Typer app provides a command line interface for training the model and making predictions. The app has three commands: `train`, `make-predictions` and `plot`.

To run the app in a standalone way:

```sh
store_sales_prediction
```
To train the model, run:

```sh
store_sales_prediction train [--xgboost]
```

To make predictions, run:

```sh
store_sales_prediction make-predictions --start-date [start_date] --n-days [n_days]
```

To plot the sales predictions vs actual sales, run:

```sh
store_sales_prediction plot
```