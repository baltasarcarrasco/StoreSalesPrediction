# Store Sales Prediction

This project aims to predict store sales using various data sources and machine learning techniques.

## Project Structure

- `data/`: Contains data files like `holidays_events.csv`, `oil.csv`, `stores.csv`, `transactions.csv`.
- `models/`: Contains the trained models `store_sales_model.pkl` and the user defined model.
- `notebooks/`: Contains Jupyter notebooks for exploratory data analysis.
- `store_sales_prediction/`: Contains Python scripts for data ingestion, processing, model training, and evaluation.
- `streamlit_app/`: Contains Streamlit app for data visualization and model predictions.
- `typer_app/`: Contains Typer CLI app for model training and predictions.

## Dependencies

The project uses several Python libraries like pandas, sqlalchemy, jupyter, matplotlib, seaborn, statsmodels, scikit-learn, streamlit, xgboost, joblib, and typer. The exact versions of these libraries are specified in the `pyproject.toml` and `poetry.lock` files.

## Setup

To set up the project, first install the dependencies using Poetry:

```sh
poetry install
```
    
Then, activate the virtual environment:

```sh
poetry shell
```

Finally, run the Streamlit app:

```sh
streamlit run streamlit_app/app.py
```

or the Typer app:

```sh
python typer_app/app.py
```

## Usage

The Streamlit app allows users to visualize the data and make predictions using the trained model. The Typer app allows users to train the model and make predictions using the command line.

### Streamlit App
... Diego

### Typer App

The Typer app provides a command line interface for training the model and making predictions. The app has three commands: `train`, `make-predictions` and `plot`.

To train the model, run:

```sh
python typer_app/app.py train
```

To make predictions, run:

```sh
python typer_app/app.py make-predictions --start-date [start_date] --n-days [n_days]
```

To plot the sales predictions vs actual sales, run:

```sh
python typer_app/app.py plot
```


