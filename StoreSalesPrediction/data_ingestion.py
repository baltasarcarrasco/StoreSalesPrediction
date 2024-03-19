import pandas as pd
from StoreSalesPrediction.db_utilities import write_table


def ingest_data():
    """
    This function reads the data from the CSV files and writes the DataFrames to the database.
    """
    # Read the data from the CSV files
    sales = pd.read_csv(r"./data/train.csv")
    stores = pd.read_csv(r"./data/stores.csv")
    transactions = pd.read_csv(r"./data/transactions.csv")
    oil = pd.read_csv(r"./data/oil.csv")
    holidays = pd.read_csv(r"./data/holidays_events.csv")

    # Write the DataFrames to the database
    write_table(sales, "sales")
    write_table(stores, "stores")
    write_table(transactions, "transactions")
    write_table(oil, "oil")
    write_table(holidays, "holidays")
