from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd

db_url = 'sqlite:///./store_sales.db'
engine = create_engine(db_url, pool_pre_ping= True)

SessionLocal = sessionmaker(
    autocommit = False,
    autoflush= False,
    bind=engine
)

create_table_sales = text('CREATE TABLE sales (ID VARCHAR(30), DATE DATETIME, STORE_NBR VARCHAR(30), FAMILY VARCHAR(30), SALES FLOAT, ONPROMOTION INTEGER)')
create_table_stores = text('CREATE TABLE stores (STORE_NBR VARCHAR(30), CITY VARCHAR(30), STATE VARCHAR(30), TYPE VARCHAR(3), CLUSTER INTEGER)')
create_table_transactions = text('CREATE TABLE transactions (DATE DATETIME, STORE_NBR VARCHAR(30), TRANSACTIONS BIGINT)')
create_table_oil = text('CREATE TABLE oil (DATE DATETIME, DCOILWTICO BIGINT)')
create_table_holidays = text('CREATE TABLE holidays (DATE DATETIME, TYPE VARCHAR(30), LOCALE VARCHAR(30), LOCALE_NAME VARCHAR(30), DESCRIPTION VARCHAR(30), TRANSFERRED BOOLEAN)')

with SessionLocal() as session:
    for i in [create_table_sales,
              create_table_stores,
              create_table_transactions,
              create_table_oil,
              create_table_holidays]:
        session.execute(i)
        

    sales = pd.read_csv(r'./data/train.csv')
    stores = pd.read_csv(r'./data/stores.csv')
    transactions = pd.read_csv(r'./data/transactions.csv')
    oil = pd.read_csv(r'./data/oil.csv')
    holidays = pd.read_csv(r'./data/holidays_events.csv')

    dfs_dict = {'sales':sales,
                'stores':stores,
                'transactions':transactions,
                'oil':oil,
                'holidays':holidays}

    for key, df in dfs_dict.items():
        df.to_sql(key,
                  session.get_bind(),
                  if_exists='append',
                  index=False)