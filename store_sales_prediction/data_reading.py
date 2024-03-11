from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd

db_url = 'sqlite:///../store_sales.db'
engine = create_engine(db_url, pool_pre_ping= True)

SessionLocal = sessionmaker(
    autocommit = False,
    autoflush= False,
    bind = engine
)

def read_table(table: str):
    ''' Reads the specified table from store_sales.db'''
    query = f'SELECT * FROM {table}'
    with SessionLocal() as session:
        q = text(query)
        df = pd.DataFrame(session.execute(q))

    return df