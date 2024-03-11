import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd

# Calculate the path to the directory ABOVE the current script directory (i.e., the project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct the absolute path to the database, assuming the database is in the project root
db_path = os.path.join(PROJECT_ROOT, 'store_sales.db')
db_url = f'sqlite:///{db_path}'

engine = create_engine(db_url, pool_pre_ping=True)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

def read_table(table: str):
    ''' Reads the specified table from store_sales.db'''
    query = f'SELECT * FROM {table}'
    with SessionLocal() as session:
        q = text(query)
        df = pd.DataFrame(session.execute(q))

    return df