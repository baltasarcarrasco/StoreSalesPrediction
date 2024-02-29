from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd

db_url = 'sqlite:///Databases/store_sales.db'
engine = create_engine(db_url, pool_pre_ping= True)

SessionLocal = sessionmaker(
    autocommit = False,
    autoflush= False,
    bind = engine
)