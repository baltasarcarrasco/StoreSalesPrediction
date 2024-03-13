from data_processing_pipeline import prepare_data, encodings
import store_sales_prediction.db_utilities as db_utilities
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

db_url = "sqlite:///./store_sales.db"
engine = create_engine(db_url, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

stores_df = db_utilities.read_table("stores")
sales_df = db_utilities.read_table("sales")
oil_df = db_utilities.read_table("oil")
holidays_df = db_utilities.read_table("holidays")

df = prepare_data(df_sales=sales_df, df_stores=stores_df, df_holidays=holidays_df)
df = encodings(df)

columns = (
    df.columns.tolist()
)  # This preserves the original column order from the DataFrame
column_types = []

# Specify the data types for specific columns
specific_column_types = {"ID": "VARCHAR(30)", "DATE": "DATETIME"}

for col in columns:
    # Wrap column names in quotes to handle spaces
    quoted_col_name = f'"{col}"'  # Add quotes around column names

    if col in specific_column_types:
        # Use the specific data type if the column is explicitly defined
        col_type = specific_column_types[col]
    elif col.startswith("SALES"):
        # If the column name starts with "SALES", use FLOAT
        col_type = "FLOAT"
    else:
        # Default to INTEGER for all other columns
        col_type = "INTEGER"
    column_types.append(f"{quoted_col_name} {col_type}")

columns_sql = ", ".join(column_types)
create_table_sql = f"CREATE TABLE sales_processed ({columns_sql});"

with SessionLocal() as session:
    session.execute(text(create_table_sql))
    df.to_sql("sales_processed", session.get_bind(), if_exists="append", index=False)
