from data_processing_pipeline import prepare_data, encodings
import data_reading
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

db_url = 'sqlite:///./store_sales.db'
engine = create_engine(db_url, pool_pre_ping= True)

SessionLocal = sessionmaker(
    autocommit = False,
    autoflush= False,
    bind=engine
)

stores_df = data_reading.read_table('stores')
sales_df = data_reading.read_table('sales')
oil_df = data_reading.read_table('oil')
holidays_df = data_reading.read_table('holidays')

df = prepare_data(df_sales=sales_df,df_stores=stores_df,df_holidays=holidays_df)
df = encodings(df)

columns = df.columns.tolist()  # This preserves the original column order from the DataFrame
column_types = []

# Specify the data types for specific columns
specific_column_types = {
    'ID': 'VARCHAR(30)',
    'DATE': 'DATETIME'
}

# Iterate over the columns to construct the column definitions with the adjusted types
for col in columns:
    if col in specific_column_types:
        # Use the specific data type if the column is explicitly defined
        col_type = specific_column_types[col]
    elif col.startswith('SALES'):
        # If the column name starts with "SALES", use FLOAT
        col_type = 'FLOAT'
    else:
        # Default to INTEGER for all other columns
        col_type = 'INTEGER'
    column_types.append(f"{col} {col_type}")

# Join all column definitions into a single string, maintaining their original order
columns_sql = ', '.join(column_types)

# Construct the full CREATE TABLE statement
create_table_sql = f"CREATE TABLE sales_processed ({columns_sql})"

with SessionLocal() as session:
    session.execute(text(create_table_sql))
