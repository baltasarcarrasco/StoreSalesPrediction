from data_processing_pipeline import prepare_data, encodings
import db_utilities as db_utilities

stores_df = db_utilities.read_table("stores")
sales_df = db_utilities.read_table("sales")
oil_df = db_utilities.read_table("oil")
holidays_df = db_utilities.read_table("holidays")

df = prepare_data(df_sales=sales_df, df_stores=stores_df, df_holidays=holidays_df)
df = encodings(df)

db_utilities.write_table(df, "sales_processed")
