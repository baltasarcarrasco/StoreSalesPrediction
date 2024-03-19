from data_processing_utilities import prepare_data, encode_data
from db_utilities import read_table, write_table


def process_data():
    stores_df = read_table("stores")
    sales_df = read_table("sales")
    holidays_df = read_table("holidays")

    # Prepare and encode the data
    df = prepare_data(df_sales=sales_df, df_stores=stores_df, df_holidays=holidays_df)
    df = encode_data(df)

    # Save the processed data to the database
    write_table(df, "sales_processed")

    # Sort the DataFrame by the DATE column to ensure the split respects the time series order
    df.sort_values("date", inplace=True)

    # Split the data into training and testing sets, respecting the time series order
    train = df.iloc[: int(0.8 * len(df))]
    test = df.iloc[int(0.8 * len(df)) :]

    # Save the training and testing sets to the database
    write_table(train, "sales_train")
    write_table(test, "sales_test")
