import pandas as pd
from db_utilities import read_table
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

stores_data = read_table("stores")
holidays_data = read_table("holidays")
sales_raw_data = read_table("sales")


def prepare_data(df_sales, df_stores, df_holidays, df_oil=None):
    # Merge store information with sales data
    df_sales = pd.merge(
        df_sales,
        df_stores[["store_nbr", "type", "cluster"]],
        on="store_nbr",
        how="left",
    )

    # Ensure DATE columns are in datetime format
    df_sales["date"] = pd.to_datetime(df_sales["date"])
    df_holidays["date"] = pd.to_datetime(df_holidays["date"])
    # df_oil['DATE'] = pd.to_datetime(df_oil['DATE'])

    # Lag oil price by one day
    # df_oil['OIL_PRICE_LAG1'] = df_oil['PRICE'].shift(1)

    # Merge lagged oil price with sales data
    # df_sales = pd.merge(df_sales, df_oil[['DATE', 'OIL_PRICE_LAG1']], on='DATE', how='left')

    # Merge holidays data
    df_sales = pd.merge(
        df_sales, df_holidays[["date", "locale"]], on="date", how="left"
    )

    df_sales = df_sales.fillna({"locale": "No Holiday"})

    # Sort the DataFrame by STORE_NBR, FAMILY, and DATE to ensure the lagged features are calculated correctly
    df_sales.sort_values(["store_nbr", "family", "date"], inplace=True)

    # Adding lagged sales features
    for lag in range(1, 6):  # 1 to 5 normal lags
        df_sales[f"sales_lag_{lag}"] = df_sales.groupby(["store_nbr", "family"])[
            "sales"
        ].shift(lag)

    # Adding seasonal lag
    df_sales["sales_seas_lag_1"] = df_sales.groupby(["store_nbr", "family"])[
        "sales"
    ].shift(7)

    # Adjusted to include rolling means for lags 1 through 5
    for lag in range(1, 6):  # Lags 1 to 5
        df_sales[f"sales_rolling_mean_{lag}"] = (
            df_sales.groupby(["store_nbr", "family"])["sales"]
            .shift(1)
            .rolling(window=lag)
            .mean()
        )

    # Adjusting seasonal MA components
    df_sales["sales_seasonal_rolling_mean_7"] = (
        df_sales.groupby(["store_nbr", "family"])["sales"]
        .shift(1)
        .rolling(window=7)
        .mean()
    )
    df_sales["sales_seasonal_rolling_mean_14"] = (
        df_sales.groupby(["store_nbr", "family"])["sales"]
        .shift(1)
        .rolling(window=14)
        .mean()
    )

    # Drop NaNs as they correspond to initial dates

    df_sales = df_sales.dropna()

    return df_sales


def encodings(df):
    # Ordinal encoding for store type
    store_type_encoder = OrdinalEncoder(categories=[["A", "B", "C", "D", "E"]])
    store_type_encoder.fit(stores_data[["type"]])
    df["type"] = store_type_encoder.transform(df[["type"]])

    # Ordinal encoding for holiday type
    holiday_type_encoder = OrdinalEncoder(
        categories=[["No Holiday", "Local", "Regional", "National"]]
    )
    holiday_type_encoder.fit(holidays_data[["locale"]])
    df["locale"] = holiday_type_encoder.transform(df[["locale"]])

    # One-hot encoding for product family
    unique_families = sales_raw_data["family"].unique().reshape(-1, 1)
    family_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    family_ohe.fit(unique_families)

    family_encoded = family_ohe.transform(df[["family"]])
    family_encoded_df = pd.DataFrame(
        family_encoded, columns=family_ohe.get_feature_names_out(["family"])
    )

    # Ensure the index aligns for concatenation
    family_encoded_df.index = df.index
    # Concatenate the one-hot encoded DataFrame with the original DataFrame
    df = pd.concat([df, family_encoded_df], axis=1)

    # drop the original 'FAMILY' column
    df.drop(["family"], axis=1, inplace=True)

    return df
