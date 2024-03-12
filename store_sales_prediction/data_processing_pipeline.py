import pandas as pd
import data_reading
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

stores_data = data_reading.read_table('stores')
holidays_data = data_reading.read_table('holidays')
sales_raw_data = data_reading.read_table('sales')

def prepare_data(df_sales, df_stores, df_holidays, df_oil = None):
    # Merge store information with sales data
    df_sales = pd.merge(df_sales, df_stores[['STORE_NBR', 'TYPE', 'CLUSTER']], on='STORE_NBR', how='left')

    # Ensure DATE columns are in datetime format
    df_sales['DATE'] = pd.to_datetime(df_sales['DATE'])
    df_holidays['DATE'] = pd.to_datetime(df_holidays['DATE'])
    #df_oil['DATE'] = pd.to_datetime(df_oil['DATE'])

    # Lag oil price by one day
    #df_oil['OIL_PRICE_LAG1'] = df_oil['PRICE'].shift(1)

    # Merge lagged oil price with sales data
    #df_sales = pd.merge(df_sales, df_oil[['DATE', 'OIL_PRICE_LAG1']], on='DATE', how='left')

    #Merge holidays data
    df_sales = pd.merge(df_sales,
                        df_holidays[['DATE', 'LOCALE']],
                        on = 'DATE',
                        how = 'left')

    df_sales = df_sales.fillna({'LOCALE': 'No Holiday'})

    # Adding lagged sales features
    for lag in range(1, 6):  # 1 to 5 normal lags
        df_sales[f'SALES_LAG_{lag}'] = df_sales.groupby(['STORE_NBR', 'FAMILY'])['SALES'].shift(lag)

    # Adding seasonal lag
    df_sales['SALES_SEAS_LAG_1'] = df_sales.groupby(['STORE_NBR', 'FAMILY'])['SALES'].shift(7)

    # Adjusted to include rolling means for lags 1 through 5
    for lag in range(1, 6):  # Lags 1 to 5
        df_sales[f'ROLLING_MEAN_{lag}'] = df_sales.groupby(['STORE_NBR', 'FAMILY'])['SALES'].shift(1).rolling(
            window=lag).mean()

    # Adjusting seasonal MA components
    df_sales['SEASONAL_ROLLING_MEAN_7'] = df_sales.groupby(['STORE_NBR', 'FAMILY'])['SALES'].shift(1).rolling(
        window=7).mean()
    df_sales['SEASONAL_ROLLING_MEAN_14'] = df_sales.groupby(['STORE_NBR', 'FAMILY'])['SALES'].shift(1).rolling(
        window=14).mean()

    #Drop NaNs as they correspond to initial dates

    df_sales = df_sales.dropna()

    return df_sales


def encodings(df):
    # Ordinal encoding for store type
    store_type_encoder = OrdinalEncoder(categories=[['A', 'B', 'C', 'D', 'E']])
    store_type_encoder.fit(stores_data[['TYPE']])
    df['TYPE'] = store_type_encoder.transform(df[['TYPE']])

    # Ordinal encoding for holiday type
    holiday_type_encoder = OrdinalEncoder(categories=[['No Holiday', 'Local', 'Regional', 'National']])
    holiday_type_encoder.fit(holidays_data[['LOCALE']])
    df['LOCALE'] = holiday_type_encoder.transform(df[['LOCALE']])

    # One-hot encoding for product family
    unique_families = sales_raw_data['FAMILY'].unique().reshape(-1, 1)
    family_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    family_ohe.fit(unique_families)

    family_encoded = family_ohe.transform(df[['FAMILY']])
    family_encoded_df = pd.DataFrame(family_encoded, columns=family_ohe.get_feature_names_out(['FAMILY']))

    # Ensure the index aligns for concatenation
    family_encoded_df.index = df.index
    # Concatenate the one-hot encoded DataFrame with the original DataFrame
    df = pd.concat([df, family_encoded_df], axis=1)

    # drop the original 'FAMILY' column
    df.drop(['FAMILY'], axis=1, inplace=True)

    return df



