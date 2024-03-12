import pandas as pd
from data_reading import read_table

df = read_table('sales_processed')
df['DATE'] = pd.to_datetime(df['DATE'])

train = df[df['DATE'] < '2017-07-01']
test = df[df['DATE'] >= '2017-07-01']

X_train = train.drop(['ID', 'DATE', 'SALES'], axis = 1)
y_train = train['SALES']
