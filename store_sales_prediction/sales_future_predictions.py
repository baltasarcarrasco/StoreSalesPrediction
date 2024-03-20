from store_sales_prediction.db_utilities import read_table, write_table
from store_sales_prediction.store_model_future_predictions_utilities import iterative_predictions
import joblib

df_sales = read_table('sales')
trained_model = joblib.load("./models/store_sales_model.pkl")
predictions = iterative_predictions(trained_model, sales_df=df_sales, input_date='2017-11-30')
write_table(predictions, 'predicted_sales')