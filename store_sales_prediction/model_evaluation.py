import pandas as pd
from sklearn.metrics import mean_squared_error
from store_sales_prediction.db_utilities import write_table, read_table
import joblib

# Load the saved model
model = joblib.load("models/store_sales_model.pkl")

# Load the test data
test = read_table("sales_test")

# Define the features and target variable
X_test = test.drop(["ID", "DATE", "SALES"], axis=1)
y_test = test["SALES"]

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Save predictions and actual values to the database
results = pd.DataFrame({"ID": test["ID"], "SALES_PRED": y_pred, "SALES_ACTUAL": y_test})
write_table(results, "sales_predictions")
