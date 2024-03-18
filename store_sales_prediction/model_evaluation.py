import pandas as pd
from sklearn.metrics import mean_squared_error
from db_utilities import write_table, read_table
import joblib

# Load the saved model
model = joblib.load("./models/store_sales_model.pkl")

# Load the test data
test = read_table("sales_test")

# Define the features and target variable
X_test = test.drop(["id", "date", "sales"], axis=1)
y_test = test["sales"]

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Save predictions and actual values to the database
results = pd.DataFrame({"id": test["id"], "sales_pred": y_pred, "sales_actual": y_test})
write_table(results, "sales_predictions")
