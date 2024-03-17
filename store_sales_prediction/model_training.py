import pandas as pd
from db_utilities import read_table, write_table
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import joblib

# Read the processed sales data
df = read_table("sales_processed")

# Ensure the DATE column is in datetime format
df["date"] = pd.to_datetime(df["date"])

# Sort the DataFrame by the DATE column to ensure the split respects the time series order
df.sort_values("date", inplace=True)

# Split the data into training and testing sets, respecting the time series order
train = df.iloc[: int(0.8 * len(df))]
test = df.iloc[int(0.8 * len(df)) :]

# Save the training and testing sets to the database
write_table(train, "sales_train")
write_table(test, "sales_test")

# Define the features and target variable
X_train = train.drop(["id", "date", "sales"], axis=1)
y_train = train["sales"]

# Define the time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Define the model
model = XGBRegressor()

# Define the hyperparameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.3],
}

# Perform the grid search
gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_grid)

# Fit the grid search to the training data
gsearch.fit(X_train, y_train)

# Print the best hyperparameters
print(gsearch.best_params_)

# Save the best model
best_model = gsearch.best_estimator_
joblib.dump(best_model, "../models/store_sales_model.pkl")
