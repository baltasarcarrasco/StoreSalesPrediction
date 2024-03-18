from db_utilities import read_table
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import joblib


def train_model(user_requested_model=False, use_xgbooster=True):
    """
    This function trains a model to predict store sales. It saves the model to the models directory.
    """
    # Read the training data from the database
    train_df = read_table("sales_train")

    # Define the features and target variable
    X_train = train_df.drop(["id", "date", "sales"], axis=1)
    y_train = train_df["sales"]

    model = XGBRegressor() if use_xgbooster else RandomForestRegressor()

    if not (user_requested_model):
        # Define the time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

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

        # Save the best model
        best_model = gsearch.best_estimator_
        joblib.dump(best_model, "../models/store_sales_model.pkl")
    else:
        model.fit(X_train, y_train)
        joblib.dump(model, "../models/user_model.pkl")
