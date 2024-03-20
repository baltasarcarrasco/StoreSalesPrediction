from typer import Typer, echo, Option
from store_sales_prediction.model_training import train_model

app = Typer()


@app.command(help="Trains the sales prediction model.")
def train(
    use_xgbooster: bool = Option(
        False,
        "--xgboost",
        "-x",
        help="Choose XGBoost as the model. Default is Random Forest.",
    ),
):
    """
    Trains the sales prediction model.
    """
    echo("Training the model...")
    train_model(user_requested_model=True, use_xgbooster=use_xgbooster)
    echo("Model trained successfully!")
