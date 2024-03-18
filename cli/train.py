import typer
from store_sales_prediction.model_training import train_model

app = typer.Typer()


@app.command(help="Trains the sales prediction model.")
def train(
    use_xgbooster: bool = typer.Option(
        ...,
        "--xgboost",
        "-x",
        help="Choose XGBoost as the model. Default is Random Forest.",
    )
):
    """
    Trains the sales prediction model.
    """
    typer.echo("Training the model...")
    train_model(user_requested_model=True, use_xgbooster=use_xgbooster)
    typer.echo("Model trained successfully!")
