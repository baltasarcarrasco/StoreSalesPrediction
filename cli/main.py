import typer
import sys

sys.path.append("C:\\Users\\balta\\OneDrive\\Escritorio\\StoreSalesPrediction")
from StoreSalesPrediction.model_training import train_model
from StoreSalesPrediction.predictions_visualization import plot_predictions
from StoreSalesPrediction.model_evaluation import predict

app = typer.Typer()


@app.command(help="Makes predictions for a specified date range.")
def make_predictions(
    initial_date: str = typer.Option(
        ...,
        "--start-date",
        "-s",
        help="The initial date for making predictions. Format: YYYY-MM-DD. Dates should be within 2016-09-15 and 2017-08-15",
    ),
    n_days: int = typer.Option(
        ..., "--n-days", "-d", help="The number of days to make predictions for."
    ),
):
    """
    Makes predictions for a specified date range.
    """
    typer.echo(f"Making predictions from {initial_date} to {n_days} days ahead...")
    predict(initial_date, n_days)
    typer.echo("Predictions made successfully!")


@app.command(help="Plots the predictions and actual values.")
def plot():
    """
    Plots the predictions and actual values.
    """
    typer.echo("Plotting predictions...")
    plot_predictions()


@app.command(help="Trains the sales prediction model.")
def train(
    use_xgbooster: bool = typer.Option(
        False,
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


if __name__ == "__main__":
    app()
