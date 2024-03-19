import typer
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
