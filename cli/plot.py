import typer
from StoreSalesPrediction.predictions_visualization import plot_predictions

app = typer.Typer()


@app.command(help="Plots the predictions and actual values.")
def plot():
    """
    Plots the predictions and actual values.
    """
    typer.echo("Plotting predictions...")
    plot_predictions()
