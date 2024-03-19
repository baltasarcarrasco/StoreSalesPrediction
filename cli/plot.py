from typer import Typer, echo
from StoreSalesPrediction.predictions_visualization import plot_predictions

# app = Typer()


# @app.command(help="Plots the predictions and actual values.")
def plot():
    """
    Plots the predictions and actual values.
    """
    echo("Plotting predictions...")
    plot_predictions()
