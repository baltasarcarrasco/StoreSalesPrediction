from typer import echo, Typer
from store_sales_prediction.predictions_visualization import plot_predictions

app = Typer()


@app.command(help="Plots the predictions and actual values.")
def plot():
    """
    Plots the predictions and actual values.
    """
    echo("Plotting predictions...")
    plot_predictions()
