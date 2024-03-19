"""
**CLI access point**.
"""

import sys

sys.path.append("C:\\Users\\balta\\OneDrive\\Escritorio\\StoreSalesPrediction")
sys.path.append("C:\\Users\\balta\\OneDrive\\Escritorio")
from typer import Typer


from cli.make_predictions import app as make_predictions
from cli.train import app as train
from cli.plot import app as plot

app = Typer()

app.add_typer(
    make_predictions,
    name="make-predictions",
    help="Makes predictions for a specified date range.",
)
app.add_typer(train, name="train", help="Trains the sales prediction model.")
app.add_typer(plot, name="plot", help="Plots the predictions and actual values.")
app()
