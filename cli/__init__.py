"""
**CLI access point**.
"""

from typer import Typer

from store_sales_prediction.cli.make_predictions import app as make_predictions
from store_sales_prediction.cli.train import app as train
from store_sales_prediction.cli.plot import app as plot


app = Typer()
app.add_typer(make_predictions, name="make-predictions")
app.add_typer(train, name="train")
app.add_typer(plot, name="plot")
