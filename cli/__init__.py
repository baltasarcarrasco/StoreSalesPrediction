"""
**CLI access point**.
"""

from typer import Typer
from cli.make_predictions import make_predictions
from cli.train import train
from cli.plot import plot

app = Typer()

app.command()(make_predictions)
app.command()(train)
app.command()(plot)
