import typer
from store_sales_prediction.model_training import train_model

app = typer.Typer()


@app.command(help="Trains the sales prediction model.")
def train():
    """
    Trains the sales prediction model.
    """
    typer.echo("Training the model...")
    train_model()
    typer.echo("Model trained successfully!")
