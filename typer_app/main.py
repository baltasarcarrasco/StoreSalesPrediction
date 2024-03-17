import typer
from functions import train_model, predict, plot_predictions

app = typer.Typer()


@app.command()
def train():
    typer.echo("Training the model...")
    train_model()
    typer.echo("Model trained successfully!")


@app.command()
def make_predictions(initial_date: str, n_days: int):
    typer.echo(
        f"Making predictions for the date range: {initial_date} to {initial_date + pd.Timedelta(days=n_days)}"
    )
    predict(initial_date, n_days)
    typer.echo("Predictions made successfully!")


@app.command()
def plot():
    typer.echo("Plotting predictions...")
    plot_predictions()


if __name__ == "__main__":
    app()
