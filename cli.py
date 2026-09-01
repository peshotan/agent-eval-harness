"""Command-line entry point for the evaluation harness."""

from typing import Annotated

import typer

from harness import __version__

app = typer.Typer(
    name="agent-eval",
    help="Evaluate language models and tool-using agents.",
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    """Print the package version and exit."""
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=version_callback, is_eager=True),
    ] = None,
) -> None:
    """Run evaluation commands."""


if __name__ == "__main__":
    app()
