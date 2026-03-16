"""
Command-line interface for inspectrum.

Example usage:
    $ python -m inspectrum.cli
    $ python -m inspectrum.cli --name "World"
"""

import click


@click.command()
@click.option("--name", default="AI", help="Name to greet (default: AI)")
@click.version_option(version="0.1.0")
def main(name: str):
    """
    A simple greeting CLI example.

    This demonstrates the basic structure of a Click CLI.
    Replace this with your actual functionality!

    Example:
        python -m inspectrum.cli
        python -m inspectrum.cli --name "Copilot"
    """
    click.echo(f"Hello, {name}! 👋")
    click.echo("\nThis is a template CLI. Replace it with your own commands!")
    click.echo("Tip: Ask Copilot to help you add functionality here.")


if __name__ == "__main__":
    main()
