# -*- coding: utf-8 -*-

"""Console script for spatial_ops."""
import sys
import click
from .. version import __version__


@click.group()
def cli():
    pass


@cli.command(help='Show version and exit')
def version():
    click.echo(__version__)


if __name__ == "__main__":
    sys.exit(cli())  # pragma: no cover
