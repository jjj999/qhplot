import typing as t

import click
import pydantic
import yaml

from .common import QHConfig
from .plots import (
    plot_bgleak,
    plot_fet,
    plot_ohmic,
    plot_ohmicr,
)


class QHPlotConfig(pydantic.BaseModel):

    bgleak: t.Optional[QHConfig] = None
    fet: t.Optional[QHConfig] = None
    ohmic: t.Optional[QHConfig] = None
    ohmicr: t.Optional[QHConfig] = None


def load_conf(path: str) -> QHPlotConfig:
    with open(path, "rt") as f:
        conf = yaml.load(f, Loader=yaml.Loader)
    return QHPlotConfig(**conf)


@click.command()
@click.argument("confpath", type=click.Path(exists=True))
def main(confpath: str) -> None:
    conf = load_conf(confpath)

    if conf.bgleak is not None:
        plot_bgleak(conf.bgleak)
    if conf.fet is not None:
        plot_fet(conf.fet)
    if conf.ohmic is not None:
        plot_ohmic(conf.ohmic)
    if conf.ohmicr is not None:
        plot_ohmicr(conf.ohmicr)
