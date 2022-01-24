import typing as t

import matplotlib.pyplot as plt
import numpy as np

from .common import (
    PlotConfig,
    QHConfig,
    get_scale_factor,
    scale2alpha,
)


PlotFunc_t = t.Callable[[np.ndarray, np.ndarray, PlotConfig, QHConfig], None]


def plot(
    conf: QHConfig,
    plot_func: PlotFunc_t,
    default_xaxis: str,
    unit_xaxis: str,
    default_yaxis: str,
    unit_yaxis: str,
) -> None:
    """Base plot function.

    Args:
        conf: Content of given config file.
        plot_func: Plot function for each purpose.
        default_xaxis: Default name of x-axis.
        default_yaxis: Default name of y-axis.

    Notes:
        `default_xaxis` and `default_yaxis` must be formattable for
        their scales.
    """
    has_labels = False
    for plot_conf in conf.plots:
        voltage = np.loadtxt(plot_conf.file_V)
        current = np.loadtxt(plot_conf.file_I)

        # Remove the offset.
        if plot_conf.remove_offset:
            index = np.where(np.abs(voltage) <= conf.offset_tolerance)[0][0]
            current -= current[index]

        plot_func(voltage, current, plot_conf, conf)
        if plot_conf.label is not None:
            has_labels = True

    # Set fontsizes
    default_fontsize = plt.rcParams["font.size"]
    if conf.fontsize is None:
        conf.fontsize = default_fontsize
    if conf.fontsize_title is None:
        conf.fontsize_title = conf.fontsize
    if conf.fontsize_xaxis is None:
        conf.fontsize_xaxis = conf.fontsize
    if conf.fontsize_yaxis is None:
        conf.fontsize_yaxis = conf.fontsize

    # Set names of each axes
    scale_x = scale2alpha(conf.scale_x)
    scale_y = scale2alpha(conf.scale_y)
    if conf.xaxis is None:
        conf.xaxis = " ".join((
            f"${default_xaxis}", r"\quad \mathrm{[",
            scale_x, unit_xaxis, r"]}$"))
    if conf.yaxis is None:
        conf.yaxis = " ".join((
            f"${default_yaxis}", r"\quad \mathrm{[",
            scale_y, unit_yaxis, r"]}$"))

    if conf.title is not None:
        plt.title(conf.title, fontsize=conf.fontsize_title)
    plt.xlabel(conf.xaxis, fontsize=conf.fontsize_xaxis)
    plt.ylabel(conf.yaxis, fontsize=conf.fontsize_yaxis)

    if conf.show_legend and has_labels:
        plt.legend()

    plt.savefig(
        conf.img.output,
        dpi=conf.img.dpi,
        bbox_inches="tight" if conf.img.tight else None)
    plt.clf()


def _plot_bgleak(
    voltage: np.ndarray,
    current: np.ndarray,
    plot_conf: PlotConfig,
    conf: QHConfig,
) -> None:
    scale_V = get_scale_factor(conf.scale_x, conf.raw_scale_V)
    scale_I = get_scale_factor(conf.scale_y, conf.raw_scale_I)
    plt.plot(
        voltage * scale_V,
        current * scale_I,
        color=plot_conf.color,
        label=plot_conf.label)


def plot_bgleak(conf: QHConfig) -> None:
    plot(conf, _plot_bgleak, "V", "V", "I", "A")

def _plot_fet(
    voltage: np.ndarray,
    current: np.ndarray,
    plot_conf: PlotConfig,
    conf: QHConfig,
) -> None:
    scale_V = get_scale_factor(conf.scale_x, conf.raw_scale_V)
    scale_I = get_scale_factor(conf.scale_y, conf.raw_scale_I)
    plt.plot(
        voltage * scale_V,
        current * scale_I,
        color=plot_conf.color,
        label=plot_conf.label)


def plot_fet(conf: QHConfig) -> None:
    plot(conf, _plot_fet, "V", "V", "I", "A")


def _plot_ohmic(
    voltage: np.ndarray,
    current: np.ndarray,
    plot_conf: PlotConfig,
    conf: QHConfig,
) -> None:
    scale_V = get_scale_factor(conf.scale_x, conf.raw_scale_V)
    scale_I = get_scale_factor(conf.scale_y, conf.raw_scale_I)
    plt.plot(
        voltage * scale_V,
        current * scale_I,
        color=plot_conf.color,
        label=plot_conf.label)


def plot_ohmic(conf: QHConfig) -> None:
    plot(conf, _plot_ohmic, "V", "V", "I", "A")


def _plot_ohmicr(
    voltage: np.ndarray,
    current: np.ndarray,
    plot_conf: PlotConfig,
    conf: QHConfig,
) -> None:
    delta_V = voltage[1] - voltage[0]
    resistance = ((current[2:] - current[:-2]) / (2 * delta_V))**(-1)
    scale_V = get_scale_factor(conf.scale_x, conf.raw_scale_V)
    default_scale_R = get_scale_factor(conf.raw_scale_V, conf.raw_scale_I)
    scale_R = get_scale_factor(conf.scale_y, "none") / default_scale_R
    plt.plot(
        voltage[1:-1] * scale_V,
        resistance * scale_R,
        color=plot_conf.color,
        label=plot_conf.label)


def plot_ohmicr(conf: QHConfig) -> None:
    plot(conf, _plot_ohmicr, "V", "V", "R", r"\Omega")
