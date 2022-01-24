import typing as t

import pydantic


Scales = t.Literal[
    "pico", "nano", "micro", "mili",
    "none", "kilo", "mega", "giga", "tera"]


def scale2alpha(scale: Scales) -> str:
    if scale == "pico":
        return "p"
    elif scale == "nano":
        return "n"
    elif scale == "micro":
        return "\mu"
    elif scale == "mili":
        return "m"
    elif scale == "none":
        return ""
    elif scale == "kilo":
        return "k"
    elif scale == "mega":
        return "M"
    elif scale == "giga":
        return "G"
    elif scale == "tera":
        return "T"


def get_scale_factor(target: Scales, base: Scales) -> float:
    scales = Scales.__args__
    power = 3 * (scales.index(base) - scales.index(target))
    return 10**power


class PlotConfig(pydantic.BaseModel):

    file_V: str
    """Path to the file of voltage data."""

    file_I: str
    """Path to the file of current data."""

    color: t.Optional[str] = None
    """Color of the plot."""

    label: t.Optional[str] = None
    """Label or legend of the plot."""

    remove_offset: bool = True
    """If the offset current is removed or not."""


class ImageSaveConfig(pydantic.BaseModel):

    output: str
    """Path to the image file to be generated."""

    dpi: int = 300
    """dpi of the image."""

    tight: bool = True
    """If the bounding box of the image is tightened or not."""


class QHConfig(pydantic.BaseModel):

    plots: t.List[PlotConfig]
    """Properties of the plots."""

    img: ImageSaveConfig
    """Config for saving image of the plot."""

    fontsize: t.Optional[int] = None
    """Font size of all charactors."""

    title: t.Optional[str] = None
    """Title of the plot."""

    fontsize_title: t.Optional[int] = None
    """Font size of the title."""

    xaxis: t.Optional[str] = None
    """Name of the x-axis."""

    fontsize_xaxis: t.Optional[int] = None
    """Font size of the x-axis."""

    yaxis: t.Optional[str] = None
    """Name of the y-axis."""

    fontsize_yaxis: t.Optional[int] = None
    """Font size of the y-axis."""

    raw_scale_V: Scales = "none"
    """Scale of the voltage of raw data."""

    raw_scale_I: Scales = "none"
    """Scale of the current of raw data."""

    scale_x: Scales = "mili"
    """Scale of the x-axis to be plotted."""

    scale_y: Scales = "micro"
    """Scale of the y-axis to be plotted."""

    show_legend: bool = True
    """If the legend of the data is printed or not."""

    output_offset: t.Optional[str] = None
    """Path to the file in which the offset currents is to be output."""

    offset_tolerance: float = 1e-9
    """Tolerance in order to judge the zero voltage for its offset current."""
