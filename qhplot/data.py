from __future__ import annotations
from functools import cached_property
from pathlib import Path
import typing as t
from matplotlib.pyplot import axis

import numpy as np

from .const import hc
from .function import gaussian_dist
from .util import (
    change_suffix,
    smooth_outliers,
)


def load_raw_data(
    file: t.Union[str, Path],
    save_npy: bool = True,
) -> np.ndarray:
    """Load given file and generate a Ndarray of the data.

    Args:
        file: Path to a file with experimental data.
        save_npy: Whether the function creates new npy files from given data.

    Returns:
        Ndarray of the data.
    """
    if isinstance(file, str):
        file = Path(file)

    if file.suffix == ".txt":
        arr = np.loadtxt(file)
        if save_npy:
            npy_file = change_suffix(file, "npy")
            np.save(npy_file, arr)
        return arr
    elif file.suffix == ".npy":
        return np.load(file)
    raise ValueError(
        f"Extension '{file.suffix[1:]}' cannot be used for SP data. "
        "Use a txt or npy file."
    )


def _check_min_max(min_: float, max_: float) -> None:
    if min_ > max_:
        raise ValueError(f"The argument 'min_' must bigger than 'max_'.")


def _get_matching_range(
    arr: np.ndarray,
    min_: float,
    max_: float,
) -> np.ndarray:
    return np.where((min_ <= arr) & (arr <= max_))[0]


class SPData:

    @staticmethod
    def load(
        file_wavelength: t.Union[str, Path],
        file_counts: t.Union[str, Path],
        save_npy: bool = True,
    ) -> SPData:
        """Load given files and generate new SPData object.

        Args:
            file_wavelength: Path to a file with wavelength profile.
            file_counts: Path to a file with CCD data.
            save_npy: Whether the function creates new npy files from given
                data.

        Returns:
            New SPData object with given data.
        """
        return SPData(
            load_raw_data(file_wavelength, save_npy=save_npy),
            load_raw_data(file_counts, save_npy=save_npy)
        )

    def __init__(self, wavelength: np.ndarray, counts: np.ndarray) -> None:
        """
        Args:
            wavelength: wavelength profile of a measurement.
            counts: 2-D array of CCD data.
        """
        self._wavelength = wavelength
        self._counts = counts

    @property
    def wavelength(self) -> np.ndarray:
        """Wavelength profile of the data."""
        return self._wavelength

    @cached_property
    def energy(self) -> np.ndarray:
        """Energy profile of the data."""
        return hc / (self._wavelength + 1e-12)

    @property
    def counts(self) -> np.ndarray:
        """CCD counts of the data."""
        return self._counts

    @property
    def num_pixels(self) -> int:
        """Number of CCD pixels."""
        return len(self.wavelength)

    @property
    def num_measure(self) -> int:
        """Number of exposure times."""
        return len(self.counts)

    def remove_darkcounts(self, darkcounts: int) -> SPData:
        """Remove dark counts from the data.

        Args:
            darkcounts: counts to be subtracted.

        Returns:
            New SPData object with same wavelength data and processed
            counts data.
        """
        return SPData(self.wavelength, self.counts - darkcounts)

    def remove_cosmic_noise(self, threshold: int, closeness: int = 3) -> SPData:
        """Remove effects of cosmic rays.

        Args:
            threshld: Threshold to judge each count are caused by cosmic rays.
            closeness: Width between indices that represents how close each
                index are.

        Returns:
            New SPData object with same wavelength data and processed
            counts data.
        """
        counts_cosmic_removed = np.apply_along_axis(
            smooth_outliers,
            1,
            self.counts,
            threshold,
            closeness,
        )
        return SPData(self.wavelength, counts_cosmic_removed)

    def filter_gaussian(self, mean: float, std: float) -> SPData:
        """Apply Gaussian filter to each spectrum for smoothing.

        Args:
            mean: Mean of the Gaussian.
            std: Standard deviation of the Gaussian.

        Returns:
            New SPData object with same wavelength data and processed
            counts data.
        """
        x = np.arange(self.counts.shape[1])
        y = gaussian_dist(x, mean, std)
        counts = np.apply_along_axis(
            np.convolve,
            1,
            self.counts,
            y,
            mode="same",
        )
        return SPData(self.wavelength, counts)

    def crop_pixel(self, min_: int, max_: int) -> SPData:
        """Crop a range of pixels.

        Args:
            min_: Minimal pixel value of the range.
            max_: Maximal pixel value of the range.

        Returns:
            New SPData object with the cropped data.
        """
        _check_min_max(min_, max_)
        return SPData(
            self.wavelength[min_:max_ + 1],
            self.counts[:, min_:max_ + 1],
        )

    def crop_wavelength(self, min_: float, max_: float) -> SPData:
        """Crop a range of wavelength.

        Args:
            min_: Minimal wavelength of the range.
            max_: Maximal wavelength of the range.

        Returns:
            New SPData object with the cropped data.
        """
        _check_min_max(min_, max_)
        indices = _get_matching_range(self.wavelength, min_, max_)
        return SPData(self.wavelength[indices], self.counts[:, indices])

    def crop_energy(self, min_: float, max_: float) -> SPData:
        """Crop a range of energy.

        Args:
            min_: Minimal energy of the range.
            max_: Maximal energy of the range.

        Returns:
            New SPData object with the cropped data.
        """
        _check_min_max(min_, max_)
        indices = _get_matching_range(self.energy, min_, max_)
        return SPData(self.wavelength[indices], self.counts[:, indices])

    def sum_counts_pixel(self, min_: int, max_: int) -> np.ndarray:
        """Sum all counts up every axes in given range of pixels.

        Args:
            min_: Minimal pixel of the range.
            max_: Maximal pixel of the range.

        Returns:
            Ndarray of the sums.
        """
        _check_min_max(min_, max_)
        return np.sum(self.counts[:, min_:max_ + 1], axis=1)

    def sum_counts_wavelength(self, min_: float, max_: float) -> np.ndarray:
        """Sum all counts up every axes in given range of wavelength.

        Args:
            min_: Minimal wavelength of the range.
            max_: Maximal wavelength of the range.

        Returns:
            Ndarray of the sums.
        """
        return np.sum(self.crop_wavelength(min_, max_).counts, axis=1)

    def sum_counts_energy(self, min_: float, max_: float) -> np.ndarray:
        """Sum all counts up every axes in given range of energy.

        Args:
            min_: Minimal energy of the range.
            max_: Maximal energy of the range.

        Returns:
            Ndarray of the sums.
        """
        return np.sum(self.crop_energy(min_, max_).counts, axis=1)
