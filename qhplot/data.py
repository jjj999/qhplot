from __future__ import annotations
from functools import cached_property
from pathlib import Path
import typing as t
from matplotlib.pyplot import axis

import numpy as np

from .const import hc
from .function import gaussian_dist
from .util import change_suffix


def load_raw_data(
    file: t.Union[str, Path],
    save_npy: bool = True,
) -> np.ndarray:
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

    @classmethod
    def cosmic_remove_calc(
        cls,
        counts: np.ndarray,
        threshold: int,
    ) -> np.ndarray:
        # TODO understand the code.
        indices, = np.where(np.abs(np.gradient(counts)) >= threshold)
        L=[]
        Ls=[]
        if indices.shape[0]>0:
            for n in range(indices.shape[0]-1):
                i1=indices[n]
                i2=indices[n+1]
                if (i2-i1)<=3:
                    Ls.append(i1)
                else:
                    Ls.append(i1)
                    L.append(Ls)
                    Ls=[]
            Ls.append(indices[-1])
        if len(Ls)>0:
            L.append(Ls)

        for l in L:
            min_l = min(l)
            max_l = max(l)
            avg=(counts[min_l - 1] + counts[max_l + 1]) / 2
            arg=np.arange(min_l,max_l+1)
            for i in arg:
                counts[i]=avg
        return counts

    @staticmethod
    def load(
        file_wavelength: t.Union[str, Path],
        file_counts: t.Union[str, Path],
        save_npy: bool = True,
    ) -> SPData:
        return SPData(
            load_raw_data(file_wavelength, save_npy=save_npy),
            load_raw_data(file_counts, save_npy=save_npy)
        )

    def __init__(self, wavelength: np.ndarray, counts: np.ndarray) -> None:
        self._wavelength = wavelength
        self._counts = counts

    @property
    def wavelength(self) -> np.ndarray:
        return self._wavelength

    @cached_property
    def energy(self) -> np.ndarray:
        return hc / (self._wavelength + 1e-12)

    @property
    def counts(self) -> np.ndarray:
        return self._counts

    @property
    def num_pixels(self) -> int:
        return len(self.wavelength)

    @property
    def num_measure(self) -> int:
        return len(self.counts)

    def remove_darkcounts(self, darkcounts: int) -> SPData:
        return SPData(self.wavelength, self.counts - darkcounts)

    def remove_cosmic_noise(self) -> SPData:
        counts_filtered = np.zeros_like(self.counts)
        for i in range(counts_filtered.shape[0]):
            counts_filtered[i, :] = self.cosmic_remove_calc(self.counts[i, :], 200)

        return SPData(self.wavelength, counts_filtered)

    def broad_fringe(self, std: float) -> SPData:
        x = np.arange(self.counts.shape[1])
        y = gaussian_dist(x, 669.5, std)
        counts = np.array([
            np.convolve(self.counts[n, :], y, mode="same")
            for n in range(self.num_measure)
        ])
        return SPData(self.wavelength, counts)

    def crop_pixel(self, min_: int, max_: int) -> SPData:
        _check_min_max(min_, max_)
        return SPData(
            self.wavelength[min_:max_ + 1],
            self.counts[:, min_:max_ + 1],
        )

    def crop_wavelength(self, min_: float, max_: float) -> SPData:
        _check_min_max(min_, max_)
        indices = _get_matching_range(self.wavelength, min_, max_)
        return SPData(self.wavelength[indices], self.counts[:, indices])

    def crop_energy(self, min_: float, max_: float) -> SPData:
        _check_min_max(min_, max_)
        indices = _get_matching_range(self.energy, min_, max_)
        return SPData(self.wavelength[indices], self.counts[:, indices])

    def sum_counts_pixel(self, min_: int, max_: int) -> np.ndarray:
        _check_min_max(min_, max_)
        return np.sum(self.counts[:, min_:max_ + 1], axis=1)

    def sum_counts_wavelength(self, min_: float, max_: float) -> np.ndarray:
        return np.sum(self.crop_wavelength(min_, max_).counts, axis=1)

    def sum_counts_energy(self, min_: float, max_: float) -> np.ndarray:
        return np.sum(self.crop_energy(min_, max_).counts, axis=1)
