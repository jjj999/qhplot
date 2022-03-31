import os.path
from pathlib import Path
import typing as t

import numpy as np


def change_suffix(file: t.Union[str, Path], new_suffix: str) -> str:
    if isinstance(file, str):
        file = Path(file)

    filename, _ = file.name.rsplit(".", 2)
    new_filename = ".".join((filename, new_suffix))
    return os.path.join(file.parent, new_filename)


def tie_neighbors(arr: t.Iterable[float], closeness: float) -> t.List[t.List[float]]:
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    if not len(arr):
        return []

    # Tie neighbors and make blocks of the neighbors.
    is_close = (arr[1:] - arr[:-1]) < closeness
    blocks = []
    block_neighbors = [arr[0]]
    for n, is_neighbor in zip(arr[1:], is_close):
        if is_neighbor:
            block_neighbors.append(n)
        else:
            blocks.append(block_neighbors)
            block_neighbors = [n]
    else:
        if len(block_neighbors):
            blocks.append(block_neighbors)

    return blocks


def smooth_outliers(
    arr: np.ndarray,
    threshold: float,
    closeness: float,
) -> np.ndarray:
    # Extract indices whose values are too big or too small
    # compared with the neighbors.
    indices, = np.where(np.abs(np.gradient(arr)) >= threshold)

    # Remove outliers and update the values as average values of
    # normal values in the neighborhoods.
    blocks = tie_neighbors(indices, closeness)
    for neighbors in blocks:
        index_min = min(neighbors)
        index_max = max(neighbors)
        avg_normal = (arr[index_min - 1] + arr[index_max + 1]) / 2
        arr[np.arange(index_min, index_max + 1)] = avg_normal

    return arr
