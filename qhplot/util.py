import os.path
from pathlib import Path
import typing as t


def change_suffix(file: t.Union[str, Path], new_suffix: str) -> str:
    if isinstance(file, str):
        file = Path(file)

    filename, _ = file.name.rsplit(".", 2)
    new_filename = ".".join((filename, new_suffix))
    return os.path.join(file.parent, new_filename)
