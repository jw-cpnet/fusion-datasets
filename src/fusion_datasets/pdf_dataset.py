from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any, Dict

import fsspec
import pandas as pd
import tabula
from kedro.io.core import (
    PROTOCOL_DELIMITER,
    AbstractVersionedDataset,
    Version,
    get_protocol_and_path,
)


class PDFDataset(AbstractVersionedDataset):
    def __init__(
        self,
        filepath: str,
        load_args: Dict[str, Any] = None,
        version: Version = None,
        credentials: Dict[str, Any] = None,
        fs_args: Dict[str, Any] = None,
    ):
        """Creates a new instance of PDFDataSet to load a PDF file as a pandas DataFrame.

        Args:
            filepath: The location of the PDF file.
            load_args: The arguments to pass to tabula.read_pdf() when loading the data.
            version: If specified, should be an instance of ``kedro.io.core.Version``.
            credentials: Credentials required to get access to the underlying filesystem.
            fs_args: Extra arguments to pass into underlying filesystem class constructor.
        """
        _fs_args = deepcopy(fs_args) or {}
        _credentials = deepcopy(credentials) or {}
        protocol, path = get_protocol_and_path(filepath, version)
        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)
        self._protocol = protocol
        self._storage_options = {**_credentials, **_fs_args}
        self._fs = fsspec.filesystem(self._protocol, **self._storage_options)
        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )
        self._load_args = load_args if load_args is not None else dict()

    def _load(self) -> pd.DataFrame:
        """Loads the PDF file as a pandas DataFrame using tabula.read_pdf()."""
        load_path = str(self._get_load_path())
        if self._protocol == "file":
            return tabula.read_pdf(load_path, **self._load_args)
        load_path = f"{self._protocol}{PROTOCOL_DELIMITER}{load_path}"
        with self._fs.open(load_path, mode="rb") as fs_file:
            return tabula.read_pdf(fs_file, **self._load_args)

    def _save(self, data: pd.DataFrame) -> None:
        """Raises an error to indicate that saving is not supported."""
        raise NotImplementedError("Saving data back to a PDF is not supported.")

    def _describe(self) -> Dict[str, Any]:
        """Returns a dictionary that describes the attributes of the dataset."""
        return dict(filepath=self._filepath, load_args=self._load_args)