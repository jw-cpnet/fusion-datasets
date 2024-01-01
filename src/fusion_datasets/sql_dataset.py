from typing import Union, Tuple, List
from urllib.parse import urlparse

import clickhouse_connect
import pandas as pd
from kedro_datasets.pandas.sql_dataset import SQLTableDataset as orig


class SQLTableDataset(orig):
    def _parse_connection_string(self):

        # Parse the connection string using urlparse
        parsed = urlparse(self._connection_str)

        # Extract the components
        host = parsed.hostname
        username = parsed.username
        password = parsed.password
        database = parsed.path.lstrip("/")  # Remove leading slash
        port = (
            int(parsed.port) if parsed.port else None
        )  # Convert to int, default None if not present

        return {
            "host": host,
            "username": username,
            "password": password,
            "database": database,
            "port": port,
        }

    def _get_clickhouse_client(self):
        # Parse the connection string
        connection_details = self._parse_connection_string()
        if "schema" in self._save_args:
            connection_details["database"] = self._save_args["schema"]
        client = clickhouse_connect.get_client(**connection_details)
        return client

    def _save(self, data: Union[pd.DataFrame, str, Tuple[str], List[str]]) -> None:
        if "clickhouse" in self._connection_str:
            client = self._get_clickhouse_client()
            match data:
                case str():
                    # used for creating one table
                    client.command(data)
                case (tuple() | list()):
                    # used for creating multiple tables
                    for sql in data:
                        client.command(sql)
                case _:
                    client.insert_df(self._save_args["name"], data)
        else:
            data.to_sql(con=self.engine, **self._save_args)
