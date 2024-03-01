import copy
import re
from pathlib import PurePosixPath
from typing import Union, Tuple, List, Any
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import clickhouse_connect
import numpy as np
import pandas as pd
from jinja2 import Environment, meta
from kedro.io.core import get_filepath_str
from kedro_datasets.pandas.sql_dataset import SQLQueryDataset as OriginalSQLQueryDataset
from kedro_datasets.pandas.sql_dataset import SQLTableDataset as OriginalSQLTableDataset
from sshtunnel import SSHTunnelForwarder


class SQLTableDataset(OriginalSQLTableDataset):
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
                    client.insert_df(
                        self._save_args["name"], data.replace({np.nan: None})
                    )
        else:
            data.to_sql(con=self.engine, **self._save_args)


class SQLQueryDataset(OriginalSQLQueryDataset):
    def __init__(
        self,
        sql: str = None,
        credentials: dict[str, Any] = None,
        ssh_credentials: dict[str, Any] | None = None,
        j2_context: dict[str, Any] | None = None,
        load_args: dict[str, Any] = None,
        fs_args: dict[str, Any] = None,
        filepath: str = None,
        execution_options: dict[str, Any] | None = None,
        metadata: dict[str, Any] = None,
    ):
        super().__init__(
            sql=sql,
            credentials=credentials,
            load_args=load_args,
            fs_args=fs_args,
            filepath=filepath,
            execution_options=execution_options,
            metadata=metadata,
        )
        self._ssh_credentials = ssh_credentials
        self._ssh_tunnel = None
        self._j2_context = j2_context

    @staticmethod
    def _is_jinja2_template(s):
        env = Environment()
        ast = env.parse(s)
        variables = meta.find_undeclared_variables(ast)
        return bool(variables)

    @staticmethod
    def _encode_url_query(url):
        # Parse the URL into components
        url_parts = urlparse(url)

        # If there's no query string, return the original URL
        if not url_parts.query:
            return url

        # Parse the query string into a list of (name, value) pairs
        query_pairs = parse_qsl(url_parts.query)

        # Check if the query is already encoded
        if any("%" in pair[1] for pair in query_pairs):
            # If it's already encoded, use it as is
            encoded_query = url_parts.query
        else:
            # If it's not encoded, encode the query parameters
            encoded_query = urlencode(query_pairs)

        # Reconstruct the URL with the encoded query parameters
        encoded_url = urlunparse(
            (
                url_parts.scheme,
                url_parts.netloc,
                url_parts.path,
                url_parts.params,
                encoded_query,
                url_parts.fragment,
            )
        )

        # Use a regular expression to replace the slashes in the encoded URL with those in the original URL
        encoded_url = re.sub(
            r"(?<=:)/+",
            lambda m: m.group(0)
            * (url.count(m.group(0)) // encoded_url.count(m.group(0))),
            encoded_url,
        )

        return encoded_url

    def create_connection(self, connection_str: str, connection_args: dict) -> None:
        if self._ssh_credentials:
            ssh_credentials = self._ssh_credentials
            # Establish the SSH tunnel
            self._ssh_tunnel = SSHTunnelForwarder(
                (ssh_credentials["ssh_address"], ssh_credentials["ssh_port"]),
                ssh_username=ssh_credentials["ssh_username"],
                ssh_pkey=ssh_credentials["ssh_pkey"],  # Path to your private key file
                remote_bind_address=(
                    ssh_credentials["remote_bind_address"],
                    ssh_credentials["remote_bind_port"],
                ),
            )
            self._ssh_tunnel.start()

            # Adjust the connection string to connect to the dynamic local port
            local_port = self._ssh_tunnel.local_bind_port
            if "{port}" in connection_str:
                connection_str = connection_str.format(port=local_port)
            connection_str = self._encode_url_query(connection_str)
            self._connection_str = connection_str

        super().create_connection(connection_str, connection_args)

    def _render_sql_template(self, sql_template):
        if self._j2_context:
            env = Environment()
            context = self._j2_context
            sql = env.from_string(sql_template).render(**context)
        else:
            raise ValueError(
                "The SQL query is a Jinja2 template, but no context was provided."
            )
        return sql

    def _load(self) -> pd.DataFrame:
        try:
            load_args = copy.deepcopy(self._load_args)

            if self._filepath:
                load_path = get_filepath_str(
                    PurePosixPath(self._filepath), self._protocol
                )
                with self._fs.open(load_path, mode="r") as fs_file:
                    load_args["sql"] = fs_file.read()

            if self._is_jinja2_template(load_args["sql"]):
                load_args["sql"] = self._render_sql_template(load_args["sql"])

            return pd.read_sql_query(
                con=self.engine.execution_options(**self._execution_options),
                **load_args,
            )
        finally:
            if self._ssh_tunnel:
                self._ssh_tunnel.stop()
                self._ssh_tunnel = None
