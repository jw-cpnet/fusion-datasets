import copy
import re
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import clickhouse_connect
import numpy as np
import pandas as pd
from jinja2 import Environment, meta
from kedro.io.core import AbstractDataset, get_filepath_str
from kedro_datasets.pandas.sql_dataset import SQLQueryDataset as OriginalSQLQueryDataset
from kedro_datasets.pandas.sql_dataset import SQLTableDataset as OriginalSQLTableDataset
from sshtunnel import SSHTunnelForwarder


class SQLTableDataset(OriginalSQLTableDataset):  # noqa: D101
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

            load_args["sql"] = load_args["sql"].rstrip().rstrip(";")

            return pd.read_sql_query(
                con=self.engine.execution_options(**self._execution_options),
                **load_args,
            )
        finally:
            if self._ssh_tunnel:
                self._ssh_tunnel.stop()
                self._ssh_tunnel = None


class ClickHouseTablesDataset(AbstractDataset):
    def __init__(
        self,
        credentials: Dict[str, Any],
        order_by: str,
        timezone: Optional[str] = None,
        database: Optional[str] = None,
        overwrite: Optional[bool] = False,
    ) -> None:
        if not (credentials and "host" in credentials and credentials["host"]):
            raise ValueError(
                "'host' argument cannot be empty. Please "
                "provide a ClickHouse connection string."
            )

        self._database = database if database else "default"
        self._order_by = order_by
        self._timezone = timezone
        self._credentials = credentials
        self._overwrite = overwrite
        self._client = clickhouse_connect.get_client(**credentials)

    def _describe(self) -> Dict[str, Any]:
        return {
            "database": self._database,
            "order_by": self._order_by,
        }

    def _load(self) -> Dict[str, pd.DataFrame]:
        tables = self._client.query(f"SHOW TABLES FROM {self._database}").result_columns
        if tables:
            tables = tables[0]
        data = {}
        for table_name in tables:
            df = self._client.query_df(f"SELECT * FROM {self._database}.{table_name}")
            data[table_name] = df
        return data

    def _save(self, data: Dict[str, List[pd.DataFrame]]) -> None:
        self._create_database_if_not_exists()
        for table_name, dfs in data.items():
            if self._overwrite and self._query_table_exists(table_name):
                self._client.command(f"DROP TABLE {self._database}.{table_name}")
            for i, df in enumerate(dfs):
                self._create_or_alter_table(table_name, df, dfs)
                self._client.insert_df(f"{self._database}.{table_name}", df)

    def _exists(self) -> bool:
        db = self._database
        query_db = self._client.command(f"EXISTS DATABASE {db}")
        query_table = self._client.query(f"show tables from {db}").result_rows
        return query_db and len(query_table) > 0

    def _query_table_exists(self, table_name: str) -> bool:
        return self._client.command(f"EXISTS TABLE {self._database}.{table_name}")

    def _create_database_if_not_exists(self) -> None:
        self._client.command(f"CREATE DATABASE IF NOT EXISTS {self._database}")

    def _get_merged_type(self, dataframes: List[pd.DataFrame], column: str) -> str:
        """
        Determines the most appropriate ClickHouse type for a given column across multiple dataframes.
        """
        # Concatenate the dataframes, then get its dtype
        merged_df = pd.concat(
            [df[[column]] for df in dataframes if column in df.columns],
            ignore_index=True,
        )
        dtype = self._get_clickhouse_type(merged_df[column].dtype)
        if column == self._order_by and "Nullable" in dtype:
            dtype = dtype.replace("Nullable(", "").replace(")", "")

        # Use the resulting dtype to get the ClickHouse type
        return dtype

    def _create_or_alter_table(
        self, table_name: str, df: pd.DataFrame, dfs: List[pd.DataFrame]
    ) -> None:
        if not self._query_table_exists(table_name):
            columns = ",\n".join(
                [
                    f"`{col}` {self._get_merged_type(dfs, col)}"
                    for col in set(col for df in dfs for col in df.columns)
                ]
            )
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self._database}.{table_name}
            (
                {columns}
            )
            ENGINE = MergeTree
            ORDER BY `{self._order_by}`
            """
            self._client.command(create_table_sql)
        else:
            existing_columns = self._client.query_df(
                f"DESCRIBE TABLE {self._database}.{table_name}"
            ).name.tolist()
            for col, dtype in df.dtypes.items():
                if col not in existing_columns:
                    alter_table_sql = (
                        f"ALTER TABLE {self._database}.{table_name} "
                        f"ADD COLUMN `{col}` {self._get_clickhouse_type(dtype)}"
                    )
                    self._client.command(alter_table_sql)

    def _get_clickhouse_type(self, dtype) -> str:
        if pd.api.types.is_integer_dtype(dtype):
            return "Nullable(Int32)"
        elif pd.api.types.is_float_dtype(dtype):
            return "Nullable(Float64)"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            if self._timezone:
                return f"DateTime('{self._timezone}')"
            return f"DateTime"
        elif pd.api.types.is_bool_dtype(dtype):
            return "Nullable(UInt8)"
        else:
            return "Nullable(String)"
