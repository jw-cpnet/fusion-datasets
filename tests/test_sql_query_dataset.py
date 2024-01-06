from unittest.mock import patch, MagicMock, PropertyMock

import pytest

# Assuming the SQLQueryDataset class is in a module named `dataset_module`
from fusion_datasets.sql_dataset import (
    SQLQueryDataset,
    OriginalSQLQueryDataset,
    SSHTunnelForwarder,
)


@pytest.fixture
def ssh_credentials():
    return {
        "ssh_address": "example.com",
        "ssh_port": 22,
        "ssh_username": "user",
        "ssh_pkey": "/path/to/private/key",
        "remote_bind_address": "remote.example.com",
        "remote_bind_port": 12345,
    }


@pytest.fixture
def connection_str():
    return (
        "mssql+pyodbc:///?odbc_connect=DRIVER={{ODBC Driver 17 for SQL Server}};SERVER=127.0.0.1,"
        "{port};DATABASE=;UID=user;PWD=password"
    )


@pytest.fixture
def sql_query():
    return "SELECT * FROM table"


@pytest.fixture
def sql_query_dataset(sql_query, ssh_credentials, connection_str):
    return SQLQueryDataset(
        sql=sql_query,
        credentials={"con": connection_str},
        ssh_credentials=ssh_credentials,
    )


@pytest.fixture
def sql_query_dataset_without_ssh(sql_query, connection_str):
    return OriginalSQLQueryDataset(
        sql=sql_query,
        credentials={"con": connection_str},
    )


def test_create_connection_without_ssh(sql_query_dataset_without_ssh, connection_str):
    with patch.object(
        OriginalSQLQueryDataset, "create_connection"
    ) as mock_super_create_connection:
        sql_query_dataset_without_ssh.create_connection(connection_str)
        mock_super_create_connection.assert_called_once_with(connection_str)


def test_create_connection_with_ssh(sql_query_dataset, connection_str):
    with (
        patch.object(SSHTunnelForwarder, "start") as mock_ssh_tunnel_start,
        patch(
            "sshtunnel.SSHTunnelForwarder.local_bind_port",
            new_callable=PropertyMock,
        ) as mock_local_bind_port,
    ):
        mock_local_bind_port.return_value = 12345

        sql_query_dataset.create_connection(connection_str)
        mock_ssh_tunnel_start.assert_called_once()


def test_load_with_ssh(sql_query_dataset):
    with (
        patch.object(SSHTunnelForwarder, "start") as mock_ssh_tunnel_start,
        patch.object(SSHTunnelForwarder, "stop") as mock_ssh_tunnel_stop,
        patch(
            "sshtunnel.SSHTunnelForwarder.local_bind_port",
            new_callable=PropertyMock,
        ) as mock_local_bind_port,
        patch("pandas.read_sql_query") as mock_read_sql_query,
    ):
        mock_local_bind_port.return_value = 12345
        mock_read_sql_query.return_value = MagicMock()

        sql_query_dataset.load()
        mock_ssh_tunnel_start.assert_called_once()
        mock_ssh_tunnel_stop.assert_called_once()
