# AtriumDB is a timeseries database software designed to best handle the unique features and
# challenges that arise from clinical waveform data.
#     Copyright (C) 2023  The Hospital for Sick Children
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.


import pytest
import mariadb
from unittest.mock import patch, MagicMock

from atriumdb.sql_handler.maria.maria_handler import SingleConnectionManager


def test_connection_failure():
    conn_args = {
        'user': 'test_user',
        'password': 'test_password',
        'host': 'localhost',
        'database': 'test_db'
    }
    manager = SingleConnectionManager(**conn_args)

    # Simulate initial connection creation
    with patch('mariadb.connect') as mock_connect:
        mock_connect.side_effect = mariadb.Error("Initial connection failed")
        with pytest.raises(mariadb.Error, match="Initial connection failed"):
            manager.get_connection()

    # Simulate disconnection and failure to reconnect
    with patch('mariadb.connect') as mock_connect:
        mock_connect.side_effect = None  # First connection succeeds
        mock_connect.return_value.ping.side_effect = mariadb.Error("Ping failed")
        manager._create_connection()  # Create initial connection

        # Connection is lost
        mock_connect.side_effect = mariadb.Error("Reconnection failed")
        with pytest.raises(mariadb.Error, match="Reconnection failed"):
            manager.get_connection()


def test_normal_operation():
    conn_args = {
        'user': 'test_user',
        'password': 'test_password',
        'host': 'localhost',
        'database': 'test_db'
    }
    manager = SingleConnectionManager(**conn_args)

    # Simulate successful connection and ping
    with patch('mariadb.connect') as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        conn = manager.get_connection()
        assert conn is not None
        assert conn == mock_conn

        # Simulate releasing the connection
        manager.release_connection()
        assert not manager._borrowed

        # Simulate closing the connection
        manager.close_connection()
        assert manager._connection is None
        assert not manager._borrowed


