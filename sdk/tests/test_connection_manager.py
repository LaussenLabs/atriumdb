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
from atriumdb import AtriumSDK
from tests.testing_framework import _test_for_both
import mariadb
import time
import threading

DB_NAME = 'connection_manager'


def test_connection_manager():
    _test_for_both(DB_NAME, _test_connection_manager)


def _test_connection_manager(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)

    # Test only mariadb
    if db_type == "sqlite":
        return

    # SQLHandler Object
    handler = sdk.sql_handler

    # Test getting a connection
    with handler.connection() as (conn, cursor):
        assert conn is not None, "Connection should be established"
        assert cursor is not None, "Cursor should be received"

    # Test connection reuse
    initial_connection = handler.connection_manager.get_connection()
    handler.connection_manager.release_connection()
    reused_connection = handler.connection_manager.get_connection()
    assert reused_connection == initial_connection, "Connection should be reused if not closed"
    handler.connection_manager.release_connection()

    # Simulate inactivity to test ping and reconnection mechanism
    time.sleep(0.6)  # 600 ms to exceed the 500 ms threshold

    # Connection should still work
    with handler.connection() as (conn, cursor):
        assert conn is not None, "Connection should be established"
        assert cursor is not None, "Cursor should be received"

    # Close connection
    handler.connection_manager.close_connection()

    # Reopen, should be a different connection object
    new_connection = handler.connection_manager.get_connection()
    assert new_connection != initial_connection, "New connection should be created after inactivity"
    handler.connection_manager.release_connection()

    # Test transaction management
    with handler.connection(begin=True) as (conn, cursor):
        cursor.execute("CREATE TABLE IF NOT EXISTS test_table (id INT PRIMARY KEY, value TEXT)")
        cursor.execute("INSERT INTO test_table (id, value) VALUES (1, 'first')")
        cursor.execute("INSERT INTO test_table (id, value) VALUES (2, 'second')")

    with handler.connection() as (conn, cursor):
        cursor.execute("SELECT COUNT(*) FROM test_table")
        result = cursor.fetchone()
        assert result[0] == 2, "Two rows should be inserted into test_table"

    # Test rollback mechanism
    try:
        with handler.connection(begin=True) as (conn, cursor):
            cursor.execute("INSERT INTO test_table (id, value) VALUES (1, 'duplicate')")
    except mariadb.Error:
        pass  # Expected due to primary key constraint

    with handler.connection() as (conn, cursor):
        cursor.execute("SELECT COUNT(*) FROM test_table")
        result = cursor.fetchone()
        assert result[0] == 2, "No row should be inserted due to rollback"

    # Clean up
    with handler.connection() as (conn, cursor):
        cursor.execute("DROP TABLE IF EXISTS test_table")

    # Test closing of connection
    handler.connection_manager.close_connection()
    assert handler.connection_manager._connection is None, "Connection should be closed"
