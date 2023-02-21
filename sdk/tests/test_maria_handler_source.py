import os
import pytest
import mariadb
from dotenv import load_dotenv

from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler
from atriumdb.sql_handler.sql_constants import DEFAULT_UNITS

load_dotenv()

# Get MariaDB connection details from .env file
host = os.getenv("MARIA_DB_HOST")
user = os.getenv("MARIA_DB_USER")
password = os.getenv("MARIA_DB_PASSWORD")
port = int(os.getenv("MARIA_DB_PORT"))

@pytest.fixture(scope="module")
def sqlite_handler():
    handler = MariaDBHandler(host, user, password, "measure_test", port)
    handler.maria_connect_no_db().cursor().execute("DROP DATABASE IF EXISTS measure_test")
    handler.create_schema()
    yield handler


def test_select_insert_source_info(sqlite_handler):
    # Insert measure
    # Test Data
    # mariadb_handler = MariaDBHandler(host, user, password, "measure_test", port)
    # mariadb_handler.create_schema()

    test_measure_tag = "test_tag"
    test_freq_nhz = 1000
    example_units = "test_units"
    test_measure_name = "test_name"
    test_id = None

    measure_id = sqlite_handler.insert_measure(test_measure_tag, test_freq_nhz, example_units, test_measure_name)

    # Select inserted measure
    result = sqlite_handler.select_measure(measure_id=measure_id)

    assert result[1] == test_measure_tag
    assert result[2] == test_measure_name
    assert result[3] == test_freq_nhz
    assert result[4] == example_units

    result = sqlite_handler.select_measure(measure_tag=test_measure_tag, freq_nhz=test_freq_nhz, units=example_units)

    assert result[1] == test_measure_tag
    assert result[2] == test_measure_name
    assert result[3] == test_freq_nhz
    assert result[4] == example_units

    # Test for case where units=None
    example_units = None
    measure_id = sqlite_handler.insert_measure(test_measure_tag, test_freq_nhz, example_units, test_measure_name)
    result = sqlite_handler.select_measure(measure_id=measure_id)

    assert result[1] == test_measure_tag
    assert result[2] == test_measure_name
    assert result[3] == test_freq_nhz
    assert result[4] == DEFAULT_UNITS

    # Select on triplet
    result = sqlite_handler.select_measure(measure_tag=test_measure_tag, freq_nhz=test_freq_nhz, units=example_units)

    assert result[1] == test_measure_tag
    assert result[2] == test_measure_name
    assert result[3] == test_freq_nhz
    assert result[4] == DEFAULT_UNITS

    # Insert device
    test_device_tag = "test_device_tag"
    test_device_name = "test_device_name"

    device_id = sqlite_handler.insert_device(test_device_tag, test_device_name)

    # Select inserted device
    result = sqlite_handler.select_device(device_id=device_id)

    assert result[1] == test_device_tag
    assert result[2] == test_device_name

    # Test for case where name=None
    test_device_tag = "test_device_tag_2"
    device_id = sqlite_handler.insert_device(test_device_tag, test_device_name)
    result = sqlite_handler.select_device(device_id=device_id)

    assert result[1] == test_device_tag
    assert result[2] == test_device_name
