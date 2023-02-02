import os
import pytest
import mariadb
from dotenv import load_dotenv

from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler

load_dotenv()

# Get MariaDB connection details from .env file
host = os.getenv("MARIA_DB_HOST")
user = os.getenv("MARIA_DB_USER")
password = os.getenv("MARIA_DB_PASSWORD")
port = int(os.getenv("MARIA_DB_PORT"))

@pytest.fixture(scope="module")
def mariadb_handler():
    handler = MariaDBHandler(host, user, password, "measure_test", port)
    handler.create_schema()
    yield handler
    handler.maria_connect().cursor().execute("DROP DATABASE measure_test")


def test_select_insert_measure(mariadb_handler):
    # Insert measure
    # Test Data
    # mariadb_handler = MariaDBHandler(host, user, password, "measure_test", port)
    # mariadb_handler.create_schema()

    test_measure_tag = "test_tag"
    test_freq_nhz = 1000
    example_units = "test_units"
    test_measure_name = "test_name"
    test_id = None

    measure_id = mariadb_handler.insert_measure(test_measure_tag, test_freq_nhz, example_units, test_measure_name)

    # Select inserted measure
    result = mariadb_handler.select_measure(measure_id=measure_id)

    print(result)
    print(type(result))

    # Test for case where units=None
    example_units = None
    measure_id = mariadb_handler.insert_measure(test_measure_tag, test_freq_nhz, example_units, test_measure_name)
    result = mariadb_handler.select_measure(measure_id=measure_id)
    print(result)
    print(type(result))
    # assert result[1] == test_measure_tag
    # assert result[2] == test_freq_nhz
    # assert result[3] is None
    # assert result[4] == test_measure_name
