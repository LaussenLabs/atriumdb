from pathlib import Path
from atriumdb import AtriumSDK

import os
import shutil
from dotenv import load_dotenv

from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler
from atriumdb.sql_handler.sqlite.sqlite_handler import SQLiteHandler

load_dotenv()

# Get MariaDB connection details from .env file
host = os.getenv("MARIA_DB_HOST")
user = os.getenv("MARIA_DB_USER")
password = os.getenv("MARIA_DB_PASSWORD")
port = int(os.getenv("MARIA_DB_PORT"))

DB_NAME = 'create_db'

MARIA_DATASET_PATH = Path(__file__).parent / f"maria_{DB_NAME}"
SQLITE_DATASET_PATH = Path(__file__).parent / f"sqlite_{DB_NAME}"


def test_create_dataset():
    db_type = 'mariadb'
    shutil.rmtree(MARIA_DATASET_PATH, ignore_errors=True)
    maria_handler = MariaDBHandler(host, user, password, DB_NAME)
    connection_params = {
        'host': host,
        'user': user,
        'password': password,
        'database': DB_NAME,
        'port': 3306}
    maria_handler.maria_connect_no_db().cursor().execute(f"DROP DATABASE IF EXISTS {DB_NAME}")
    _test_create_dataset(db_type, MARIA_DATASET_PATH, connection_params)

    db_type = 'sqlite'
    connection_params = None
    shutil.rmtree(SQLITE_DATASET_PATH, ignore_errors=True)
    SQLITE_DATASET_PATH.unlink(missing_ok=True)
    _test_create_dataset(db_type, SQLITE_DATASET_PATH, connection_params)


def _test_create_dataset(db_type, dataset_location, connection_params):
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)
    sdk = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)
