import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

from atriumdb import AtriumSDK
from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler


def _test_for_both(db_name, test_function, *args):
    load_dotenv()

    # Get MariaDB connection details from .env file
    host = os.getenv("MARIA_DB_HOST")
    user = os.getenv("MARIA_DB_USER")
    password = os.getenv("MARIA_DB_PASSWORD")
    port = int(os.getenv("MARIA_DB_PORT"))

    maria_dataset_path = Path(__file__).parent / "test_datasets" / f"maria_{db_name}"
    sqlite_dataset_path = Path(__file__).parent / "test_datasets" / f"sqlite_{db_name}"

    db_type = 'mariadb'
    shutil.rmtree(maria_dataset_path, ignore_errors=True)
    maria_handler = MariaDBHandler(host, user, password, db_name)
    connection_params = {
        'sqltype': db_type,
        'host': host,
        'user': user,
        'password': password,
        'database': db_name,
        'port': 3306}
    maria_handler.maria_connect_no_db().cursor().execute(f"DROP DATABASE IF EXISTS `{db_name}`")
    test_function(db_type, maria_dataset_path, connection_params, *args)

    db_type = 'sqlite'
    connection_params = None
    shutil.rmtree(sqlite_dataset_path, ignore_errors=True)
    sqlite_dataset_path.unlink(missing_ok=True)
    test_function(db_type, sqlite_dataset_path, connection_params, *args)


def create_sibling_sdk(connection_params, dataset_location, db_type):
    dataset_location = str(dataset_location) + "_2"
    shutil.rmtree(dataset_location, ignore_errors=True)
    if db_type in ['mysql', 'mariadb']:
        connection_params['database'] += "-2"
        host = connection_params['host']
        user = connection_params['user']
        password = connection_params['password']
        db_name = connection_params['database']
        port = connection_params['port']

        maria_handler = MariaDBHandler(host, user, password, db_name)
        maria_handler.maria_connect_no_db().cursor().execute(f"DROP DATABASE IF EXISTS `{db_name}`")
    sdk_2 = AtriumSDK.create_dataset(
        dataset_location=dataset_location, database_type=db_type, connection_params=connection_params)
    return sdk_2
