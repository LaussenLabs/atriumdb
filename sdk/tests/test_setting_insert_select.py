from pathlib import Path

import os
from dotenv import load_dotenv

from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler
from atriumdb.sql_handler.sqlite.sqlite_handler import SQLiteHandler

load_dotenv()

# Get MariaDB connection details from .env file
host = os.getenv("MARIA_DB_HOST")
user = os.getenv("MARIA_DB_USER")
password = os.getenv("MARIA_DB_PASSWORD")
port = int(os.getenv("MARIA_DB_PORT"))

DB_NAME = 'settings'

SQLITE_FILE = Path(__file__).parent / DB_NAME / 'meta' / 'index.db'


def test_setting_insert_select():
    maria_handler = MariaDBHandler(host, user, password, DB_NAME)
    maria_handler.maria_connect_no_db().cursor().execute(f"DROP DATABASE IF EXISTS {DB_NAME}")
    maria_handler.create_schema()
    _test_setting_insert_select(maria_handler)

    SQLITE_FILE.unlink(missing_ok=True)
    SQLITE_FILE.parent.mkdir(parents=True, exist_ok=True)
    sqlite_handler = SQLiteHandler(SQLITE_FILE)
    sqlite_handler.create_schema()
    _test_setting_insert_select(sqlite_handler)


def _test_setting_insert_select(sql_handler):
    for i in range(5):
        setting_name = f"setting_{i}"
        setting_value = f"value_{i}"
        sql_handler.insert_setting(setting_name, setting_value)

    for i in range(5):
        setting_name = f"setting_{i}"
        setting_value = f"value_{i}"
        setting = sql_handler.select_setting(setting_name)
        assert setting == (setting_name, setting_value)
