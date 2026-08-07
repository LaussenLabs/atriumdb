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

from pathlib import Path

import os
import pytest
from dotenv import load_dotenv

from atriumdb.sql_handler.maria.maria_handler import MariaDBHandler
from atriumdb.sql_handler.sqlite.sqlite_handler import SQLiteHandler
from tests.testing_framework import maria_db_credentials

load_dotenv()

# Read through the shared helper: it returns None instead of raising when the
# MariaDB variables are absent, so a run without them degrades to SQLite-only
# rather than dying at import and taking the whole collection down with it.
_maria_credentials = maria_db_credentials() or {}
host = _maria_credentials.get("host")
user = _maria_credentials.get("user")
password = _maria_credentials.get("password")
port = _maria_credentials.get("port")

# Every test here builds a MariaDBHandler directly, so the file as a whole
# requires MariaDB and is deselected by ``-m "not mariadb"``.
pytestmark = pytest.mark.mariadb

DB_NAME = 'settings'

SQLITE_FILE = Path(__file__).parent / DB_NAME / 'meta' / 'index.db'


def test_setting_insert_select():
    maria_handler = MariaDBHandler(host, user, password, DB_NAME, port=port)
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
