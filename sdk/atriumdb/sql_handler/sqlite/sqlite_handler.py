import sqlite3
from pathlib import Path
from typing import Union

from atriumdb.sql_handler.sql_constants import DEFAULT_UNITS
from atriumdb.sql_handler.sql_handler import SQLHandler
from atriumdb.sql_handler.sqlite.sqlite_functions import sqlite_insert_ignore_measure_query, \
    sqlite_select_measure_from_triplet_query, sqlite_select_measure_from_id_query, sqlite_insert_ignore_device_query, \
    sqlite_select_device_from_tag_query, sqlite_select_device_from_id_query
from atriumdb.sql_handler.sqlite.sqlite_tables import sqlite_measures_table_create_query, \
    sqlite_devices_table_create_query, sqlite_file_index_create_query, sqlite_block_index_create_query, \
    sqlite_interval_index_create_query, sqlite_encounter_create_query, sqlite_settings_create_query, \
    sqlite_file_index_idx_query, sqlite_block_index_idx_query, sqlite_interval_index_idx_query, \
    sqlite_encounter_index_idx_query


class SQLiteHandler(SQLHandler):
    def __init__(self, db_file: Union[str, Path]):
        self.db_file = db_file

    def sqlite_connect(self):
        conn = sqlite3.connect(self.db_file)

        return conn

    def create_schema(self):
        conn = self.sqlite_connect()
        cursor = conn.cursor()

        # Create Tables
        cursor.execute(sqlite_measures_table_create_query)
        cursor.execute(sqlite_devices_table_create_query)
        cursor.execute(sqlite_file_index_create_query)
        cursor.execute(sqlite_block_index_create_query)
        cursor.execute(sqlite_interval_index_create_query)
        cursor.execute(sqlite_encounter_create_query)
        cursor.execute(sqlite_settings_create_query)

        # Create Indices
        cursor.execute(sqlite_file_index_idx_query)
        cursor.execute(sqlite_block_index_idx_query)
        cursor.execute(sqlite_interval_index_idx_query)
        cursor.execute(sqlite_encounter_index_idx_query)

        conn.commit()
        cursor.close()
        conn.close()

    def insert_measure(self, measure_tag: str, freq_nhz: int, units: str = None, measure_name: str = None):
        units = DEFAULT_UNITS if units is None else units

        conn = self.sqlite_connect()
        cursor = conn.cursor()

        try:
            cursor.execute(sqlite_insert_ignore_measure_query, (measure_tag, freq_nhz, units, measure_name))
            conn.commit()
            cursor.execute(sqlite_select_measure_from_triplet_query, (measure_tag, freq_nhz, units))
            measure_id = cursor.fetchone()[0]
        except sqlite3.Error as e:
            raise e
        finally:
            cursor.close()
            conn.commit()
            conn.close()
        return measure_id


    def select_measure(self, measure_id: int = None, measure_tag: str = None, freq_nhz: int = None, units: str = None):
        units = DEFAULT_UNITS if units is None else units
        conn = self.sqlite_connect()
        cursor = conn.cursor()

        try:
            if measure_id is not None:
                cursor.execute(sqlite_select_measure_from_id_query, (measure_id,))
            else:
                cursor.execute(sqlite_select_measure_from_triplet_query,
                               (measure_tag, freq_nhz, units))
            row = cursor.fetchone()
        except sqlite3.Error as e:
            raise e
        finally:
            cursor.close()
            conn.close()
        return row

    def insert_device(self, device_tag: str, device_name: str = None):
        conn = self.sqlite_connect()
        cursor = conn.cursor()

        try:
            cursor.execute(sqlite_insert_ignore_device_query, (device_tag, device_name))
            conn.commit()
            cursor.execute(sqlite_select_device_from_tag_query, (device_tag,))
            device_id = cursor.fetchone()[0]
        except sqlite3.Error as e:
            raise e
        finally:
            cursor.close()
            conn.commit()
            conn.close()
        return device_id

    def select_device(self, device_id: int = None, device_tag: str = None):
        conn = self.sqlite_connect()
        cursor = conn.cursor()

        try:
            if device_id is not None:
                cursor.execute(sqlite_select_device_from_id_query, (device_id,))
            else:
                cursor.execute(sqlite_select_device_from_tag_query, (device_tag,))
            row = cursor.fetchone()
        except sqlite3.Error as e:
            raise e
        finally:
            cursor.close()
            conn.close()
        return row



